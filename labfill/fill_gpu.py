"""Recipe 4: Fill One GPU — saturating protocol-model training under variable sequence lengths.

Addresses common causes of low single-GPU throughput on science/protocol tasks:
  (a) padded_baseline         : dynamic batch padding on lognormal 8..512 protocol sequences
  (b) length_bucketed         : pool + sort by sequence length before batching
  (c) sequence_packed         : bin-pack into fixed rows with reset position_ids + block-diagonal mask
  (d) varlen_nested_sdpa      : zero-pad flat tokens + PyTorch jagged nested_tensor SDPA
  (e) packed_bf16_autocast    : sequence packing + bfloat16 autocast (CPU AMX/AVX512 or GPU Tensor Cores)
  (f) packed_torch_compile    : torch.compile wrapper with honest fallback/error capture
  (g) sync_slow_tokenizer vs async_dataloader : DataLoader worker overlap vs main-thread stalls
  (h) block_sparse_local_global : tiled local-window + global-token attention (+ FlexAttention probe)
  (i) cuda_graphs             : static-shape CUDA Graph capture/replay (behind `if cuda`)
"""

from __future__ import annotations

import hashlib
import heapq
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from labfill.protocol import default_device

try:
    from flash_attn import flash_attn_varlen_func as _FLASH_ATTN_VARLEN_FUNC  # type: ignore[import-not-found]
except ImportError:
    _FLASH_ATTN_VARLEN_FUNC = None

_COMPILED_FLEX_ATTENTION: Any = None

VOCAB_SIZE = 256
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
NUM_ERROR_CLASSES = 8  # 0 = valid token, 1..7 = seeded error class
IGNORE_LABEL = -100
VARLEN_DISPATCH_THRESHOLD = 4096


# ---------------------------------------------------------------------------
# 1. Synthetic variable-length protocol sequence dataset
# ---------------------------------------------------------------------------


@dataclass
class ProtocolSequenceItem:
    tokens: list[int]
    labels: list[int]
    raw_text: str = ""


def generate_protocol_sequences(
    num_sequences: int = 256,
    min_len: int = 8,
    max_len: int = 512,
    lognormal_mean: float = 3.65,
    lognormal_sigma: float = 0.95,
    seed: int = 0,
) -> list[ProtocolSequenceItem]:
    """Generate variable-length tokenized protocol traces (lognormal lengths in [min_len, max_len])."""
    rng = np.random.default_rng(seed)
    raw_lens = rng.lognormal(mean=lognormal_mean, sigma=lognormal_sigma, size=num_sequences)
    lengths = np.clip(np.round(raw_lens).astype(int), min_len, max_len)

    items: list[ProtocolSequenceItem] = []
    op_names = ("PICK", "DROP", "XFER", "MIX", "INC", "THAW", "SEAL", "SPIN")

    for i in range(num_sequences):
        seq_len = int(lengths[i])
        toks = [BOS_TOKEN_ID]
        lbls = [0]
        words = ["BOS"]

        # Choose one corrupted step region in 80% of protocols (every i with i % 5 != 0)
        has_err = (i % 5) != 0
        err_cls = int((i % (NUM_ERROR_CLASSES - 1)) + 1) if has_err else 0
        err_pos = int(rng.integers(1, max(seq_len - 2, 2)))

        pos = 1
        step_k = 0
        while pos < seq_len - 1:
            op_idx = int((i + step_k * 3) % len(op_names))
            src_c = int((step_k + i) % 12)
            dst_c = int((step_k * 2 + 1 + i) % 12)
            param = int((step_k * 17 + i * 7) % 180)

            step_toks = [
                16 + op_idx,
                32 + src_c,
                48 + dst_c,
                64 + param,
                SEP_TOKEN_ID,
            ]
            for st_tok in step_toks:
                if pos >= seq_len - 1:
                    break
                is_err_tok = has_err and (err_pos <= pos < err_pos + 4)
                tok_val = (st_tok + (120 if is_err_tok else 0)) % VOCAB_SIZE
                tok_val = max(tok_val, 4)
                toks.append(tok_val)
                lbls.append(err_cls if is_err_tok else 0)
                words.append(f"{op_names[op_idx]}:{src_c}->{dst_c}@{param}")
                pos += 1
            step_k += 1

        toks.append(EOS_TOKEN_ID)
        lbls.append(0)
        words.append("EOS")
        items.append(ProtocolSequenceItem(tokens=toks, labels=lbls, raw_text=" ".join(words)))

    return items


class SlowTokenizerDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset that simulates CPU-bound protocol parsing/tokenization inside __getitem__."""

    def __init__(self, items: list[ProtocolSequenceItem], slow_work_iters: int = 250) -> None:
        self.items = items
        self.slow_work_iters = slow_work_iters

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        item = self.items[idx]
        # Deterministic CPU string/hash work (releases/uses worker processes cleanly)
        buf = item.raw_text.encode("utf-8")
        acc = 0
        for _ in range(self.slow_work_iters):
            digest = hashlib.blake2b(buf, digest_size=8).digest()
            acc ^= digest[0]
            buf = digest + buf[:64]
        if acc == -1:  # never True; keeps work live
            return torch.empty(0), torch.empty(0)
        return (
            torch.tensor(item.tokens, dtype=torch.int64),
            torch.tensor(item.labels, dtype=torch.int64),
        )


# ---------------------------------------------------------------------------
# 2. Batching, bucketing, sequence packing, and jagged varlen collation
# ---------------------------------------------------------------------------


@dataclass
class PreparedBatch:
    mode: str                         # 'dense', 'varlen', 'packed_varlen', 'block_sparse', 'flex_block_sparse'
    input_ids: Tensor                 # (B, S) or (T_real,)
    position_ids: Tensor              # (B, S) or (T_real,)
    labels: Tensor                    # (B, S) or (T_real,)
    seq_id: Tensor | None = None      # (B, S): 0=pad, 1..K=packed sequence id within row
    cu_seqlens: Tensor | None = None  # (N_seq + 1,) int32 (CUDA) / int64 (CPU) cumulative offsets
    max_seqlen: int = 0               # maximum single-sequence length within batch
    token_weights: Tensor | None = None  # optional per-token / curriculum weights matching labels shape
    block_mask: Any = None            # optional cached FlexAttention BlockMask for static packed rows
    seq_lens: list[int] = field(default_factory=list)
    real_tokens: int = 0
    total_tokens: int = 0

    def to(self, device: torch.device, non_blocking: bool = False) -> "PreparedBatch":
        cu_out: Tensor | None = None
        if self.cu_seqlens is not None:
            target_dtype = torch.int32 if device.type == "cuda" else self.cu_seqlens.dtype
            cu_out = self.cu_seqlens.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
        cached_bm = self.block_mask
        if cached_bm is not None:
            bm_kv = getattr(cached_bm, "kv_num_blocks", None)
            if bm_kv is not None and bm_kv.device != device:
                cached_bm = None
        return PreparedBatch(
            mode=self.mode,
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            position_ids=self.position_ids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            seq_id=self.seq_id.to(device, non_blocking=non_blocking) if self.seq_id is not None else None,
            cu_seqlens=cu_out,
            max_seqlen=self.max_seqlen,
            token_weights=(
                self.token_weights.to(device, non_blocking=non_blocking)
                if self.token_weights is not None
                else None
            ),
            block_mask=cached_bm,
            seq_lens=self.seq_lens,
            real_tokens=self.real_tokens,
            total_tokens=self.total_tokens,
        )


def build_cu_seqlens(
    seq_id_or_lens: Tensor | list[int] | tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    is_lengths: bool = False,
) -> tuple[Tensor, int]:
    """Build `(N_seq + 1,)` cumulative sequence offsets and `max_seqlen` in O(B * S) without dense masks.

    Accepts either:
      - `list[int]` / `tuple[int, ...]` of positive sequence lengths (or 1D Tensor with `is_lengths=True`), or
      - a `(B, S)` or `(S,)` `seq_id` tensor where contiguous runs of identical `seq_id > 0` within each row
        denote packed sequences (`0` = pad).
    """
    if isinstance(seq_id_or_lens, (list, tuple)) or is_lengths:
        lens = (
            [int(x) for x in seq_id_or_lens.tolist()]
            if isinstance(seq_id_or_lens, Tensor)
            else [int(x) for x in seq_id_or_lens]
        )
        for idx, ln in enumerate(lens):
            if ln <= 0:
                raise ValueError(f"Sequence {idx} length must be >= 1, got {ln}")
        dev = torch.device(device) if device is not None else torch.device("cpu")
        out_dtype = dtype if dtype is not None else (torch.int32 if dev.type == "cuda" else torch.int64)
        offsets = [0]
        acc = 0
        max_s = 0
        for ln in lens:
            acc += ln
            offsets.append(acc)
            if ln > max_s:
                max_s = ln
        return torch.tensor(offsets, dtype=out_dtype, device=dev), max_s

    if not isinstance(seq_id_or_lens, Tensor):
        raise TypeError(f"Expected Tensor or sequence of ints, got {type(seq_id_or_lens).__name__}")

    sid = seq_id_or_lens.unsqueeze(0) if seq_id_or_lens.ndim == 1 else seq_id_or_lens
    if sid.ndim != 2:
        raise ValueError(f"seq_id must be 1D or 2D, got shape {tuple(seq_id_or_lens.shape)}")

    dev = torch.device(device) if device is not None else sid.device
    out_dtype = dtype if dtype is not None else (torch.int32 if dev.type == "cuda" else torch.int64)

    _b, s = sid.shape
    active = sid > 0
    is_start = torch.zeros_like(sid, dtype=torch.bool)
    if s > 0:
        is_start[:, 0] = active[:, 0]
    if s > 1:
        is_start[:, 1:] = active[:, 1:] & (sid[:, 1:] != sid[:, :-1])

    starts_active = is_start[active]
    t_real = int(starts_active.numel())
    if t_real == 0:
        return torch.zeros((1,), dtype=out_dtype, device=dev), 0

    start_pos = torch.nonzero(starts_active, as_tuple=False).flatten()
    end_pos = torch.tensor([t_real], dtype=start_pos.dtype, device=sid.device)
    cu = torch.cat([start_pos, end_pos], dim=0)
    seg_lens = cu[1:] - cu[:-1]
    max_s = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
    return cu.to(device=dev, dtype=out_dtype), max_s


def chunk_sequence_pairs(
    pairs: list[tuple[Tensor, Tensor]],
    chunk_len: int,
) -> list[tuple[Tensor, Tensor]]:
    """Split oversized `(tokens, labels)` sequence pairs cleanly into contiguous chunks `<= chunk_len`."""
    if chunk_len < 1:
        raise ValueError(f"chunk_len must be >= 1, got {chunk_len}")
    chunked: list[tuple[Tensor, Tensor]] = []
    for seq_idx, (tok, lbl) in enumerate(pairs):
        raw_len = int(tok.numel())
        if int(lbl.numel()) != raw_len:
            raise ValueError(
                f"Sequence {seq_idx} token/label length mismatch: {raw_len} vs {int(lbl.numel())}"
            )
        if raw_len <= 0:
            raise ValueError(f"Sequence {seq_idx} must be non-empty, got length {raw_len}")
        for start in range(0, raw_len, chunk_len):
            end = min(start + chunk_len, raw_len)
            chunked.append((tok[start:end], lbl[start:end]))
    return chunked


def collate_padded_pairs(
    pairs: list[tuple[Tensor, Tensor]],
    pad_multiple: int = 1,
    max_seq_len: int | None = None,
) -> PreparedBatch:
    """Pad a list of (tokens, labels) tensors to the max length in the batch."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    b = len(pairs)
    lens: list[int] = []
    for seq_idx, (tok, lbl) in enumerate(pairs):
        n = int(tok.numel())
        if int(lbl.numel()) != n:
            raise ValueError(f"Sequence {seq_idx} token/label length mismatch: {n} vs {int(lbl.numel())}")
        if n <= 0:
            raise ValueError(f"Sequence {seq_idx} must have length >= 1, got {n}")
        if max_seq_len is not None and n > max_seq_len:
            raise ValueError(f"Sequence {seq_idx} length {n} exceeds max_seq_len={max_seq_len}")
        lens.append(n)

    max_len = max(lens)
    max_s = max_len
    if pad_multiple > 1:
        max_s = ((max_s + pad_multiple - 1) // pad_multiple) * pad_multiple

    input_ids = torch.full((b, max_s), PAD_TOKEN_ID, dtype=torch.int64)
    position_ids = torch.zeros((b, max_s), dtype=torch.int64)
    labels = torch.full((b, max_s), IGNORE_LABEL, dtype=torch.int64)
    seq_id = torch.zeros((b, max_s), dtype=torch.int64)

    for i, (tok, lbl) in enumerate(pairs):
        n = lens[i]
        input_ids[i, :n] = tok
        position_ids[i, :n] = torch.arange(n, dtype=torch.int64)
        labels[i, :n] = lbl
        seq_id[i, :n] = 1

    return PreparedBatch(
        mode="dense",
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        seq_id=seq_id,
        max_seqlen=max_len,
        seq_lens=lens,
        real_tokens=sum(lens),
        total_tokens=b * max_s,
    )


def _normalize_pack_inputs(
    pairs: list[tuple[Tensor, Tensor]],
    pack_len: int,
    rows_per_batch: int,
    attn_mode: str,
    truncate: bool,
    overflow_policy: str,
    curriculum_weights: list[float] | tuple[float, ...] | None,
) -> list[tuple[Tensor, Tensor, float | None]]:
    if pack_len < 1:
        raise ValueError(f"pack_len must be >= 1, got {pack_len}")
    if rows_per_batch < 1:
        raise ValueError(f"rows_per_batch must be >= 1, got {rows_per_batch}")
    valid_modes = ("dense", "packed_varlen", "varlen", "block_sparse", "flex_block_sparse")
    if attn_mode not in valid_modes:
        raise ValueError(f"Unsupported attn_mode={attn_mode!r}; expected one of {valid_modes}")

    effective_policy = "truncate" if (truncate and overflow_policy == "raise") else overflow_policy
    if effective_policy not in ("raise", "truncate", "chunk", "quarantine"):
        raise ValueError(
            f"Unsupported overflow_policy={effective_policy!r}; expected 'raise', 'truncate', 'chunk', or 'quarantine'"
        )
    if curriculum_weights is not None and len(curriculum_weights) != len(pairs):
        raise ValueError(
            f"curriculum_weights length {len(curriculum_weights)} != pairs length {len(pairs)}"
        )

    normalized_items: list[tuple[Tensor, Tensor, float | None]] = []
    for seq_idx, (tok, lbl) in enumerate(pairs):
        raw_len = int(tok.numel())
        if int(lbl.numel()) != raw_len:
            raise ValueError(
                f"Sequence {seq_idx} token/label length mismatch: {raw_len} vs {int(lbl.numel())}"
            )
        if raw_len <= 0:
            raise ValueError(f"Sequence {seq_idx} must have length >= 1, got {raw_len}")
        cw = float(curriculum_weights[seq_idx]) if curriculum_weights is not None else None
        if raw_len > pack_len:
            if effective_policy == "raise":
                raise ValueError(
                    f"Sequence {seq_idx} has length {raw_len} > pack_len={pack_len}; "
                    "pass truncate=True or overflow_policy='chunk'|'quarantine'"
                )
            if effective_policy == "truncate":
                normalized_items.append((tok[:pack_len], lbl[:pack_len], cw))
            elif effective_policy == "chunk":
                for start in range(0, raw_len, pack_len):
                    end = min(start + pack_len, raw_len)
                    normalized_items.append((tok[start:end], lbl[start:end], cw))
            elif effective_policy == "quarantine":
                continue
        else:
            normalized_items.append((tok, lbl, cw))
    return normalized_items


def _materialize_packed_bins(
    bins_entries: list[list[tuple[Tensor, Tensor, float | None]]],
    pack_len: int,
    rows_per_batch: int,
    attn_mode: str,
    has_curriculum_weights: bool,
) -> list[PreparedBatch]:
    batches: list[PreparedBatch] = []
    is_varlen_pack = attn_mode in ("packed_varlen", "varlen")

    for i in range(0, len(bins_entries), rows_per_batch):
        bin_group = bins_entries[i : i + rows_per_batch]
        flat_entries = [entry for b_list in bin_group for entry in b_list]
        flat_lens = [int(t.numel()) for t, _, _ in flat_entries]
        real_toks = sum(flat_lens)

        if is_varlen_pack:
            toks_1d = torch.cat([t for t, _, _ in flat_entries], dim=0)
            lbls_1d = torch.cat([lbl for _, lbl, _ in flat_entries], dim=0)
            pos_1d = torch.cat([torch.arange(ln, dtype=torch.int64) for ln in flat_lens], dim=0)
            cu_seq, max_s = build_cu_seqlens(flat_lens)
            tw_1d: Tensor | None = None
            if has_curriculum_weights:
                tw_1d = torch.cat(
                    [
                        torch.full((int(t.numel()),), float(cw or 0.0), dtype=torch.bfloat16)
                        for t, _, cw in flat_entries
                    ],
                    dim=0,
                )
            batches.append(
                PreparedBatch(
                    mode=attn_mode,
                    input_ids=toks_1d,
                    position_ids=pos_1d,
                    labels=lbls_1d,
                    seq_id=None,
                    cu_seqlens=cu_seq,
                    max_seqlen=max_s,
                    token_weights=tw_1d,
                    seq_lens=flat_lens,
                    real_tokens=real_toks,
                    total_tokens=real_toks,
                )
            )
        else:
            num_rows = len(bin_group)
            chunk_tok = torch.full((num_rows, pack_len), PAD_TOKEN_ID, dtype=torch.int64)
            chunk_pos = torch.zeros((num_rows, pack_len), dtype=torch.int64)
            chunk_lbl = torch.full((num_rows, pack_len), IGNORE_LABEL, dtype=torch.int64)
            chunk_sid = torch.zeros((num_rows, pack_len), dtype=torch.int64)
            chunk_tw = (
                torch.zeros((num_rows, pack_len), dtype=torch.bfloat16)
                if has_curriculum_weights
                else None
            )
            for r_idx, b_list in enumerate(bin_group):
                cursor = 0
                for sid_idx, (tok, lbl, cw) in enumerate(b_list, start=1):
                    n = int(tok.numel())
                    chunk_tok[r_idx, cursor : cursor + n] = tok
                    chunk_pos[r_idx, cursor : cursor + n] = torch.arange(n, dtype=torch.int64)
                    chunk_lbl[r_idx, cursor : cursor + n] = lbl
                    chunk_sid[r_idx, cursor : cursor + n] = sid_idx
                    if chunk_tw is not None:
                        chunk_tw[r_idx, cursor : cursor + n] = float(cw or 0.0)
                    cursor += n

            batches.append(
                PreparedBatch(
                    mode=attn_mode,
                    input_ids=chunk_tok,
                    position_ids=chunk_pos,
                    labels=chunk_lbl,
                    seq_id=chunk_sid,
                    max_seqlen=max(flat_lens, default=0),
                    token_weights=chunk_tw,
                    seq_lens=flat_lens,
                    real_tokens=real_toks,
                    total_tokens=int(chunk_tok.numel()),
                )
            )
    return batches


def pack_lengths_obfd(
    lengths: Sequence[int],
    bin_cap: int = 512,
    *,
    ordered_indices: Sequence[int] | None = None,
) -> list[list[tuple[int, int, int]]]:
    """Ding et al. (2024) `O(N log N)` Segment-Tree Best-Fit-Decreasing (`OBFD`) bin assignment.

    Maintains an iterative power-of-2 max-capacity segment tree over remaining capacities `1..bin_cap`
    with per-capacity min-heaps on bin creation index, returning exact BFD bin placements
    `[[(orig_idx, offset, length), ...], ...]` in `O(N log L + N log B)` time (zero `O(N^2)` bin scan).
    """
    if bin_cap < 1:
        raise ValueError(f"bin_cap must be >= 1, got {bin_cap}")
    if not lengths:
        return []

    int_lens: list[int] = []
    for i, raw_l in enumerate(lengths):
        ln = int(raw_l)
        if ln <= 0 or ln > bin_cap:
            raise ValueError(f"Sequence {i} length {ln} outside [1, {bin_cap}]")
        int_lens.append(ln)

    if ordered_indices is not None:
        order = [int(idx) for idx in ordered_indices]
    else:
        # O(N + bin_cap) stable counting sort in decreasing length order
        by_len: list[list[int]] = [[] for _ in range(bin_cap + 1)]
        for idx, ln in enumerate(int_lens):
            by_len[ln].append(idx)
        order = [idx for ln in range(bin_cap, 0, -1) for idx in by_len[ln]]

    size = 1
    while size <= bin_cap:
        size <<= 1
    tree = [0] * (2 * size)
    bins_by_rem: list[list[int]] = [[] for _ in range(bin_cap + 1)]
    bins_used: list[int] = []
    bins_slices: list[list[tuple[int, int, int]]] = []

    def _set_leaf(cap: int, active: bool) -> None:
        pos = size + cap
        tree[pos] = cap if active else 0
        pos >>= 1
        while pos >= 1:
            left_v = tree[pos << 1]
            right_v = tree[(pos << 1) | 1]
            new_v = left_v if left_v >= right_v else right_v
            if tree[pos] == new_v:
                break
            tree[pos] = new_v
            pos >>= 1

    for orig_idx in order:
        ln = int_lens[orig_idx]
        if tree[1] < ln:
            b_idx = len(bins_used)
            bins_used.append(ln)
            bins_slices.append([(orig_idx, 0, ln)])
            rem_after = bin_cap - ln
            if rem_after > 0:
                heap = bins_by_rem[rem_after]
                was_empty = not heap
                heapq.heappush(heap, b_idx)
                if was_empty:
                    _set_leaf(rem_after, True)
        else:
            # Walk down segment tree in exact log2(size) steps to minimum active capacity >= ln
            node = 1
            while node < size:
                left = node << 1
                if tree[left] >= ln:
                    node = left
                else:
                    node = left | 1
            best_rem = node - size
            heap = bins_by_rem[best_rem]
            b_idx = heapq.heappop(heap)
            if not heap:
                _set_leaf(best_rem, False)

            offset = bins_used[b_idx]
            bins_slices[b_idx].append((orig_idx, offset, ln))
            bins_used[b_idx] = offset + ln
            rem_after = best_rem - ln
            if rem_after > 0:
                heap_after = bins_by_rem[rem_after]
                was_empty = not heap_after
                heapq.heappush(heap_after, b_idx)
                if was_empty:
                    _set_leaf(rem_after, True)

    return bins_slices


def pack_sequence_pairs(
    pairs: list[tuple[Tensor, Tensor]],
    pack_len: int = 512,
    rows_per_batch: int = 4,
    attn_mode: str = "dense",
    truncate: bool = False,
    overflow_policy: str = "raise",
    curriculum_weights: list[float] | tuple[float, ...] | None = None,
) -> list[PreparedBatch]:
    """First-fit / greedy bin-pack sequences into fixed-capacity bins of `pack_len`.

    Resets `position_ids` to `0..len-1` per packed sequence. When `attn_mode == 'packed_varlen'`,
    emits unpadded 1D `(T_real,)` tensors with cumulative `cu_seqlens` and `max_seqlen <= pack_len`
    for zero-padding varlen FlashAttention / jagged SDPA without allocating dense `(B, 1, S, S)` masks.
    Raises ValueError if any sequence is empty (`<= 0`) or exceeds `pack_len` under `overflow_policy='raise'`.
    """
    normalized_items = _normalize_pack_inputs(
        pairs=pairs,
        pack_len=pack_len,
        rows_per_batch=rows_per_batch,
        attn_mode=attn_mode,
        truncate=truncate,
        overflow_policy=overflow_policy,
        curriculum_weights=curriculum_weights,
    )

    bins_used: list[int] = []
    bins_entries: list[list[tuple[Tensor, Tensor, float | None]]] = []

    for tok, lbl, cw in normalized_items:
        n = int(tok.numel())
        placed = -1
        for b_idx, used in enumerate(bins_used):
            if used + n <= pack_len:
                placed = b_idx
                break
        if placed < 0:
            placed = len(bins_used)
            bins_used.append(0)
            bins_entries.append([])
        bins_entries[placed].append((tok, lbl, cw))
        bins_used[placed] += n

    return _materialize_packed_bins(
        bins_entries=bins_entries,
        pack_len=pack_len,
        rows_per_batch=rows_per_batch,
        attn_mode=attn_mode,
        has_curriculum_weights=(curriculum_weights is not None),
    )


def pack_sequences_obfd(
    pairs: list[tuple[Tensor, Tensor]],
    pack_len: int = 512,
    rows_per_batch: int = 4,
    attn_mode: str = "dense",
    truncate: bool = False,
    overflow_policy: str = "raise",
    curriculum_weights: list[float] | tuple[float, ...] | None = None,
) -> list[PreparedBatch]:
    """Pack sequences using Ding et al. (2024) `O(N log N)` Segment-Tree Best-Fit-Decreasing (`OBFD`).

    Minimizes wasted bin capacity and total packed bins without truncations (when `pack_len >= max_len`
    or `overflow_policy='chunk'`) while resetting `position_ids` and isolating `seq_id` / `cu_seqlens`.
    """
    normalized_items = _normalize_pack_inputs(
        pairs=pairs,
        pack_len=pack_len,
        rows_per_batch=rows_per_batch,
        attn_mode=attn_mode,
        truncate=truncate,
        overflow_policy=overflow_policy,
        curriculum_weights=curriculum_weights,
    )
    if not normalized_items:
        return []

    item_lens = [int(tok.numel()) for tok, _, _ in normalized_items]
    bin_assignments = pack_lengths_obfd(item_lens, bin_cap=pack_len)
    bins_entries: list[list[tuple[Tensor, Tensor, float | None]]] = [
        [normalized_items[orig_idx] for orig_idx, _off, _ln in b_slices]
        for b_slices in bin_assignments
    ]
    return _materialize_packed_bins(
        bins_entries=bins_entries,
        pack_len=pack_len,
        rows_per_batch=rows_per_batch,
        attn_mode=attn_mode,
        has_curriculum_weights=(curriculum_weights is not None),
    )


def collate_pack_for_dataloader(
    pairs: list[tuple[Tensor, Tensor]],
    pack_len: int = 512,
) -> PreparedBatch:
    """Collate function for DataLoader workers: packs one micro-pool of sequences into rows."""
    packed_list = pack_sequence_pairs(
        pairs, pack_len=pack_len, rows_per_batch=max(len(pairs), 1), attn_mode="dense"
    )
    return packed_list[0]


@dataclass(frozen=True)
class PackCollate:
    """Top-level picklable collate callable binding `pack_len` for `spawn` DataLoader workers."""

    pack_len: int = 512

    def __call__(self, pairs: list[tuple[Tensor, Tensor]]) -> PreparedBatch:
        return collate_pack_for_dataloader(pairs, pack_len=self.pack_len)


def collate_varlen_pairs(
    pairs: list[tuple[Tensor, Tensor]],
    max_seq_len: int | None = None,
) -> PreparedBatch:
    """Concatenate sequences into 1D flat tensors + jagged offsets (zero padding)."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    lens: list[int] = []
    for seq_idx, (t, lbl) in enumerate(pairs):
        n = int(t.numel())
        if int(lbl.numel()) != n:
            raise ValueError(f"Sequence {seq_idx} token/label length mismatch: {n} vs {int(lbl.numel())}")
        if n <= 0:
            raise ValueError(f"Sequence {seq_idx} must have length >= 1, got {n}")
        if max_seq_len is not None and n > max_seq_len:
            raise ValueError(f"Sequence {seq_idx} length {n} exceeds max_seq_len={max_seq_len}")
        lens.append(n)
    toks = torch.cat([t for t, _ in pairs], dim=0)
    lbls = torch.cat([lbl for _, lbl in pairs], dim=0)
    pos = torch.cat([torch.arange(n, dtype=torch.int64) for n in lens], dim=0)
    offsets, max_s = build_cu_seqlens(lens)
    total = int(toks.numel())
    return PreparedBatch(
        mode="varlen",
        input_ids=toks,
        position_ids=pos,
        labels=lbls,
        cu_seqlens=offsets,
        max_seqlen=max_s,
        seq_lens=lens,
        real_tokens=total,
        total_tokens=total,
    )


def build_batches_from_items(
    items: list[ProtocolSequenceItem],
    strategy: str,
    batch_size: int = 16,
    pack_len: int = 512,
    pack_rows_per_batch: int = 3,
) -> list[PreparedBatch]:
    """Pre-build batches for compute-side ablations: padded, bucketed, packed, packed_varlen, varlen, block_sparse."""
    pairs = [
        (
            torch.tensor(it.tokens, dtype=torch.int64),
            torch.tensor(it.labels, dtype=torch.int64),
        )
        for it in items
    ]

    if strategy == "padded":
        return [
            collate_padded_pairs(pairs[i : i + batch_size])
            for i in range(0, len(pairs), batch_size)
        ]

    if strategy == "bucketed":
        sorted_pairs = sorted(pairs, key=lambda x: int(x[0].numel()))
        return [
            collate_padded_pairs(sorted_pairs[i : i + batch_size])
            for i in range(0, len(sorted_pairs), batch_size)
        ]

    if strategy == "packed":
        return pack_sequence_pairs(
            pairs, pack_len=pack_len, rows_per_batch=pack_rows_per_batch, attn_mode="dense"
        )

    if strategy == "packed_obfd":
        return pack_sequences_obfd(
            pairs, pack_len=pack_len, rows_per_batch=pack_rows_per_batch, attn_mode="dense"
        )

    if strategy == "packed_varlen":
        return pack_sequence_pairs(
            pairs, pack_len=pack_len, rows_per_batch=pack_rows_per_batch, attn_mode="packed_varlen"
        )

    if strategy == "packed_obfd_varlen":
        return pack_sequences_obfd(
            pairs, pack_len=pack_len, rows_per_batch=pack_rows_per_batch, attn_mode="packed_varlen"
        )

    if strategy == "block_sparse":
        return pack_sequence_pairs(
            pairs, pack_len=pack_len, rows_per_batch=pack_rows_per_batch, attn_mode="block_sparse"
        )

    if strategy == "varlen":
        return [
            collate_varlen_pairs(pairs[i : i + batch_size])
            for i in range(0, len(pairs), batch_size)
        ]

    raise ValueError(f"Unknown batching strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# 3. Small Protocol Transformer supporting dense, packed, varlen, and block-sparse
# ---------------------------------------------------------------------------


def build_block_diag_mask(seq_id: Tensor, causal: bool = False) -> Tensor:
    """Construct (B, 1, S, S) boolean SDPA attention mask from packed `seq_id` (B, S).

    Positions attend iff they belong to the same non-pad sequence (`seq_id > 0`),
    preventing any cross-contamination between packed protocols.
    """
    same_seq = (seq_id.unsqueeze(2) == seq_id.unsqueeze(1)) & (seq_id.unsqueeze(2) > 0)
    if causal:
        s = seq_id.shape[1]
        idx = torch.arange(s, device=seq_id.device)
        causal_ok = idx.unsqueeze(1) >= idx.unsqueeze(0)
        same_seq = same_seq & causal_ok.unsqueeze(0)
    # Ensure pad queries attend to self so softmax denominator is never all-False (NaN-free!)
    s = seq_id.shape[1]
    eye = torch.eye(s, dtype=torch.bool, device=seq_id.device).unsqueeze(0)
    is_pad_q = (seq_id == 0).unsqueeze(2)
    safe_mask = torch.where(is_pad_q, eye, same_seq)
    return safe_mask.unsqueeze(1)  # (B, 1, S, S)


def _sequence_start_offsets(seq_id: Tensor) -> Tensor:
    """Return (B, S) start column index of each token's contiguous packed sequence (`0` on pad)."""
    b, s = seq_id.shape
    cols = torch.arange(s, device=seq_id.device).unsqueeze(0).expand(b, s)
    is_start = seq_id > 0
    if s > 1:
        is_start = is_start.clone()
        is_start[:, 1:] = (seq_id[:, 1:] > 0) & (seq_id[:, 1:] != seq_id[:, :-1])
    markers = torch.where(is_start, cols, torch.zeros_like(cols))
    return markers.cummax(dim=1).values


def tiled_local_global_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_id: Tensor,
    block_size: int = 64,
    num_global_tokens: int = 4,
    causal: bool = False,
) -> Tensor:
    """Tiled block-sparse attention: each block of W queries attends to [prev_block, cur_block] + G per-sequence globals.

    Respects `seq_id` boundaries so packed sequences never leak across sequence borders, and gathers
    the first `G` tokens of each query block's leading packed sequence so packed sequences `seq_id >= 2`
    retain their own global header context.
    Reduces attention key-value evaluations from S (e.g. 512) to 2*W + G (e.g. 132).
    """
    b, h, s, d = q.shape
    if s % block_size != 0 or s <= 2 * block_size:
        mask = build_block_diag_mask(seq_id, causal=causal)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    n_blk = s // block_size
    w = block_size
    g_cnt = min(num_global_tokens, w)

    # Queries reshaped into blocks: (B, H, N_blk, W, D) -> (B * N_blk, H, W, D)
    q_blk = q.view(b, h, n_blk, w, d).permute(0, 2, 1, 3, 4).reshape(b * n_blk, h, w, d)

    # Build previous + current block keys/values: pad 1 zero block on the left
    k_blocks = k.view(b, h, n_blk, w, d)
    v_blocks = v.view(b, h, n_blk, w, d)
    zero_kv = torch.zeros((b, h, 1, w, d), dtype=k.dtype, device=k.device)
    k_padded = torch.cat([zero_kv, k_blocks], dim=2)  # (B, H, N_blk + 1, W, D)
    v_padded = torch.cat([zero_kv, v_blocks], dim=2)

    # Local 2W window per block: [prev_blk, cur_blk]
    k_local = torch.cat([k_padded[:, :, :-1], k_padded[:, :, 1:]], dim=3)  # (B, H, N_blk, 2W, D)
    v_local = torch.cat([v_padded[:, :, :-1], v_padded[:, :, 1:]], dim=3)

    # Per-sequence global tokens: at most one packed sequence per block started before `cur_blk`
    # (namely the sequence crossing `blk * W`); any sequence starting inside `cur_blk` already has
    # its first G tokens inside `k_local`.
    seq_starts = _sequence_start_offsets(seq_id)         # (B, S)
    lead_start = seq_starts[:, ::w]                      # (B, N_blk)
    g_offs = torch.arange(g_cnt, device=q.device).view(1, 1, g_cnt)
    glob_idx = (lead_start.unsqueeze(2) + g_offs).clamp(0, s - 1)  # (B, N_blk, G)

    idx_4d = glob_idx.view(b, 1, n_blk * g_cnt, 1).expand(b, h, n_blk * g_cnt, d)
    k_glob = k.gather(2, idx_4d).view(b, h, n_blk, g_cnt, d)
    v_glob = v.gather(2, idx_4d).view(b, h, n_blk, g_cnt, d)

    kv_len = 2 * w + g_cnt
    k_win = (
        torch.cat([k_glob, k_local], dim=3)
        .permute(0, 2, 1, 3, 4)
        .reshape(b * n_blk, h, kv_len, d)
    )
    v_win = (
        torch.cat([v_glob, v_local], dim=3)
        .permute(0, 2, 1, 3, 4)
        .reshape(b * n_blk, h, kv_len, d)
    )

    # Construct matching sequence-ID & position masks of shape (B * N_blk, 1, W, kv_len)
    sid_blocks = seq_id.view(b, n_blk, w)
    sid_pad0 = torch.zeros((b, 1, w), dtype=seq_id.dtype, device=seq_id.device)
    sid_padded = torch.cat([sid_pad0, sid_blocks], dim=1)
    sid_local = torch.cat([sid_padded[:, :-1], sid_padded[:, 1:]], dim=2)  # (B, N_blk, 2W)
    sid_glob = seq_id.gather(1, glob_idx.view(b, n_blk * g_cnt)).view(b, n_blk, g_cnt)
    sid_kv = torch.cat([sid_glob, sid_local], dim=2).reshape(b * n_blk, kv_len)
    sid_q = sid_blocks.reshape(b * n_blk, w)

    same_seq = (sid_q.unsqueeze(2) == sid_kv.unsqueeze(1)) & (sid_q.unsqueeze(2) > 0)

    # Drop global slots whose key already sits in the local [prev_blk, cur_blk] window
    # so no key is counted twice in the softmax.
    blk_start = (torch.arange(n_blk, device=q.device) * w).view(1, n_blk, 1)
    glob_dup = (glob_idx >= (blk_start - w)) & (glob_idx < (blk_start + w))  # (B, N_blk, G)
    keep_kv = torch.cat(
        [~glob_dup, torch.ones((b, n_blk, 2 * w), dtype=torch.bool, device=q.device)], dim=2
    ).reshape(b * n_blk, kv_len)
    same_seq = same_seq & keep_kv.unsqueeze(1)

    if causal:
        pos_row = torch.arange(s, device=q.device).unsqueeze(0).expand(b, s)
        pos_blocks = pos_row.view(b, n_blk, w)
        pos_pad = torch.full((b, 1, w), s + 1, dtype=torch.int64, device=q.device)
        pos_padded = torch.cat([pos_pad, pos_blocks], dim=1)
        pos_local = torch.cat([pos_padded[:, :-1], pos_padded[:, 1:]], dim=2)
        pos_glob = glob_idx
        pos_kv = torch.cat([pos_glob, pos_local], dim=2).reshape(b * n_blk, kv_len)
        pos_q = pos_blocks.reshape(b * n_blk, w)
        same_seq = same_seq & (pos_q.unsqueeze(2) >= pos_kv.unsqueeze(1))

    # Safe self-attend fallback for padded query rows (where sid_q == 0): attend to cur_block diagonal
    diag_cur = torch.zeros((w, kv_len), dtype=torch.bool, device=q.device)
    diag_idx = torch.arange(w, device=q.device)
    diag_cur[diag_idx, g_cnt + w + diag_idx] = True
    safe_mask = torch.where((sid_q == 0).unsqueeze(2), diag_cur.unsqueeze(0), same_seq).unsqueeze(1)

    out_blk = F.scaled_dot_product_attention(q_blk, k_win, v_win, attn_mask=safe_mask)
    return out_blk.view(b, n_blk, h, w, d).permute(0, 2, 1, 3, 4).reshape(b, h, s, d)


def build_flex_block_mask(
    seq_id: Tensor,
    block_size: int = 64,
    num_global_tokens: int = 4,
    causal: bool = False,
    compile_mask: bool = False,
) -> Any:
    """Prebuild and return a reusable PyTorch FlexAttention `BlockMask` for a `(B, S)` `seq_id` tensor."""
    from torch.nn.attention.flex_attention import create_block_mask

    b, s = seq_id.shape
    w = block_size
    g_cnt = min(num_global_tokens, w)
    seq_starts = _sequence_start_offsets(seq_id)
    dense_fallback = s % w != 0 or s <= 2 * w

    def mask_mod(b_idx: Tensor, _h_idx: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        sq = seq_id[b_idx, q_idx]
        sk = seq_id[b_idx, kv_idx]
        is_pad_q = sq == 0
        same = (sq == sk) & (sq > 0)
        if not dense_fallback:
            q_blk = q_idx // w
            kv_blk = kv_idx // w
            in_win = (kv_blk == q_blk) | (kv_blk == q_blk - 1)
            is_glob = (kv_idx - seq_starts[b_idx, kv_idx]) < g_cnt
            same = same & (in_win | is_glob)
        if causal:
            same = same & (kv_idx <= q_idx)
        return torch.where(is_pad_q, q_idx == kv_idx, same)

    return create_block_mask(
        mask_mod,
        B=b,
        H=None,
        Q_LEN=s,
        KV_LEN=s,
        device=str(seq_id.device),
        _compile=compile_mask,
    )


def _get_flex_attention_fn(use_compile: bool) -> Any:
    """Return compiled `flex_attention` on CUDA (with clean eager fallback) or eager on CPU."""
    global _COMPILED_FLEX_ATTENTION
    from torch.nn.attention.flex_attention import flex_attention

    if not use_compile:
        return flex_attention
    if _COMPILED_FLEX_ATTENTION is None:
        try:
            _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)
        except Exception:
            return flex_attention
    return _COMPILED_FLEX_ATTENTION


def flex_local_global_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_id: Tensor,
    block_size: int = 64,
    num_global_tokens: int = 4,
    causal: bool = False,
    block_mask: Any = None,
    compile_flex: bool | None = None,
) -> Tensor:
    """PyTorch FlexAttention execution path matching `tiled_local_global_attention` semantics."""
    from torch.nn.attention.flex_attention import flex_attention

    bm = (
        block_mask
        if block_mask is not None
        else build_flex_block_mask(
            seq_id=seq_id,
            block_size=block_size,
            num_global_tokens=num_global_tokens,
            causal=causal,
            compile_mask=False,
        )
    )
    should_compile = (q.device.type == "cuda") if compile_flex is None else compile_flex
    if should_compile:
        fn = _get_flex_attention_fn(use_compile=True)
        try:
            return fn(q, k, v, block_mask=bm)
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"flex_attention called without torch\.compile\(\).*",
            category=UserWarning,
        )
        return flex_attention(q, k, v, block_mask=bm)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 1024,
        causal: bool = False,
        use_flex_attention: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.use_flex_attention = use_flex_attention

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward_dense(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        seq_id: Tensor | None = None,
        block_sparse: bool = False,
        use_flex: bool = False,
        block_size: int = 64,
        block_mask: Any = None,
    ) -> Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        if (use_flex or (block_sparse and self.use_flex_attention)) and seq_id is not None:
            attn_out = flex_local_global_attention(
                q,
                k,
                v,
                seq_id=seq_id,
                block_size=block_size,
                causal=self.causal,
                block_mask=block_mask,
            )
        elif block_sparse and seq_id is not None:
            attn_out = tiled_local_global_attention(
                q, k, v, seq_id=seq_id, block_size=block_size, causal=self.causal
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=(self.causal and attn_mask is None)
            )

        x = x + self.out_proj(attn_out.transpose(1, 2).contiguous().view(b, s, d))
        x = x + self.ffn(self.ln2(x))
        return x

    def _run_varlen_qkv(
        self,
        q_raw: Tensor,
        k_raw: Tensor,
        v_raw: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int | None = None,
    ) -> Tensor:
        t_tok, d = q_raw.shape
        if t_tok == 0 or cu_seqlens.numel() <= 1:
            return q_raw.new_zeros((t_tok, d))

        if (
            _FLASH_ATTN_VARLEN_FUNC is not None
            and q_raw.device.type == "cuda"
            and q_raw.dtype in (torch.float16, torch.bfloat16)
        ):
            cu_i32 = cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
            eff_max = (
                int(max_seqlen)
                if max_seqlen is not None and max_seqlen > 0
                else int((cu_i32[1:] - cu_i32[:-1]).max().item())
            )
            q_h = q_raw.contiguous().view(t_tok, self.n_heads, self.d_head)
            k_h = k_raw.contiguous().view(t_tok, self.n_heads, self.d_head)
            v_h = v_raw.contiguous().view(t_tok, self.n_heads, self.d_head)
            attn_fa = _FLASH_ATTN_VARLEN_FUNC(
                q_h, k_h, v_h, cu_i32, cu_i32, eff_max, eff_max, causal=self.causal
            )
            return attn_fa.reshape(t_tok, d)

        if self.causal and q_raw.device.type != "cuda":
            # PyTorch's CPU math nested_tensor SDPA backend rejects `is_causal=True`;
            # run exact zero-padding per-segment SDPA over `cu_seqlens` slices on CPU.
            q_h = q_raw.contiguous().view(t_tok, self.n_heads, self.d_head)
            k_h = k_raw.contiguous().view(t_tok, self.n_heads, self.d_head)
            v_h = v_raw.contiguous().view(t_tok, self.n_heads, self.d_head)
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to("cpu").tolist()
            outs: list[Tensor] = []
            for q_s, k_s, v_s in zip(
                q_h.split(lengths, dim=0),
                k_h.split(lengths, dim=0),
                v_h.split(lengths, dim=0),
                strict=True,
            ):
                if q_s.shape[0] == 0:
                    continue
                seg_out = F.scaled_dot_product_attention(
                    q_s.transpose(0, 1).unsqueeze(0),
                    k_s.transpose(0, 1).unsqueeze(0),
                    v_s.transpose(0, 1).unsqueeze(0),
                    is_causal=True,
                )
                outs.append(seg_out.squeeze(0).transpose(0, 1).contiguous().view(-1, d))
            return torch.cat(outs, dim=0) if outs else q_raw.new_zeros((0, d))

        # Nested tensors require contiguous storage before transpose(1, 2)
        q_nt = torch.nested.nested_tensor_from_jagged(
            q_raw.contiguous().view(t_tok, self.n_heads, self.d_head), cu_seqlens
        ).transpose(1, 2)
        k_nt = torch.nested.nested_tensor_from_jagged(
            k_raw.contiguous().view(t_tok, self.n_heads, self.d_head), cu_seqlens
        ).transpose(1, 2)
        v_nt = torch.nested.nested_tensor_from_jagged(
            v_raw.contiguous().view(t_tok, self.n_heads, self.d_head), cu_seqlens
        ).transpose(1, 2)

        # PyTorch's CPU jagged SDPA internally converts jagged -> strided nested_tensor,
        # triggering a prototype UserWarning even when inputs already use layout=torch.jagged.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            attn_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt, is_causal=self.causal)
        return attn_nt.transpose(1, 2).values().contiguous().view(t_tok, d)

    def forward_varlen(
        self,
        x: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int | None = None,
    ) -> Tensor:
        q_raw, k_raw, v_raw = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        attn_flat = self._run_varlen_qkv(q_raw, k_raw, v_raw, cu_seqlens, max_seqlen=max_seqlen)
        x = x + self.out_proj(attn_flat)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_packed_segmented(
        self,
        x: Tensor,
        active_mask: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int | None = None,
    ) -> Tensor:
        """Execute dense `(B, S, D)` packed rows via `cu_seqlens` without O(B * S^2) mask allocation."""
        q_raw, k_raw, v_raw = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        # Pad slots self-attend with weight 1.0 under build_block_diag_mask -> exact output is v_raw
        attn_full = v_raw.clone()
        if cu_seqlens.numel() > 1:
            attn_act = self._run_varlen_qkv(
                q_raw[active_mask],
                k_raw[active_mask],
                v_raw[active_mask],
                cu_seqlens,
                max_seqlen=max_seqlen,
            )
            attn_full[active_mask] = attn_act.to(attn_full.dtype)
        x = x + self.out_proj(attn_full)
        x = x + self.ffn(self.ln2(x))
        return x


class ProtocolTransformer(nn.Module):
    """Small Transformer (default 4 layers, d=256) for protocol error-step / token tasks."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 512,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        num_classes: int = NUM_ERROR_CLASSES,
        causal: bool = False,
        block_size: int = 64,
        use_flex_attention: bool = False,
        varlen_dispatch_threshold: int = VARLEN_DISPATCH_THRESHOLD,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.causal = causal
        self.block_size = block_size
        self.use_flex_attention = use_flex_attention
        self.varlen_dispatch_threshold = varlen_dispatch_threshold

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    causal=causal,
                    use_flex_attention=use_flex_attention,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _validate_position_ids(self, position_ids: Tensor) -> None:
        max_pos_allowed = self.pos_emb.num_embeddings
        if torch.compiler.is_compiling():
            return
        if position_ids.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return
        if position_ids.numel() == 0:
            return
        min_pos = int(position_ids.min().item())
        max_pos = int(position_ids.max().item())
        if min_pos < 0 or max_pos >= max_pos_allowed:
            raise ValueError(
                f"position_ids out of range [0, {max_pos_allowed - 1}]: got min={min_pos}, max={max_pos}; "
                f"increase max_seq_len (currently {max_pos_allowed}) or chunk sequences cleanly"
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_id: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        mode: str = "dense",
        max_seqlen: int | None = None,
        block_mask: Any = None,
        chunk_oversized: bool = False,
    ) -> Tensor:
        max_pos_allowed = self.pos_emb.num_embeddings
        if (
            chunk_oversized
            and mode == "dense"
            and input_ids.ndim == 2
            and input_ids.shape[1] > max_pos_allowed
        ):
            outs: list[Tensor] = []
            s_total = input_ids.shape[1]
            for start in range(0, s_total, max_pos_allowed):
                end = min(start + max_pos_allowed, s_total)
                sub_pos = torch.remainder(position_ids[:, start:end], max_pos_allowed)
                sub_sid = seq_id[:, start:end] if seq_id is not None else None
                outs.append(
                    self.forward(
                        input_ids[:, start:end],
                        sub_pos,
                        seq_id=sub_sid,
                        mode="dense",
                        chunk_oversized=False,
                    )
                )
            return torch.cat(outs, dim=1)

        self._validate_position_ids(position_ids)
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids)

        if mode in ("varlen", "packed_varlen"):
            if cu_seqlens is None:
                raise ValueError(f"cu_seqlens is required when mode={mode!r}")
            for blk in self.blocks:
                x = blk.forward_varlen(x, cu_seqlens, max_seqlen=max_seqlen)
            return self.head(self.ln_f(x))

        use_flex = mode == "flex_block_sparse"
        block_sparse = mode in ("block_sparse", "flex_block_sparse")

        is_capturing = input_ids.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        if (
            not block_sparse
            and seq_id is not None
            and input_ids.ndim == 2
            and input_ids.shape[1] >= self.varlen_dispatch_threshold
            and not is_capturing
            and not torch.compiler.is_compiling()
        ):
            active_mask = seq_id > 0
            cu_seg, seg_max = build_cu_seqlens(seq_id)
            for blk in self.blocks:
                x = blk.forward_packed_segmented(x, active_mask, cu_seg, max_seqlen=seg_max)
            return self.head(self.ln_f(x))

        attn_mask = None
        if not block_sparse and seq_id is not None:
            attn_mask = build_block_diag_mask(seq_id, causal=self.causal)

        shared_bm = block_mask
        if (use_flex or (block_sparse and self.use_flex_attention)) and seq_id is not None and shared_bm is None:
            shared_bm = build_flex_block_mask(
                seq_id=seq_id,
                block_size=self.block_size,
                causal=self.causal,
                compile_mask=False,
            )

        for blk in self.blocks:
            x = blk.forward_dense(
                x,
                attn_mask=attn_mask,
                seq_id=seq_id,
                block_sparse=block_sparse,
                use_flex=use_flex,
                block_size=self.block_size,
                block_mask=shared_bm,
            )
        return self.head(self.ln_f(x))


def _per_sequence_inv_len_weights(
    labels: Tensor,
    active: Tensor,
    *,
    seq_id: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute `(tok_inv_len, num_active_seqs)` without host-device sync (`CUDA Graph` & `compile` safe)."""
    active_f = active.to(torch.float32)

    if cu_seqlens is not None:
        seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
        n_segs = int(seg_lens.numel())
        if n_segs == 0 or labels.numel() == 0:
            return torch.zeros_like(active_f), torch.zeros((), dtype=torch.float32, device=labels.device)
        seg_ids = torch.repeat_interleave(
            torch.arange(n_segs, device=labels.device, dtype=torch.int64), seg_lens
        )
        counts = torch.zeros(n_segs, dtype=torch.float32, device=labels.device)
        counts.scatter_add_(0, seg_ids, active_f.reshape(-1))
        inv_len = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
        tok_inv_len = inv_len[seg_ids].reshape(labels.shape) * active_f
        k_seqs = (counts > 0).sum().to(torch.float32)
        return tok_inv_len, k_seqs

    if seq_id is not None:
        sid2d = seq_id.unsqueeze(0) if seq_id.ndim == 1 else seq_id
        act2d = active.unsqueeze(0) if active.ndim == 1 else active
        valid2d = act2d & (sid2d > 0)
        b_dim, s_dim = sid2d.shape
        is_new = torch.zeros_like(sid2d, dtype=torch.bool)
        if s_dim > 0:
            is_new[:, 0] = sid2d[:, 0] > 0
        if s_dim > 1:
            is_new[:, 1:] = (sid2d[:, 1:] > 0) & (sid2d[:, 1:] != sid2d[:, :-1])
        run_id = is_new.reshape(-1).to(torch.int64).cumsum(dim=0)
        valid_flat = valid2d.reshape(-1)
        bucket = torch.where(valid_flat, run_id, torch.zeros_like(run_id))
        counts = torch.zeros(b_dim * s_dim + 1, dtype=torch.float32, device=labels.device)
        counts.scatter_add_(0, bucket, valid_flat.to(torch.float32))
        counts[0] = 0.0
        inv_len = torch.where(counts > 0, 1.0 / counts.clamp(min=1.0), torch.zeros_like(counts))
        tok_inv_len = (inv_len[bucket] * valid_flat.to(torch.float32)).reshape(labels.shape)
        k_seqs = (counts > 0).sum().to(torch.float32)
        return tok_inv_len, k_seqs

    if labels.ndim == 2:
        row_counts = active_f.sum(dim=1, keepdim=True)
        tok_inv_len = torch.where(
            row_counts > 0, active_f / row_counts.clamp(min=1.0), torch.zeros_like(active_f)
        )
        k_seqs = (row_counts.squeeze(1) > 0).sum().to(torch.float32)
        return tok_inv_len, k_seqs

    total_cnt = active_f.sum()
    tok_inv_len = torch.where(
        total_cnt > 0, active_f / total_cnt.clamp(min=1.0), torch.zeros_like(active_f)
    )
    k_seqs = (total_cnt > 0).to(torch.float32)
    return tok_inv_len, k_seqs


def packed_sequence_ce_loss(
    logits: Tensor,
    labels: Tensor,
    loss_reduction: str = "sequence_mean",
    *,
    seq_id: Tensor | None = None,
    cu_seqlens: Tensor | None = None,
    token_weights: Tensor | None = None,
    fixed_denom: float | None = None,
    ignore_index: int = IGNORE_LABEL,
) -> Tensor:
    """Compute cross-entropy loss over packed variable-length protocol sequences without length bias.

    Supported `loss_reduction` modes:
      - `'sequence_mean'`       : normalizes each packed sequence `i` by `1 / (K_b * s_i)` so co-packed
                                  short and long protocols contribute equally per example (and incorporates
                                  `token_weights` per sequence when supplied).
      - `'curriculum_weighted'` : multiplies per-token losses by `token_weights` (e.g. `curriculum_weights_bf16`
                                  `p * (1 - p)`) normalized over active weighted mass (and length-normalized
                                  per sequence when `seq_id` or `cu_seqlens` is provided).
      - `'token_mean'`          : legacy flat active-token mean reduction.
      - `'fixed_horizon'`       : divides summed active loss by constant `fixed_denom` to avoid tail-batch LR spikes.
    """
    logits_f32 = logits.float()
    if labels.numel() == 0:
        return logits_f32.sum() * 0.0

    tok_loss = F.cross_entropy(
        logits_f32.reshape(-1, logits_f32.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)

    active = labels != ignore_index
    active_f = active.to(torch.float32)

    if loss_reduction == "token_mean":
        if token_weights is not None:
            w = token_weights.to(device=labels.device, dtype=torch.float32).reshape(labels.shape) * active_f
            denom = w.sum()
            return torch.where(denom > 0, (tok_loss * w).sum() / denom.clamp(min=1e-12), logits_f32.sum() * 0.0)
        denom = active_f.sum()
        return torch.where(denom > 0, (tok_loss * active_f).sum() / denom.clamp(min=1.0), logits_f32.sum() * 0.0)

    if loss_reduction == "fixed_horizon":
        denom_val = float(fixed_denom) if fixed_denom is not None else float(max(labels.numel(), 1))
        if denom_val <= 0.0:
            raise ValueError(f"fixed_denom must be > 0, got {denom_val}")
        if token_weights is not None:
            w = token_weights.to(device=labels.device, dtype=torch.float32).reshape(labels.shape) * active_f
            return (tok_loss * w).sum() / denom_val
        return (tok_loss * active_f).sum() / denom_val

    if loss_reduction == "sequence_mean":
        tok_inv_len, k_seqs = _per_sequence_inv_len_weights(
            labels, active, seq_id=seq_id, cu_seqlens=cu_seqlens
        )
        if token_weights is not None:
            cw = token_weights.to(device=labels.device, dtype=torch.float32).reshape(labels.shape)
            w = tok_inv_len * cw
            denom = w.sum()
            return torch.where(denom > 0, (tok_loss * w).sum() / denom.clamp(min=1e-12), logits_f32.sum() * 0.0)
        return torch.where(
            k_seqs > 0, (tok_loss * tok_inv_len).sum() / k_seqs.clamp(min=1.0), logits_f32.sum() * 0.0
        )

    if loss_reduction == "curriculum_weighted":
        if token_weights is None:
            raise ValueError("token_weights is required when loss_reduction='curriculum_weighted'")
        cw = token_weights.to(device=labels.device, dtype=torch.float32).reshape(labels.shape) * active_f
        if seq_id is not None or cu_seqlens is not None:
            tok_inv_len, _ = _per_sequence_inv_len_weights(
                labels, active, seq_id=seq_id, cu_seqlens=cu_seqlens
            )
            w = tok_inv_len * cw
        else:
            w = cw
        denom = w.sum()
        return torch.where(denom > 0, (tok_loss * w).sum() / denom.clamp(min=1e-12), logits_f32.sum() * 0.0)

    raise ValueError(
        f"Unsupported loss_reduction={loss_reduction!r}; expected 'sequence_mean', "
        "'curriculum_weighted', 'token_mean', or 'fixed_horizon'"
    )


compute_packed_loss = packed_sequence_ce_loss


# ---------------------------------------------------------------------------
# 4. Analytic FLOP counter, GPU peak lookup, FlexAttention probe, CUDA Graph runner, and profiler
# ---------------------------------------------------------------------------


_GPU_PEAK_FLOPS_TABLE: tuple[tuple[tuple[str, ...], float, float, float], ...] = (
    # (substring_tokens, bf16_tflops, tf32_tflops, fp32_tflops) - dense non-sparse vendor specs
    (("h200",), 989.5, 494.7, 67.0),
    (("h100", "nvl"), 835.0, 417.5, 60.0),
    (("h100", "pcie"), 756.5, 378.2, 51.2),
    (("h800", "pcie"), 756.5, 378.2, 51.2),
    (("h100",), 989.5, 494.7, 67.0),
    (("h800",), 989.5, 494.7, 67.0),
    (("a100",), 312.0, 156.0, 19.5),
    (("a800",), 312.0, 156.0, 19.5),
    (("l40s",), 362.05, 181.0, 91.6),
    (("l40",), 181.05, 90.5, 90.5),
    (("rtx 4090",), 330.3, 82.6, 82.6),
    (("rtx 3090",), 142.0, 35.6, 35.6),
    (("a10g",), 125.0, 62.5, 31.2),
    (("a10",), 125.0, 62.5, 31.2),
    (("l4",), 121.0, 60.5, 30.3),
)


def detect_gpu_peak_flops(
    gpu_name: str | None,
    precision: str = "bf16",
    override_tflops: float | None = None,
) -> tuple[float | None, str]:
    """Return `(peak_flops_per_sec, description)` from `--peak-tflops` override or NVIDIA datasheet lookup."""
    if override_tflops is not None:
        if override_tflops <= 0.0:
            raise ValueError(f"override_tflops must be > 0, got {override_tflops}")
        return float(override_tflops) * 1e12, f"manual override ({override_tflops:.2f} TFLOPS)"

    if not gpu_name:
        return None, "N/A"
    lowered = gpu_name.strip().lower()
    if lowered in ("", "none", "none (cpu only)", "cpu"):
        return None, "N/A"

    prec = precision.lower()
    if "v100" in lowered:
        if prec == "fp16":
            return 125.0e12, "NVIDIA V100 FP16 Tensor Core (125.0 TFLOPS)"
        if prec == "fp32":
            return 15.7e12, "NVIDIA V100 FP32 (15.7 TFLOPS)"
        return None, "N/A (V100 lacks native BF16 Tensor Cores; pass --peak-tflops)"

    for tokens, bf16_tf, tf32_tf, fp32_tf in _GPU_PEAK_FLOPS_TABLE:
        if all(tok in lowered for tok in tokens):
            if prec in ("bf16", "fp16"):
                val = bf16_tf
            elif prec == "tf32":
                val = tf32_tf
            elif prec == "fp32":
                val = fp32_tf
            else:
                raise ValueError(f"Unsupported precision={precision!r}")
            return val * 1e12, f"{gpu_name} {prec.upper()} peak ({val:.2f} TFLOPS)"

    return None, f"N/A (unknown GPU peak for {gpu_name!r}; pass --peak-tflops)"


def count_batch_training_flops(
    model: ProtocolTransformer,
    batch: PreparedBatch,
) -> tuple[float, float]:
    """Return (useful_flops, executed_flops) for one forward + backward step (3x fwd FLOPs).

    Separates real protocol tokens and true within-sequence attention from wasted pad tokens
    and masked cross-sequence attention entries.
    """
    n_l = model.n_layers
    d = model.d_model
    f = model.d_ff
    c_out = model.num_classes

    # Forward linear FLOPs per token: QKV (6 d^2) + Out (2 d^2) + FFN (4 d f) per layer + head (2 d c_out)
    linear_fwd_per_tok = float(n_l * (8 * d * d + 4 * d * f) + 2 * d * c_out)

    real_toks = float(batch.real_tokens)
    if model.causal:
        attn_pairs_real = float(sum(0.5 * s * (s + 1) for s in batch.seq_lens))
    else:
        attn_pairs_real = float(sum(s * s for s in batch.seq_lens))
    useful_fwd = linear_fwd_per_tok * real_toks + float(4 * n_l * d) * attn_pairs_real

    if batch.mode in ("varlen", "packed_varlen"):
        exec_fwd = useful_fwd
    elif batch.mode in ("block_sparse", "flex_block_sparse"):
        b_rows, s_len = batch.input_ids.shape
        if s_len % model.block_size != 0 or s_len <= 2 * model.block_size:
            eff_kv = float(s_len)
        else:
            eff_kv = float(2 * model.block_size + min(4, model.block_size))
        exec_fwd = linear_fwd_per_tok * float(batch.total_tokens) + float(
            4 * n_l * d * b_rows * s_len
        ) * eff_kv
    else:
        b_rows, s_len = batch.input_ids.shape
        if batch.seq_id is not None and s_len >= model.varlen_dispatch_threshold:
            attn_pairs_exec = attn_pairs_real
        elif batch.seq_id is None and model.causal:
            attn_pairs_exec = 0.5 * float(b_rows) * float(s_len) * float(s_len + 1)
        else:
            attn_pairs_exec = float(b_rows * s_len * s_len)
        exec_fwd = linear_fwd_per_tok * float(batch.total_tokens) + float(4 * n_l * d) * attn_pairs_exec

    return 3.0 * useful_fwd, 3.0 * exec_fwd


def probe_flex_attention(device: str | torch.device | None = None) -> dict[str, Any]:
    """Check whether PyTorch FlexAttention supports forward and backward on `device`."""
    dev = default_device(device)
    info: dict[str, Any] = {
        "device": str(dev),
        "forward_ok": False,
        "backward_ok": False,
        "compiled_ok": False,
        "note": "",
    }
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def local_win(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            return (q_idx - kv_idx).abs() <= 32

        bm = create_block_mask(
            local_win, B=None, H=None, Q_LEN=128, KV_LEN=128, device=str(dev), _compile=False
        )
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"flex_attention called without torch\.compile\(\).*",
                category=UserWarning,
            )
            q_fwd = torch.randn(2, 4, 128, 32, device=dev, dtype=dtype, requires_grad=False)
            _ = flex_attention(q_fwd, q_fwd, q_fwd, block_mask=bm)
            info["forward_ok"] = True

            q_bwd = q_fwd.detach().clone().requires_grad_(True)
            if dev.type == "cuda":
                try:
                    compiled_fn = _get_flex_attention_fn(use_compile=True)
                    out = compiled_fn(q_bwd, q_bwd, q_bwd, block_mask=bm)
                    out.sum().backward()
                    info["backward_ok"] = True
                    info["compiled_ok"] = True
                    info["note"] = "Compiled FlexAttention forward+backward supported"
                    return info
                except Exception:
                    q_bwd = q_fwd.detach().clone().requires_grad_(True)
            out = flex_attention(q_bwd, q_bwd, q_bwd, block_mask=bm)
            out.sum().backward()
            info["backward_ok"] = True
            info["note"] = "FlexAttention forward+backward supported"
    except Exception as exc:
        info["note"] = f"{type(exc).__name__}: {exc}"
    return info


class CudaGraphStepRunner:
    """Capture and replay static-shape packed training steps using CUDA Graphs (`if cuda`)."""

    def __init__(
        self,
        model: ProtocolTransformer,
        optimizer: torch.optim.Optimizer,
        pack_rows: int,
        pack_len: int,
        use_bf16: bool = True,
        loss_reduction: str = "sequence_mean",
        device: torch.device | None = None,
    ) -> None:
        dev = default_device(device)
        if dev.type != "cuda":
            raise RuntimeError("CudaGraphStepRunner requires a CUDA device")
        self.device = dev
        self.model = model
        self.optimizer = optimizer
        self.use_bf16 = use_bf16
        self.loss_reduction = loss_reduction

        self.static_input_ids = torch.zeros((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_pos_ids = torch.zeros((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_seq_id = torch.ones((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_labels = torch.zeros((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_token_weights = torch.ones((pack_rows, pack_len), dtype=torch.float32, device=dev)
        self.static_loss = torch.zeros((), dtype=torch.float32, device=dev)
        self.graph = torch.cuda.CUDAGraph()
        self._captured = False

    def _step_body(self) -> Tensor:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            logits = self.model(
                self.static_input_ids,
                self.static_pos_ids,
                seq_id=self.static_seq_id,
                mode="dense",
            )
            static_tw = getattr(self, "static_token_weights", None)
            loss_red = getattr(self, "loss_reduction", "sequence_mean")
            loss = packed_sequence_ce_loss(
                logits,
                self.static_labels,
                loss_reduction=loss_red,
                seq_id=self.static_seq_id,
                token_weights=static_tw,
            )
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def capture(self) -> None:
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_body()
        torch.cuda.current_stream(device=self.device).wait_stream(s)

        with torch.cuda.graph(self.graph):
            self.static_loss = self._step_body()
        self._captured = True

    def replay_step(self, batch: PreparedBatch) -> float:
        if tuple(batch.input_ids.shape) != tuple(self.static_input_ids.shape):
            raise ValueError(
                f"Batch shape {tuple(batch.input_ids.shape)} != captured static shape "
                f"{tuple(self.static_input_ids.shape)}"
            )
        mdl = getattr(self, "model", None)
        if mdl is not None and batch.seq_lens:
            max_allowed = mdl.pos_emb.num_embeddings
            if max(batch.seq_lens) > max_allowed:
                raise ValueError(
                    f"Batch max sequence length {max(batch.seq_lens)} exceeds model max_seq_len={max_allowed}"
                )
        if not self._captured:
            self.capture()
        self.static_input_ids.copy_(batch.input_ids, non_blocking=True)
        self.static_pos_ids.copy_(batch.position_ids, non_blocking=True)
        if batch.seq_id is not None:
            self.static_seq_id.copy_(batch.seq_id, non_blocking=True)
        else:
            self.static_seq_id.fill_(1)
        self.static_labels.copy_(batch.labels, non_blocking=True)
        static_tw = getattr(self, "static_token_weights", None)
        if static_tw is not None:
            if batch.token_weights is not None:
                static_tw.copy_(batch.token_weights.to(dtype=torch.float32), non_blocking=True)
            else:
                static_tw.fill_(1.0)
        self.graph.replay()
        return float(self.static_loss.item())


@dataclass
class AblationRow:
    config: str
    status: str
    real_tokens_per_sec: float
    useful_token_fraction: float
    step_time_ms: float
    data_wait_ms: float
    compute_ms: float
    data_wait_fraction: float
    useful_tflops_per_sec: float
    executed_tflops_per_sec: float
    mfu: str
    speedup_vs_baseline: float
    final_loss: float
    notes: str = ""
    real_tokens: int = 0
    total_tokens: int = 0


def run_torch_profiler_summary(
    model: ProtocolTransformer,
    batch: PreparedBatch,
    use_bf16: bool = False,
    loss_reduction: str = "sequence_mean",
) -> list[dict[str, Any]]:
    """Run one `torch.profiler.profile` step and extract top 5 ops (sorted by CUDA time on GPU, CPU time on CPU)."""
    dev = next(model.parameters()).device
    b_dev = batch.to(dev)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            logits = model(
                b_dev.input_ids,
                b_dev.position_ids,
                seq_id=b_dev.seq_id,
                cu_seqlens=b_dev.cu_seqlens,
                mode=b_dev.mode,
                max_seqlen=b_dev.max_seqlen,
                block_mask=b_dev.block_mask,
            )
            loss = packed_sequence_ce_loss(
                logits,
                b_dev.labels,
                loss_reduction=loss_reduction,
                seq_id=b_dev.seq_id,
                cu_seqlens=b_dev.cu_seqlens,
                token_weights=b_dev.token_weights,
            )
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()

    def _device_us(ev: Any) -> float:
        return float(
            getattr(ev, "self_device_time_total", getattr(ev, "self_cuda_time_total", 0.0))
        )

    sort_fn = (
        (lambda e: (_device_us(e), float(getattr(e, "self_cpu_time_total", 0.0))))
        if dev.type == "cuda"
        else (lambda e: float(getattr(e, "self_cpu_time_total", 0.0)))
    )
    events = sorted(prof.key_averages(), key=sort_fn, reverse=True)[:5]
    return [
        {
            "op": ev.key,
            "calls": int(ev.count),
            "self_cpu_ms": round(float(getattr(ev, "self_cpu_time_total", 0.0)) / 1000.0, 3),
            "self_cuda_ms": round(_device_us(ev) / 1000.0, 3),
        }
        for ev in events
    ]


def _execute_timed_batches(
    config_name: str,
    model: nn.Module,
    base_model: ProtocolTransformer,
    batch_iter_fn: Any,
    device: torch.device,
    use_bf16: bool = False,
    warmup_steps: int = 1,
    max_steps: int | None = None,
    peak_flops: float | None = None,
    mfu_na_label: str = "N/A",
    loss_reduction: str = "sequence_mean",
    notes: str = "",
) -> AblationRow:
    opt = torch.optim.AdamW(base_model.parameters(), lr=3e-4)

    # Warmup
    w_it: Iterator[PreparedBatch] = iter(batch_iter_fn())
    for _ in range(warmup_steps):
        try:
            wb = next(w_it)
        except StopIteration:
            w_it = iter(batch_iter_fn())
            wb = next(w_it)
        wb_d = wb.to(device, non_blocking=(device.type == "cuda"))
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(
                wb_d.input_ids,
                wb_d.position_ids,
                seq_id=wb_d.seq_id,
                cu_seqlens=wb_d.cu_seqlens,
                mode=wb_d.mode,
                max_seqlen=wb_d.max_seqlen,
                block_mask=wb_d.block_mask,
            )
            loss = packed_sequence_ce_loss(
                logits,
                wb_d.labels,
                loss_reduction=loss_reduction,
                seq_id=wb_d.seq_id,
                cu_seqlens=wb_d.cu_seqlens,
                token_weights=wb_d.token_weights,
            )
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    data_waits: list[float] = []
    computes: list[float] = []
    step_totals: list[float] = []
    total_real = 0
    total_slots = 0
    total_useful_flops = 0.0
    total_exec_flops = 0.0
    last_loss = 0.0

    it: Iterator[PreparedBatch] = iter(batch_iter_fn())
    step_idx = 0
    while True:
        if max_steps is not None and step_idx >= max_steps:
            break
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        batch_d = batch.to(device, non_blocking=(device.type == "cuda"))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        start_ev = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_ev = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        if start_ev is not None:
            start_ev.record()

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(
                batch_d.input_ids,
                batch_d.position_ids,
                seq_id=batch_d.seq_id,
                cu_seqlens=batch_d.cu_seqlens,
                mode=batch_d.mode,
                max_seqlen=batch_d.max_seqlen,
                block_mask=batch_d.block_mask,
            )
            loss = packed_sequence_ce_loss(
                logits,
                batch_d.labels,
                loss_reduction=loss_reduction,
                seq_id=batch_d.seq_id,
                cu_seqlens=batch_d.cu_seqlens,
                token_weights=batch_d.token_weights,
            )
        loss.backward()
        opt.step()
        if end_ev is not None and start_ev is not None:
            end_ev.record()
            torch.cuda.synchronize()
            cp = float(start_ev.elapsed_time(end_ev)) / 1000.0
            t2 = time.perf_counter()
        else:
            t2 = time.perf_counter()
            cp = t2 - t1

        dw = t1 - t0
        data_waits.append(dw)
        computes.append(cp)
        step_totals.append(dw + cp if device.type == "cuda" else (t2 - t0))

        u_fl, e_fl = count_batch_training_flops(base_model, batch_d)
        total_useful_flops += u_fl
        total_exec_flops += e_fl
        total_real += batch_d.real_tokens
        total_slots += batch_d.total_tokens
        last_loss = float(loss.detach().item())
        step_idx += 1

    wall = max(sum(step_totals), 1e-9)
    sum_dw = sum(data_waits)
    sum_cp = sum(computes)
    n_s = max(len(step_totals), 1)
    achieved_u = total_useful_flops / wall
    achieved_e = total_exec_flops / wall
    mfu_str = f"{100.0 * achieved_u / peak_flops:.2f}%" if peak_flops is not None else mfu_na_label

    return AblationRow(
        config=config_name,
        status="ok",
        real_tokens_per_sec=total_real / wall,
        useful_token_fraction=total_real / max(total_slots, 1),
        step_time_ms=1000.0 * statistics.median(step_totals) if step_totals else 0.0,
        data_wait_ms=1000.0 * (sum_dw / n_s),
        compute_ms=1000.0 * (sum_cp / n_s),
        data_wait_fraction=sum_dw / wall,
        useful_tflops_per_sec=achieved_u / 1e12,
        executed_tflops_per_sec=achieved_e / 1e12,
        mfu=mfu_str,
        speedup_vs_baseline=1.0,
        final_loss=last_loss,
        notes=notes,
        real_tokens=total_real,
        total_tokens=total_slots,
    )


def _failed_row(config_name: str, exc: Exception, mfu_na_label: str = "N/A") -> AblationRow:
    """Record a config that raised instead of aborting the whole ablation."""
    short_err = (str(exc).splitlines() or [""])[0][:120]
    return AblationRow(
        config=config_name,
        status=f"failed ({type(exc).__name__})",
        real_tokens_per_sec=0.0,
        useful_token_fraction=0.0,
        step_time_ms=0.0,
        data_wait_ms=0.0,
        compute_ms=0.0,
        data_wait_fraction=0.0,
        useful_tflops_per_sec=0.0,
        executed_tflops_per_sec=0.0,
        mfu=mfu_na_label,
        speedup_vs_baseline=0.0,
        final_loss=float("nan"),
        notes=short_err,
    )


_collate_pack_top = PackCollate(pack_len=512)


def _worker_init_single_thread(_worker_id: int) -> None:
    """Keep each DataLoader worker at 1 thread so workers do not contend with main compute."""
    torch.set_num_threads(1)


def run_fill_gpu_ablation(
    num_sequences: int = 128,
    batch_size: int = 16,
    pack_len: int = 512,
    pack_rows_per_batch: int = 2,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    slow_work_iters: int = 400,
    peak_flops: float | None = None,
    seed: int = 0,
    device: str | torch.device | None = None,
    include_varlen_bf16: bool | None = None,
    include_obfd: bool = False,
    loss_reduction: str = "sequence_mean",
    mfu_na_label: str = "N/A",
) -> tuple[list[AblationRow], list[dict[str, Any]], dict[str, Any]]:
    """Run all Fill-One-GPU toggles and return (rows, profiler_top_ops, flex_probe)."""
    dev = default_device(device)
    run_varlen_bf16 = (dev.type == "cuda") if include_varlen_bf16 is None else include_varlen_bf16
    base_warmup = 2 if dev.type == "cuda" else 1
    compile_warmup = 3

    items = generate_protocol_sequences(
        num_sequences=num_sequences,
        min_len=min(8, pack_len),
        max_len=pack_len,
        seed=seed,
    )
    flex_info = probe_flex_attention(dev)

    def fresh_model(use_flex: bool = False) -> ProtocolTransformer:
        torch.manual_seed(seed)
        return ProtocolTransformer(
            max_seq_len=max(pack_len, 512),
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=4 * d_model,
            causal=False,
            block_size=64,
            use_flex_attention=use_flex,
        ).to(dev)

    rows: list[AblationRow] = []

    # (a) Padded baseline
    padded_batches = build_batches_from_items(items, "padded", batch_size=batch_size)
    m_a = fresh_model()
    row_a = _execute_timed_batches(
        "a_padded_baseline",
        m_a,
        m_a,
        lambda: padded_batches,
        dev,
        use_bf16=False,
        warmup_steps=base_warmup,
        peak_flops=peak_flops,
        mfu_na_label=mfu_na_label,
        loss_reduction=loss_reduction,
    )
    rows.append(row_a)
    base_tps = max(row_a.real_tokens_per_sec, 1e-6)

    # Run torch.profiler snapshot once (sorted by CUDA time on GPU, CPU time on CPU)
    prof_ops = run_torch_profiler_summary(
        fresh_model(), padded_batches[0], use_bf16=(dev.type == "cuda"), loss_reduction=loss_reduction
    )

    # (b) Length-bucketed batching
    bucket_batches = build_batches_from_items(items, "bucketed", batch_size=batch_size)
    m_b = fresh_model()
    rows.append(
        _execute_timed_batches(
            "b_length_bucketed",
            m_b,
            m_b,
            lambda: bucket_batches,
            dev,
            use_bf16=False,
            warmup_steps=base_warmup,
            peak_flops=peak_flops,
            mfu_na_label=mfu_na_label,
            loss_reduction=loss_reduction,
        )
    )

    # (c) Sequence packing with block-diagonal mask & reset position IDs
    packed_batches = build_batches_from_items(
        items, "packed", pack_len=pack_len, pack_rows_per_batch=pack_rows_per_batch
    )
    m_c = fresh_model()
    rows.append(
        _execute_timed_batches(
            "c_sequence_packed",
            m_c,
            m_c,
            lambda: packed_batches,
            dev,
            use_bf16=False,
            warmup_steps=base_warmup,
            peak_flops=peak_flops,
            mfu_na_label=mfu_na_label,
            loss_reduction=loss_reduction,
        )
    )

    # (d) Varlen nested_tensor SDPA (zero padding, zero cross-sequence attention)
    varlen_batches = build_batches_from_items(items, "varlen", batch_size=batch_size)
    m_d = fresh_model()
    rows.append(
        _execute_timed_batches(
            "d_varlen_nested_sdpa",
            m_d,
            m_d,
            lambda: varlen_batches,
            dev,
            use_bf16=False,
            warmup_steps=base_warmup,
            peak_flops=peak_flops,
            mfu_na_label=mfu_na_label,
            loss_reduction=loss_reduction,
            notes="PyTorch jagged nested_tensor + SDPA (is_causal=False)",
        )
    )

    # (e) Sequence packing + BF16 autocast
    m_e = fresh_model()
    rows.append(
        _execute_timed_batches(
            "e_packed_bf16_autocast",
            m_e,
            m_e,
            lambda: packed_batches,
            dev,
            use_bf16=True,
            warmup_steps=base_warmup,
            peak_flops=peak_flops,
            mfu_na_label=mfu_na_label,
            loss_reduction=loss_reduction,
        )
    )

    if include_obfd:
        obfd_batches = build_batches_from_items(
            items, "packed_obfd", pack_len=pack_len, pack_rows_per_batch=pack_rows_per_batch
        )
        m_obfd = fresh_model()
        rows.append(
            _execute_timed_batches(
                "e1_obfd_packed_bf16",
                m_obfd,
                m_obfd,
                lambda: obfd_batches,
                dev,
                use_bf16=True,
                warmup_steps=base_warmup,
                peak_flops=peak_flops,
                mfu_na_label=mfu_na_label,
                loss_reduction=loss_reduction,
                notes="Ding et al. (2024) O(N log N) Segment-Tree OBFD + BF16",
            )
        )

    # (f) torch.compile on packed + BF16 (record honestly if C++ compiler missing or slow)
    m_f_base = fresh_model()
    try:
        m_f_compiled = torch.compile(m_f_base)
        row_f = _execute_timed_batches(
            "f_packed_torch_compile",
            m_f_compiled,
            m_f_base,
            lambda: packed_batches,
            dev,
            use_bf16=True,
            warmup_steps=compile_warmup,
            peak_flops=peak_flops,
            mfu_na_label=mfu_na_label,
            loss_reduction=loss_reduction,
        )
        rows.append(row_f)
    except Exception as exc:
        rows.append(_failed_row("f_packed_torch_compile", exc, mfu_na_label=mfu_na_label))

    # Optional / GPU BF16 packed_varlen (flash_attn_varlen_func / jagged FlashAttention + torch.compile)
    if run_varlen_bf16:
        packed_varlen_batches = build_batches_from_items(
            items, "packed_varlen", pack_len=pack_len, pack_rows_per_batch=pack_rows_per_batch
        )
        varlen_kernel_note = (
            "flash_attn_varlen_func BF16"
            if (_FLASH_ATTN_VARLEN_FUNC is not None and dev.type == "cuda")
            else "BF16 packed_varlen jagged SDPA"
        )
        m_e2 = fresh_model()
        rows.append(
            _execute_timed_batches(
                "e2_packed_varlen_bf16",
                m_e2,
                m_e2,
                lambda: packed_varlen_batches,
                dev,
                use_bf16=True,
                warmup_steps=base_warmup,
                peak_flops=peak_flops,
                mfu_na_label=mfu_na_label,
                loss_reduction=loss_reduction,
                notes=varlen_kernel_note,
            )
        )
        m_f2_base = fresh_model()
        try:
            m_f2_compiled = torch.compile(m_f2_base)
            rows.append(
                _execute_timed_batches(
                    "f2_varlen_bf16_compile",
                    m_f2_compiled,
                    m_f2_base,
                    lambda: packed_varlen_batches,
                    dev,
                    use_bf16=True,
                    warmup_steps=compile_warmup,
                    peak_flops=peak_flops,
                    mfu_na_label=mfu_na_label,
                    loss_reduction=loss_reduction,
                    notes=f"{varlen_kernel_note} + torch.compile",
                )
            )
        except Exception as exc:
            rows.append(_failed_row("f2_varlen_bf16_compile", exc, mfu_na_label=mfu_na_label))

    # (g1) Synchronous slow tokenizer (num_workers=0) vs (g2) Async DataLoader (num_workers=2, prefetch)
    slow_ds = SlowTokenizerDataset(items, slow_work_iters=slow_work_iters)
    pack_collate = PackCollate(pack_len=pack_len)
    dl_pool_size = max(batch_size, pack_rows_per_batch * 8)
    sync_loader = DataLoader(
        slow_ds,
        batch_size=dl_pool_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pack_collate,
    )
    m_g1 = fresh_model()
    rows.append(
        _execute_timed_batches(
            "g1_sync_slow_tokenizer",
            m_g1,
            m_g1,
            lambda: sync_loader,
            dev,
            use_bf16=True,
            warmup_steps=base_warmup,
            peak_flops=peak_flops,
            mfu_na_label=mfu_na_label,
            loss_reduction=loss_reduction,
            notes="num_workers=0 with CPU tokenizer work",
        )
    )

    async_loader = DataLoader(
        slow_ds,
        batch_size=dl_pool_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=_worker_init_single_thread,
        collate_fn=pack_collate,
    )
    m_g2 = fresh_model()
    try:
        rows.append(
            _execute_timed_batches(
                "g2_async_dataloader",
                m_g2,
                m_g2,
                lambda: async_loader,
                dev,
                use_bf16=True,
                warmup_steps=1,
                peak_flops=peak_flops,
                mfu_na_label=mfu_na_label,
                loss_reduction=loss_reduction,
                notes="num_workers=2 (spawn), prefetch_factor=2, pin_memory=(cuda)",
            )
        )
    finally:
        if getattr(async_loader, "_iterator", None) is not None:
            async_loader._iterator._shutdown_workers()

    # (h) Block-sparse (local window W=64 + 4 per-seq global tokens) + BF16
    sparse_batches = build_batches_from_items(
        items, "block_sparse", pack_len=pack_len, pack_rows_per_batch=pack_rows_per_batch
    )
    if flex_info["backward_ok"]:
        try:
            premasked_batches: list[PreparedBatch] = []
            for sb in sparse_batches:
                sb_dev = sb.to(dev, non_blocking=(dev.type == "cuda"))
                assert sb_dev.seq_id is not None
                sb_dev.block_mask = build_flex_block_mask(
                    sb_dev.seq_id, block_size=64, num_global_tokens=4, causal=False
                )
                premasked_batches.append(sb_dev)
            m_h_flex = fresh_model(use_flex=True)
            flex_note = (
                "Compiled FlexAttention W=64+G=4 (cached BlockMask)"
                if flex_info.get("compiled_ok")
                else "FlexAttention W=64+G=4 (cached BlockMask)"
            )
            rows.append(
                _execute_timed_batches(
                    "h_block_sparse_bf16",
                    m_h_flex,
                    m_h_flex,
                    lambda: premasked_batches,
                    dev,
                    use_bf16=True,
                    warmup_steps=(compile_warmup if flex_info.get("compiled_ok") else base_warmup),
                    peak_flops=peak_flops,
                    mfu_na_label=mfu_na_label,
                    loss_reduction=loss_reduction,
                    notes=flex_note,
                )
            )
        except Exception as flex_exc:
            m_h = fresh_model(use_flex=False)
            short_f = (str(flex_exc).splitlines() or [""])[0][:48]
            rows.append(
                _execute_timed_batches(
                    "h_block_sparse_bf16",
                    m_h,
                    m_h,
                    lambda: sparse_batches,
                    dev,
                    use_bf16=True,
                    warmup_steps=base_warmup,
                    peak_flops=peak_flops,
                    mfu_na_label=mfu_na_label,
                    loss_reduction=loss_reduction,
                    notes=f"Tiled SDPA W=64+G=4 (FlexAttn fallback: {short_f})",
                )
            )
    else:
        m_h = fresh_model(use_flex=False)
        rows.append(
            _execute_timed_batches(
                "h_block_sparse_bf16",
                m_h,
                m_h,
                lambda: sparse_batches,
                dev,
                use_bf16=True,
                warmup_steps=base_warmup,
                peak_flops=peak_flops,
                mfu_na_label=mfu_na_label,
                loss_reduction=loss_reduction,
                notes=f"Tiled SDPA W=64+G=4 ({flex_info['note'][:68]})",
            )
        )

    # (i) CUDA Graphs: run on CUDA if available, else mark "GPU only, not yet measured"
    if dev.type == "cuda":
        try:
            # Filter full-sized batches for static graph shape
            full_packed = [b for b in packed_batches if b.input_ids.shape[0] == pack_rows_per_batch]
            if full_packed:
                m_i = fresh_model()
                # optimizer.step() runs inside graph capture, which requires capturable=True
                opt_i = torch.optim.AdamW(m_i.parameters(), lr=3e-4, capturable=True)
                runner = CudaGraphStepRunner(
                    m_i,
                    opt_i,
                    pack_rows=pack_rows_per_batch,
                    pack_len=pack_len,
                    use_bf16=True,
                    loss_reduction=loss_reduction,
                    device=dev,
                )
                runner.replay_step(full_packed[0].to(dev, non_blocking=True))
                torch.cuda.synchronize()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                real_sum = 0
                slot_sum = 0
                u_sum = 0.0
                e_sum = 0.0
                loss_v = 0.0
                for b in full_packed:
                    bd = b.to(dev, non_blocking=True)
                    loss_v = runner.replay_step(bd)
                    uf, ef = count_batch_training_flops(m_i, bd)
                    u_sum += uf
                    e_sum += ef
                    real_sum += bd.real_tokens
                    slot_sum += bd.total_tokens
                end_ev.record()
                torch.cuda.synchronize()
                dt = max(float(start_ev.elapsed_time(end_ev)) / 1000.0, 1e-9)
                rows.append(
                    AblationRow(
                        config="i_cuda_graphs",
                        status="ok",
                        real_tokens_per_sec=real_sum / dt,
                        useful_token_fraction=real_sum / max(slot_sum, 1),
                        step_time_ms=1000.0 * dt / len(full_packed),
                        data_wait_ms=0.0,
                        compute_ms=1000.0 * dt / len(full_packed),
                        data_wait_fraction=0.0,
                        useful_tflops_per_sec=(u_sum / dt) / 1e12,
                        executed_tflops_per_sec=(e_sum / dt) / 1e12,
                        mfu=f"{100.0 * (u_sum / dt) / peak_flops:.2f}%" if peak_flops else mfu_na_label,
                        speedup_vs_baseline=(real_sum / dt) / base_tps,
                        final_loss=loss_v,
                        notes="CUDA Graph static replay (CUDA Event timed)",
                        real_tokens=real_sum,
                        total_tokens=slot_sum,
                    )
                )
            else:
                rows.append(
                    AblationRow(
                        config="i_cuda_graphs",
                        status="skipped (no full static batch)",
                        real_tokens_per_sec=0.0,
                        useful_token_fraction=0.0,
                        step_time_ms=0.0,
                        data_wait_ms=0.0,
                        compute_ms=0.0,
                        data_wait_fraction=0.0,
                        useful_tflops_per_sec=0.0,
                        executed_tflops_per_sec=0.0,
                        mfu=mfu_na_label,
                        speedup_vs_baseline=0.0,
                        final_loss=float("nan"),
                        notes=f"fewer than pack_rows_per_batch={pack_rows_per_batch} packed rows",
                    )
                )
        except Exception as exc:
            rows.append(_failed_row("i_cuda_graphs", exc, mfu_na_label=mfu_na_label))
    else:
        rows.append(
            AblationRow(
                config="i_cuda_graphs",
                status="GPU only, not yet measured",
                real_tokens_per_sec=0.0,
                useful_token_fraction=0.0,
                step_time_ms=0.0,
                data_wait_ms=0.0,
                compute_ms=0.0,
                data_wait_fraction=0.0,
                useful_tflops_per_sec=0.0,
                executed_tflops_per_sec=0.0,
                mfu=mfu_na_label,
                speedup_vs_baseline=0.0,
                final_loss=float("nan"),
                notes="GPU only, not yet measured",
            )
        )

    for r in rows:
        if r.status == "ok":
            r.speedup_vs_baseline = r.real_tokens_per_sec / base_tps

    return rows, prof_ops, flex_info
