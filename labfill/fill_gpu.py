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

from dataclasses import dataclass, field
import hashlib
import math
import statistics
import time
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from labfill.protocol import default_device

VOCAB_SIZE = 256
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
NUM_ERROR_CLASSES = 8  # 0 = valid token, 1..7 = seeded error class
IGNORE_LABEL = -100


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
    mode: str                         # 'dense', 'varlen', 'block_sparse'
    input_ids: Tensor                 # (B, S) or (T_real,)
    position_ids: Tensor              # (B, S) or (T_real,)
    labels: Tensor                    # (B, S) or (T_real,)
    seq_id: Tensor | None = None      # (B, S): 0=pad, 1..K=packed sequence id within row
    cu_seqlens: Tensor | None = None  # (N_seq + 1,) int64 offsets for jagged varlen
    seq_lens: list[int] = field(default_factory=list)
    real_tokens: int = 0
    total_tokens: int = 0

    def to(self, device: torch.device, non_blocking: bool = False) -> "PreparedBatch":
        return PreparedBatch(
            mode=self.mode,
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            position_ids=self.position_ids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            seq_id=self.seq_id.to(device, non_blocking=non_blocking) if self.seq_id is not None else None,
            cu_seqlens=self.cu_seqlens.to(device, non_blocking=non_blocking)
            if self.cu_seqlens is not None
            else None,
            seq_lens=self.seq_lens,
            real_tokens=self.real_tokens,
            total_tokens=self.total_tokens,
        )


def collate_padded_pairs(pairs: list[tuple[Tensor, Tensor]], pad_multiple: int = 1) -> PreparedBatch:
    """Pad a list of (tokens, labels) tensors to the max length in the batch."""
    b = len(pairs)
    lens = [int(t.numel()) for t, _ in pairs]
    max_s = max(lens)
    if pad_multiple > 1:
        max_s = ((max_s + pad_multiple - 1) // pad_multiple) * pad_multiple

    input_ids = torch.full((b, max_s), PAD_TOKEN_ID, dtype=torch.int64)
    position_ids = torch.zeros((b, max_s), dtype=torch.int64)
    labels = torch.full((b, max_s), IGNORE_LABEL, dtype=torch.int64)
    seq_id = torch.zeros((b, max_s), dtype=torch.int64)

    for i, (tok, lbl) in enumerate(pairs):
        n = int(tok.numel())
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
        seq_lens=lens,
        real_tokens=sum(lens),
        total_tokens=b * max_s,
    )


def pack_sequence_pairs(
    pairs: list[tuple[Tensor, Tensor]],
    pack_len: int = 512,
    rows_per_batch: int = 4,
    attn_mode: str = "dense",
) -> list[PreparedBatch]:
    """First-fit decreasing / greedy bin-pack sequences into fixed-length rows of `pack_len`.

    Resets `position_ids` to `0..len-1` per packed sequence and tags each sequence with a
    distinct `seq_id >= 1` (`0` = trailing pad) for exact block-diagonal attention masking.
    """
    rows_tok: list[Tensor] = []
    rows_pos: list[Tensor] = []
    rows_lbl: list[Tensor] = []
    rows_sid: list[Tensor] = []
    rows_lens: list[list[int]] = []

    # Open bins tracking remaining capacity
    bins_used: list[int] = []
    bins_next_sid: list[int] = []

    for tok, lbl in pairs:
        n = min(int(tok.numel()), pack_len)
        tok = tok[:n]
        lbl = lbl[:n]

        placed = -1
        for b_idx, used in enumerate(bins_used):
            if used + n <= pack_len:
                placed = b_idx
                break

        if placed < 0:
            placed = len(bins_used)
            bins_used.append(0)
            bins_next_sid.append(1)
            rows_tok.append(torch.full((pack_len,), PAD_TOKEN_ID, dtype=torch.int64))
            rows_pos.append(torch.zeros((pack_len,), dtype=torch.int64))
            rows_lbl.append(torch.full((pack_len,), IGNORE_LABEL, dtype=torch.int64))
            rows_sid.append(torch.zeros((pack_len,), dtype=torch.int64))
            rows_lens.append([])

        start = bins_used[placed]
        sid = bins_next_sid[placed]
        rows_tok[placed][start : start + n] = tok
        rows_pos[placed][start : start + n] = torch.arange(n, dtype=torch.int64)
        rows_lbl[placed][start : start + n] = lbl
        rows_sid[placed][start : start + n] = sid
        rows_lens[placed].append(n)

        bins_used[placed] += n
        bins_next_sid[placed] += 1

    batches: list[PreparedBatch] = []
    for i in range(0, len(rows_tok), rows_per_batch):
        chunk_tok = torch.stack(rows_tok[i : i + rows_per_batch], dim=0)
        chunk_pos = torch.stack(rows_pos[i : i + rows_per_batch], dim=0)
        chunk_lbl = torch.stack(rows_lbl[i : i + rows_per_batch], dim=0)
        chunk_sid = torch.stack(rows_sid[i : i + rows_per_batch], dim=0)
        flat_lens = [ln for sub in rows_lens[i : i + rows_per_batch] for ln in sub]
        batches.append(
            PreparedBatch(
                mode=attn_mode,
                input_ids=chunk_tok,
                position_ids=chunk_pos,
                labels=chunk_lbl,
                seq_id=chunk_sid,
                seq_lens=flat_lens,
                real_tokens=sum(flat_lens),
                total_tokens=int(chunk_tok.numel()),
            )
        )
    return batches


def collate_pack_for_dataloader(
    pairs: list[tuple[Tensor, Tensor]],
    pack_len: int = 512,
) -> PreparedBatch:
    """Collate function for DataLoader workers: packs one micro-pool of sequences into rows."""
    packed_list = pack_sequence_pairs(
        pairs, pack_len=pack_len, rows_per_batch=max(len(pairs), 1), attn_mode="dense"
    )
    return packed_list[0]


def collate_varlen_pairs(pairs: list[tuple[Tensor, Tensor]]) -> PreparedBatch:
    """Concatenate sequences into 1D flat tensors + jagged offsets (zero padding)."""
    lens = [int(t.numel()) for t, _ in pairs]
    toks = torch.cat([t for t, _ in pairs], dim=0)
    lbls = torch.cat([l for _, l in pairs], dim=0)
    pos = torch.cat([torch.arange(n, dtype=torch.int64) for n in lens], dim=0)
    offsets = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int64)
    total = int(toks.numel())
    return PreparedBatch(
        mode="varlen",
        input_ids=toks,
        position_ids=pos,
        labels=lbls,
        cu_seqlens=offsets,
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
    """Pre-build batches for compute-side ablations: padded, bucketed, packed, varlen, block_sparse."""
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


def tiled_local_global_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    seq_id: Tensor,
    block_size: int = 64,
    num_global_tokens: int = 4,
    causal: bool = False,
) -> Tensor:
    """Tiled block-sparse attention: each block of W queries attends to [prev_block, cur_block] + G globals.

    Respects `seq_id` boundaries so packed sequences never leak across sequence borders.
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

    # Global tokens (first G tokens of each row, broadcast to all blocks)
    k_glob = k[:, :, :g_cnt, :].unsqueeze(2).expand(b, h, n_blk, g_cnt, d)
    v_glob = v[:, :, :g_cnt, :].unsqueeze(2).expand(b, h, n_blk, g_cnt, d)

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
    sid_glob = seq_id[:, :g_cnt].unsqueeze(1).expand(b, n_blk, g_cnt)
    sid_kv = torch.cat([sid_glob, sid_local], dim=2).reshape(b * n_blk, kv_len)
    sid_q = sid_blocks.reshape(b * n_blk, w)

    same_seq = (sid_q.unsqueeze(2) == sid_kv.unsqueeze(1)) & (sid_q.unsqueeze(2) > 0)

    # Drop global slots whose key already sits in the local [prev_blk, cur_blk] window
    # (blocks 0 and 1), so no key is counted twice in the softmax.
    blk_start = torch.arange(n_blk, device=q.device) * w
    glob_dup = torch.arange(g_cnt, device=q.device).unsqueeze(0) >= (blk_start - w).unsqueeze(1)
    keep_kv = torch.cat(
        [~glob_dup, torch.ones((n_blk, 2 * w), dtype=torch.bool, device=q.device)], dim=1
    )  # (N_blk, kv_len)
    same_seq = same_seq & keep_kv.repeat(b, 1).unsqueeze(1)

    if causal:
        pos_row = torch.arange(s, device=q.device).unsqueeze(0).expand(b, s)
        pos_blocks = pos_row.view(b, n_blk, w)
        pos_pad = torch.full((b, 1, w), s + 1, dtype=torch.int64, device=q.device)
        pos_padded = torch.cat([pos_pad, pos_blocks], dim=1)
        pos_local = torch.cat([pos_padded[:, :-1], pos_padded[:, 1:]], dim=2)
        pos_glob = pos_row[:, :g_cnt].unsqueeze(1).expand(b, n_blk, g_cnt)
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


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024, causal: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

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
        block_size: int = 64,
    ) -> Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        if block_sparse and seq_id is not None:
            attn_out = tiled_local_global_attention(
                q, k, v, seq_id=seq_id, block_size=block_size, causal=self.causal
            )
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        x = x + self.out_proj(attn_out.transpose(1, 2).contiguous().view(b, s, d))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_varlen(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        t_tok, d = x.shape
        q_raw, k_raw, v_raw = self.qkv(self.ln1(x)).chunk(3, dim=-1)
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

        attn_nt = F.scaled_dot_product_attention(q_nt, k_nt, v_nt, is_causal=self.causal)
        attn_flat = attn_nt.transpose(1, 2).values().contiguous().view(t_tok, d)
        x = x + self.out_proj(attn_flat)
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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.num_classes = num_classes
        self.causal = causal
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, causal=causal)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        seq_id: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        mode: str = "dense",
    ) -> Tensor:
        x = self.tok_emb(input_ids) + self.pos_emb(position_ids.clamp(0, self.pos_emb.num_embeddings - 1))

        if mode == "varlen":
            if cu_seqlens is None:
                raise ValueError("cu_seqlens is required when mode='varlen'")
            for blk in self.blocks:
                x = blk.forward_varlen(x, cu_seqlens)
            return self.head(self.ln_f(x))

        block_sparse = mode == "block_sparse"
        attn_mask = None
        if not block_sparse and seq_id is not None:
            attn_mask = build_block_diag_mask(seq_id, causal=self.causal)

        for blk in self.blocks:
            x = blk.forward_dense(
                x,
                attn_mask=attn_mask,
                seq_id=seq_id,
                block_sparse=block_sparse,
                block_size=self.block_size,
            )
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------------------
# 4. Analytic FLOP counter, FlexAttention probe, CUDA Graph runner, and profiler
# ---------------------------------------------------------------------------


def count_batch_training_flops(
    model: ProtocolTransformer,
    batch: PreparedBatch,
) -> tuple[float, float]:
    """Return (useful_flops, executed_flops) for one forward + backward step (3x fwd FLOPs).

    Separates real protocol tokens and true within-sequence attention from wasted pad tokens
    and masked cross-sequence attention entries.
    """
    l = model.n_layers
    d = model.d_model
    f = model.d_ff
    c_out = model.num_classes

    # Forward linear FLOPs per token: QKV (6 d^2) + Out (2 d^2) + FFN (4 d f) per layer + head (2 d c_out)
    linear_fwd_per_tok = float(l * (8 * d * d + 4 * d * f) + 2 * d * c_out)

    real_toks = float(batch.real_tokens)
    sum_sq_real = float(sum(s * s for s in batch.seq_lens))
    useful_fwd = linear_fwd_per_tok * real_toks + float(4 * l * d) * sum_sq_real

    if batch.mode == "varlen":
        exec_fwd = useful_fwd
    elif batch.mode == "block_sparse":
        b_rows, s_len = batch.input_ids.shape
        eff_kv = min(s_len, 2 * model.block_size + 4)
        exec_fwd = linear_fwd_per_tok * float(batch.total_tokens) + float(
            4 * l * d * b_rows * s_len * eff_kv
        )
    else:
        b_rows, s_len = batch.input_ids.shape
        exec_fwd = linear_fwd_per_tok * float(batch.total_tokens) + float(
            4 * l * d * b_rows * s_len * s_len
        )

    return 3.0 * useful_fwd, 3.0 * exec_fwd


def probe_flex_attention(device: str | torch.device | None = None) -> dict[str, Any]:
    """Check whether PyTorch FlexAttention supports forward and backward on `device`."""
    dev = default_device(device)
    info: dict[str, Any] = {"device": str(dev), "forward_ok": False, "backward_ok": False, "note": ""}
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def local_win(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            return (q_idx - kv_idx).abs() <= 32

        bm = create_block_mask(local_win, B=None, H=None, Q_LEN=128, KV_LEN=128, device=str(dev))
        q = torch.randn(2, 4, 128, 32, device=dev, requires_grad=True)
        out = flex_attention(q, q, q, block_mask=bm)
        info["forward_ok"] = True
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
        device: torch.device | None = None,
    ) -> None:
        dev = default_device(device)
        if dev.type != "cuda":
            raise RuntimeError("CudaGraphStepRunner requires a CUDA device")
        self.device = dev
        self.model = model
        self.optimizer = optimizer
        self.use_bf16 = use_bf16

        self.static_input_ids = torch.zeros((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_pos_ids = torch.zeros((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_seq_id = torch.ones((pack_rows, pack_len), dtype=torch.int64, device=dev)
        self.static_labels = torch.zeros((pack_rows, pack_len), dtype=torch.int64, device=dev)
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
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                self.static_labels.view(-1),
                ignore_index=IGNORE_LABEL,
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
        if not self._captured:
            self.capture()
        self.static_input_ids.copy_(batch.input_ids, non_blocking=True)
        self.static_pos_ids.copy_(batch.position_ids, non_blocking=True)
        if batch.seq_id is not None:
            self.static_seq_id.copy_(batch.seq_id, non_blocking=True)
        self.static_labels.copy_(batch.labels, non_blocking=True)
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


def run_torch_profiler_summary(
    model: ProtocolTransformer,
    batch: PreparedBatch,
    use_bf16: bool = False,
) -> list[dict[str, Any]]:
    """Run one `torch.profiler.profile` step and extract the top 5 CPU/CUDA ops by self time."""
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
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                b_dev.labels.reshape(-1),
                ignore_index=IGNORE_LABEL,
            )
        loss.backward()
        opt.step()

    events = sorted(
        prof.key_averages(),
        key=lambda e: float(getattr(e, "self_cpu_time_total", 0.0)),
        reverse=True,
    )[:5]
    return [
        {
            "op": ev.key,
            "calls": int(ev.count),
            "self_cpu_ms": round(float(ev.self_cpu_time_total) / 1000.0, 3),
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
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                wb_d.labels.reshape(-1),
                ignore_index=IGNORE_LABEL,
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

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(
                batch_d.input_ids,
                batch_d.position_ids,
                seq_id=batch_d.seq_id,
                cu_seqlens=batch_d.cu_seqlens,
                mode=batch_d.mode,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch_d.labels.reshape(-1),
                ignore_index=IGNORE_LABEL,
            )
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        dw = t1 - t0
        cp = t2 - t1
        data_waits.append(dw)
        computes.append(cp)
        step_totals.append(t2 - t0)

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
    mfu_str = f"{100.0 * achieved_u / peak_flops:.2f}%" if peak_flops is not None else "N/A"

    return AblationRow(
        config=config_name,
        status="ok",
        real_tokens_per_sec=total_real / wall,
        useful_token_fraction=total_real / max(total_slots, 1),
        step_time_ms=1000.0 * statistics.median(step_totals),
        data_wait_ms=1000.0 * (sum_dw / n_s),
        compute_ms=1000.0 * (sum_cp / n_s),
        data_wait_fraction=sum_dw / wall,
        useful_tflops_per_sec=achieved_u / 1e12,
        executed_tflops_per_sec=achieved_e / 1e12,
        mfu=mfu_str,
        speedup_vs_baseline=1.0,
        final_loss=last_loss,
        notes=notes,
    )


def _failed_row(config_name: str, exc: Exception) -> AblationRow:
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
        mfu="N/A",
        speedup_vs_baseline=0.0,
        final_loss=float("nan"),
        notes=short_err,
    )


def _collate_pack_top(pairs: list[tuple[Tensor, Tensor]]) -> PreparedBatch:
    """Top-level picklable collate function for multiprocessing DataLoader."""
    return collate_pack_for_dataloader(pairs, pack_len=512)


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
) -> tuple[list[AblationRow], list[dict[str, Any]], dict[str, Any]]:
    """Run all Fill-One-GPU toggles and return (rows, profiler_top_ops, flex_probe)."""
    dev = default_device(device)
    items = generate_protocol_sequences(num_sequences=num_sequences, seed=seed)
    flex_info = probe_flex_attention(dev)

    def fresh_model() -> ProtocolTransformer:
        torch.manual_seed(seed)
        return ProtocolTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=4 * d_model,
            causal=False,
            block_size=64,
        ).to(dev)

    rows: list[AblationRow] = []

    # (a) Padded baseline
    padded_batches = build_batches_from_items(items, "padded", batch_size=batch_size)
    m_a = fresh_model()
    row_a = _execute_timed_batches(
        "a_padded_baseline", m_a, m_a, lambda: padded_batches, dev, use_bf16=False, peak_flops=peak_flops
    )
    rows.append(row_a)
    base_tps = max(row_a.real_tokens_per_sec, 1e-6)

    # Run torch.profiler snapshot on packed vs padded once
    prof_ops = run_torch_profiler_summary(fresh_model(), padded_batches[0], use_bf16=False)

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
            peak_flops=peak_flops,
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
            peak_flops=peak_flops,
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
            peak_flops=peak_flops,
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
            peak_flops=peak_flops,
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
            peak_flops=peak_flops,
        )
        rows.append(row_f)
    except Exception as exc:
        rows.append(_failed_row("f_packed_torch_compile", exc))

    # (g1) Synchronous slow tokenizer (num_workers=0) vs (g2) Async DataLoader (num_workers=4, prefetch)
    slow_ds = SlowTokenizerDataset(items, slow_work_iters=slow_work_iters)
    sync_loader = DataLoader(
        slow_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_pack_top,
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
            peak_flops=peak_flops,
            notes="num_workers=0 with CPU tokenizer work",
        )
    )

    async_loader = DataLoader(
        slow_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=True,
        multiprocessing_context="spawn",
        worker_init_fn=_worker_init_single_thread,
        collate_fn=_collate_pack_top,
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
                notes="num_workers=2 (spawn), prefetch_factor=2, pin_memory=(cuda)",
            )
        )
    finally:
        if getattr(async_loader, "_iterator", None) is not None:
            async_loader._iterator._shutdown_workers()

    # (h) Tiled block-sparse (local window W=64 + 4 global tokens) + BF16
    sparse_batches = build_batches_from_items(
        items, "block_sparse", pack_len=pack_len, pack_rows_per_batch=pack_rows_per_batch
    )
    m_h = fresh_model()
    flex_note = "FlexAttn fwd+bwd ok" if flex_info["backward_ok"] else f"Tiled W=64+G=4 ({flex_info['note'][:68]})"
    rows.append(
        _execute_timed_batches(
            "h_block_sparse_bf16",
            m_h,
            m_h,
            lambda: sparse_batches,
            dev,
            use_bf16=True,
            peak_flops=peak_flops,
            notes=flex_note,
        )
    )

    # (i) CUDA Graphs: run on CUDA if available, else mark "GPU only, not yet measured"
    if dev.type == "cuda":
        try:
            m_i = fresh_model()
            # optimizer.step() runs inside graph capture, which requires capturable=True
            opt_i = torch.optim.AdamW(m_i.parameters(), lr=3e-4, capturable=True)
            # Filter full-sized batches for static graph shape
            full_packed = [b for b in packed_batches if b.input_ids.shape[0] == pack_rows_per_batch]
            runner = CudaGraphStepRunner(
                m_i, opt_i, pack_rows=pack_rows_per_batch, pack_len=pack_len, use_bf16=True, device=dev
            )
            if full_packed:
                runner.replay_step(full_packed[0].to(dev, non_blocking=True))
                torch.cuda.synchronize()
                t0 = time.perf_counter()
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
                torch.cuda.synchronize()
                dt = max(time.perf_counter() - t0, 1e-9)
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
                        mfu=f"{100.0 * (u_sum / dt) / peak_flops:.2f}%" if peak_flops else "N/A",
                        speedup_vs_baseline=(real_sum / dt) / base_tps,
                        final_loss=loss_v,
                        notes="CUDA Graph static replay",
                    )
                )
        except Exception as exc:
            rows.append(_failed_row("i_cuda_graphs", exc))
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
                mfu="N/A",
                speedup_vs_baseline=0.0,
                final_loss=float("nan"),
                notes="GPU only, not yet measured",
            )
        )

    for r in rows:
        if r.status == "ok":
            r.speedup_vs_baseline = r.real_tokens_per_sec / base_tps

    return rows, prof_ops, flex_info
