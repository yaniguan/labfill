"""Auto-Lab 15-repository wire node for labfill (autolab-wire/v1, Layer L3).

Connects upstream `twinstep` error-twin sequences and `lab-attrib` post-run blame context
to downstream `labbench-posttrain` SFT/MPO/DPO/GRPO/GDPO updates via verified non-overlapping
BF16 sequence packing (`LabFillLean.checkPackedBin_conserves_tokens`), emitting per-bin slice
layouts, `cu_seqlens` varlen offsets, and prompt/completion `loss_mask_spans`.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

REPO_ID = "labfill"
LAYER = "L3"
SCHEMA_VERSION = "autolab-wire/v1"

DEFAULT_WIRE_BIN_CAP = 512
LEAN_DEFAULT_BIN_CAP = 2048
DEFAULT_MAX_BIN_CAP = 8192

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _load_local_curriculum_packer() -> Any:
    """Load `community_extensions/curriculum_difficulty_packer.py` without cross-repo collisions."""
    mod_name = "_labfill_curriculum_difficulty_packer"
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    packer_path = REPO_DIR / "community_extensions" / "curriculum_difficulty_packer.py"
    spec = importlib.util.spec_from_file_location(mod_name, packer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {packer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


try:
    import torch  # noqa: E402

    from labfill.fill_gpu import (  # noqa: E402
        BOS_TOKEN_ID,
        EOS_TOKEN_ID,
        build_cu_seqlens,
        pack_sequence_pairs,
    )

    _packer_mod = _load_local_curriculum_packer()
    PackedCurriculumBin = _packer_mod.PackedCurriculumBin
    PackedSequenceSlice = _packer_mod.PackedSequenceSlice
    materialize_bf16_packed_batch = _packer_mod.materialize_bf16_packed_batch
    verify_packed_bin = _packer_mod.verify_packed_bin

    _HAS_TORCH_LABFILL = True
except ImportError:  # pragma: no cover - minimal environment without PyTorch
    torch = None  # type: ignore[assignment]
    _HAS_TORCH_LABFILL = False


class PurePackedSlice(NamedTuple):
    """Pure-Python mirror of `LabFillLean.PackedSlice` when PyTorch is unavailable."""

    seq_id: int
    offset: int
    length: int
    curriculum_weight: float = 0.25
    prompt_len: int = 0


def _make_slice(
    seq_id: int,
    offset: int,
    length: int,
    weight: float = 0.25,
    prompt_len: int = 0,
) -> Any:
    if _HAS_TORCH_LABFILL:
        return PackedSequenceSlice(
            seq_id=int(seq_id),
            offset=int(offset),
            length=int(length),
            curriculum_weight=float(weight),
            prompt_len=int(prompt_len),
        )
    return PurePackedSlice(
        seq_id=int(seq_id),
        offset=int(offset),
        length=int(length),
        curriculum_weight=float(weight),
        prompt_len=int(prompt_len),
    )


def _next_pow2_at_least(val: int, floor: int = 512) -> int:
    cap = max(1, floor)
    while cap < val:
        cap *= 2
    return cap


def pack_lengths_pure_python(
    token_lengths: list[int],
    bin_cap: int,
    prompt_lengths: list[int] | None = None,
    curriculum_weights: list[float] | None = None,
    seq_ids: list[int] | None = None,
) -> list[list[Any]]:
    """First-fit non-overlapping bin packer matching `labfill.fill_gpu.pack_sequence_pairs`."""
    if bin_cap < 1:
        raise ValueError(f"bin_cap must be >= 1, got {bin_cap}")
    p_lens = [0] * len(token_lengths) if prompt_lengths is None else prompt_lengths
    c_weights = [0.25] * len(token_lengths) if curriculum_weights is None else curriculum_weights
    s_ids = list(range(len(token_lengths))) if seq_ids is None else seq_ids

    if len(p_lens) != len(token_lengths) or len(c_weights) != len(token_lengths) or len(s_ids) != len(token_lengths):
        raise ValueError("Metadata list lengths must match token_lengths")

    bins_used: list[int] = []
    bins_slices: list[list[Any]] = []

    for idx, raw_len in enumerate(token_lengths):
        n = int(raw_len)
        p_len = int(p_lens[idx])
        w = float(c_weights[idx])
        sid = int(s_ids[idx])
        if n <= 0 or n > bin_cap:
            raise ValueError(f"Sequence {sid} length {n} outside [1, {bin_cap}]")
        if p_len < 0 or p_len > n:
            raise ValueError(f"Sequence {sid} prompt_len {p_len} outside [0, {n}]")

        placed = -1
        for b_idx, used in enumerate(bins_used):
            if used + n <= bin_cap:
                placed = b_idx
                break

        if placed < 0:
            placed = len(bins_used)
            bins_used.append(0)
            bins_slices.append([])

        offset = bins_used[placed]
        bins_slices[placed].append(
            _make_slice(seq_id=sid, offset=offset, length=n, weight=w, prompt_len=p_len)
        )
        bins_used[placed] += n

    return bins_slices


def _pack_with_labfill(
    token_lengths: list[int],
    bin_cap: int,
    prompt_lengths: list[int] | None = None,
    curriculum_weights: list[float] | None = None,
    seq_ids: list[int] | None = None,
) -> tuple[list[list[Any]], bool]:
    """Pack sequence lengths using `pack_sequence_pairs`, O(bin_cap) boundary verification & BF16 materializer."""
    if bin_cap < 1:
        raise ValueError(f"bin_cap must be >= 1, got {bin_cap}")

    p_lens = [0] * len(token_lengths) if prompt_lengths is None else prompt_lengths
    c_weights = [0.25] * len(token_lengths) if curriculum_weights is None else curriculum_weights
    s_ids = list(range(len(token_lengths))) if seq_ids is None else seq_ids

    if len(p_lens) != len(token_lengths) or len(c_weights) != len(token_lengths) or len(s_ids) != len(token_lengths):
        raise ValueError("Metadata list lengths must match token_lengths")

    pairs = []
    meta_by_sid: dict[int, tuple[int, int, float]] = {}
    for idx, raw_len in enumerate(token_lengths):
        n = int(raw_len)
        p_len = int(p_lens[idx])
        w = float(c_weights[idx])
        orig_sid = int(s_ids[idx])
        if n <= 0 or n > bin_cap:
            raise ValueError(f"Sequence {orig_sid} length {n} outside [1, {bin_cap}]")
        if p_len < 0 or p_len > n:
            raise ValueError(f"Sequence {orig_sid} prompt_len {p_len} outside [0, {n}]")
        if orig_sid in meta_by_sid:
            raise ValueError(f"Duplicate seq_id {orig_sid} in _pack_with_labfill input")
        meta_by_sid[orig_sid] = (n, p_len, w)

        tok = torch.full((n,), 4 + (idx % 240), dtype=torch.int64)
        tok[0] = BOS_TOKEN_ID
        if n > 1:
            tok[-1] = EOS_TOKEN_ID
        # Tag every token with its exact original sequence ID (>= 0) so bin placement preserves identity
        lbl = torch.full((n,), orig_sid, dtype=torch.int64)
        pairs.append((tok, lbl))

    packed_batches = pack_sequence_pairs(
        pairs,
        pack_len=bin_cap,
        rows_per_batch=1,
        attn_mode="dense",
        truncate=False,
        curriculum_weights=c_weights,
    )

    bins_slices: list[list[Any]] = []
    curriculum_bins: list[Any] = []
    mask_isolated = True
    seen_global_sids: set[int] = set()

    for row_batch in packed_batches:
        assert row_batch.seq_id is not None
        sid_row = row_batch.seq_id[0]
        pos_row = row_batch.position_ids[0]
        lbl_row = row_batch.labels[0]

        # O(bin_cap) sequence isolation & position_ids reset check (zero O(bin_cap^2) mask allocation!)
        cu_row, _max_row = build_cu_seqlens(row_batch.seq_id)
        sid_list = sid_row.tolist()
        pos_list = pos_row.tolist()
        lbl_list = lbl_row.tolist()

        row_slices: list[Any] = []
        seen_row_local_sids: set[int] = set()
        saw_trailing_pad = False
        col = 0
        while col < bin_cap:
            cur_local_sid = int(sid_list[col])
            if cur_local_sid <= 0:
                saw_trailing_pad = True
                col += 1
                continue
            if saw_trailing_pad or cur_local_sid in seen_row_local_sids:
                mask_isolated = False
            seen_row_local_sids.add(cur_local_sid)

            start = col
            while col < bin_cap and int(sid_list[col]) == cur_local_sid:
                col += 1
            length = col - start
            if pos_list[start : start + length] != list(range(length)):
                mask_isolated = False

            orig_sid = int(lbl_list[start])
            if any(int(x) != orig_sid for x in lbl_list[start : start + length]):
                mask_isolated = False
            if orig_sid not in meta_by_sid or orig_sid in seen_global_sids:
                mask_isolated = False
                p_len, w = 0, 0.25
            else:
                seen_global_sids.add(orig_sid)
                exp_len, p_len, w = meta_by_sid[orig_sid]
                if exp_len != length:
                    mask_isolated = False

            row_slices.append(
                _make_slice(seq_id=orig_sid, offset=start, length=length, weight=w, prompt_len=p_len)
            )

        expected_row_lens = [int(s.length) for s in row_slices]
        actual_row_lens = (cu_row[1:] - cu_row[:-1]).tolist() if cu_row.numel() > 1 else []
        if actual_row_lens != expected_row_lens:
            mask_isolated = False

        bins_slices.append(row_slices)
        curriculum_bins.append(PackedCurriculumBin(bin_cap=bin_cap, slices=tuple(row_slices)))

    bf16_batch = materialize_bf16_packed_batch(curriculum_bins, bin_cap=bin_cap)
    if bf16_batch["curriculum_weights_bf16"].dtype != torch.bfloat16:
        mask_isolated = False

    return bins_slices, mask_isolated


def _parse_slice_entry(entry: Any, fallback_seq_id: int) -> Any:
    if hasattr(entry, "seq_id") and hasattr(entry, "offset") and hasattr(entry, "length"):
        return entry
    if isinstance(entry, dict):
        seq_id = entry.get("seq_id", entry.get("seqId", fallback_seq_id))
        offset = entry.get("offset", 0)
        length = entry.get("length", entry.get("len", 0))
        weight = entry.get("curriculum_weight", 0.25)
        prompt_len = entry.get("prompt_len", 0)
        return _make_slice(
            seq_id=int(seq_id),
            offset=int(offset),
            length=int(length),
            weight=float(weight),
            prompt_len=int(prompt_len),
        )
    if isinstance(entry, (list, tuple)):
        if len(entry) == 3:
            return _make_slice(seq_id=int(entry[0]), offset=int(entry[1]), length=int(entry[2]))
        if len(entry) == 2:
            return _make_slice(seq_id=fallback_seq_id, offset=int(entry[0]), length=int(entry[1]))
    raise ValueError(f"Unsupported slice specification: {entry!r}")


def _parse_forced_bins(forced: Any) -> list[list[Any]]:
    items = list(forced)
    if not items:
        return []
    first = items[0]
    is_nested_bins = isinstance(first, (list, tuple)) and (
        len(first) == 0
        or isinstance(first[0], (dict, list, tuple))
        or hasattr(first[0], "offset")
    )
    if is_nested_bins:
        bins: list[list[Any]] = []
        seq_counter = 0
        for raw_bin in items:
            parsed_bin: list[Any] = []
            for raw_sl in raw_bin:
                parsed_bin.append(_parse_slice_entry(raw_sl, seq_counter))
                seq_counter += 1
            bins.append(parsed_bin)
        return bins
    return [[_parse_slice_entry(raw_sl, idx) for idx, raw_sl in enumerate(items)]]


def _inject_overlap_into_bins(bins: list[list[Any]], bin_cap: int, expected_tokens: int) -> list[list[Any]]:
    """Construct an explicit overlapping slice layout to exercise `LabFillLean.checkPackedBin`."""
    safe_cap = max(bin_cap, 2)
    flat = [sl for b in bins for sl in b]
    if len(flat) >= 2:
        s0, s1 = flat[0], flat[1]
        len0 = max(1, min(int(s0.length), safe_cap))
        len1 = max(1, min(int(s1.length), safe_cap))
        overlap_offset = max(0, min(len0 // 2, safe_cap - len1))
        if overlap_offset >= len0:
            overlap_offset = 0
        corrupted_bin0 = [
            _make_slice(seq_id=0, offset=0, length=len0),
            _make_slice(seq_id=1, offset=overlap_offset, length=len1),
        ]
        rest = flat[2:]
        if rest:
            return [
                corrupted_bin0,
                [_make_slice(seq_id=i + 2, offset=0, length=int(s.length)) for i, s in enumerate(rest)],
            ]
        return [corrupted_bin0]

    half = max(1, min(max(expected_tokens, 2) // 2 + 1, safe_cap))
    return [[_make_slice(seq_id=0, offset=0, length=half), _make_slice(seq_id=1, offset=0, length=half)]]


def verify_packed_bins_lean_guard(
    bin_cap: int,
    expected_total_tokens: int,
    bins: list[list[Any]],
) -> tuple[bool, str | None]:
    """Verify `LabFillLean.checkPackedBin` unique-seqId disjointness, capacity, and conservation across bins."""
    if bin_cap < 1:
        return False, "packed_slice_within_bin_cap"
    if expected_total_tokens < 0:
        return False, "packed_real_token_conservation"

    seen_seq_ids: set[int] = set()
    for bin_slices in bins:
        if not bin_slices:
            return False, "packed_slice_within_bin_cap"

        bin_expected = sum(int(s.length) for s in bin_slices)
        if _HAS_TORCH_LABFILL:
            ok, rule = verify_packed_bin(bin_cap, bin_expected, bin_slices)
            if not ok:
                return False, rule

        for s in bin_slices:
            p_len = int(getattr(s, "prompt_len", 0))
            if (
                int(s.length) <= 0
                or int(s.offset) < 0
                or int(s.offset) + int(s.length) > bin_cap
                or p_len < 0
                or p_len > int(s.length)
            ):
                return False, "packed_slice_within_bin_cap"

        for idx_a, a in enumerate(bin_slices):
            for idx_b, b in enumerate(bin_slices):
                if idx_a == idx_b:
                    continue
                if int(a.seq_id) == int(b.seq_id):
                    return False, "packed_slices_non_overlapping"
                disjoint = (
                    int(a.offset) + int(a.length) <= int(b.offset)
                    or int(b.offset) + int(b.length) <= int(a.offset)
                )
                if not disjoint:
                    return False, "packed_slices_non_overlapping"

        for s in bin_slices:
            sid_int = int(s.seq_id)
            if sid_int in seen_seq_ids:
                return False, "packed_slices_non_overlapping"
            seen_seq_ids.add(sid_int)

    packed_sum = sum(int(s.length) for b in bins for s in b)
    if packed_sum != expected_total_tokens:
        return False, "packed_real_token_conservation"

    return True, None


def _extract_curriculum_weights(
    error_twins: dict[str, Any],
    num_seqs: int,
) -> tuple[list[float], str | None]:
    """Compute per-sequence Bernoulli frontier variance weights `p * (1 - p)` from rewards or pass_rates."""
    if "rollout_rewards" in error_twins and error_twins["rollout_rewards"] is not None:
        raw_rw = list(error_twins["rollout_rewards"])
        if len(raw_rw) != num_seqs:
            return [], "rollout_rewards_length_mismatch"
        weights: list[float] = []
        for group_rw in raw_rw:
            if not isinstance(group_rw, (list, tuple)) or len(group_rw) == 0:
                return [], "invalid_rollout_rewards"
            passed = sum(1 for r in group_rw if float(r) > 0.0)
            p = float(passed) / float(len(group_rw))
            weights.append(p * (1.0 - p))
        return weights, None

    if "pass_rates" in error_twins and error_twins["pass_rates"] is not None:
        raw_pr = list(error_twins["pass_rates"])
        if len(raw_pr) != num_seqs:
            return [], "pass_rates_length_mismatch"
        weights = []
        for raw_p in raw_pr:
            p = float(raw_p)
            if p < 0.0 or p > 1.0:
                return [], "invalid_pass_rates"
            weights.append(p * (1.0 - p))
        return weights, None

    return [0.25] * num_seqs, None


def _build_wire_metadata(bins: list[list[Any]]) -> dict[str, Any]:
    """Serialize packed bins into JSON-ready `bins`, `cu_seqlens`, `bin_cu_seqlens`, and `loss_mask_spans`."""
    serialized_bins: list[list[dict[str, Any]]] = []
    flat_cu: list[int] = [0]
    bin_cu_list: list[list[int]] = []
    loss_mask_spans: list[dict[str, Any]] = []
    flat_cursor = 0
    max_seqlen = 0
    total_comp = 0

    for b_idx, bin_slices in enumerate(bins):
        b_dicts: list[dict[str, Any]] = []
        b_cu: list[int] = [0]
        b_acc = 0
        for s in bin_slices:
            sid = int(s.seq_id)
            off = int(s.offset)
            ln = int(s.length)
            p_len = int(getattr(s, "prompt_len", 0))
            c_off = off + p_len
            c_len = ln - p_len
            cw = round(float(getattr(s, "curriculum_weight", 0.25)), 6)

            b_dicts.append(
                {
                    "seq_id": sid,
                    "offset": off,
                    "length": ln,
                    "prompt_len": p_len,
                    "completion_offset": c_off,
                    "completion_len": c_len,
                    "curriculum_weight": cw,
                }
            )
            loss_mask_spans.append(
                {
                    "bin_idx": b_idx,
                    "seq_id": sid,
                    "offset": off,
                    "length": ln,
                    "prompt_len": p_len,
                    "completion_offset": c_off,
                    "completion_len": c_len,
                    "prompt_span": [off, c_off],
                    "completion_span": [c_off, off + ln],
                    "flat_start": flat_cursor,
                    "flat_prompt_end": flat_cursor + p_len,
                    "flat_end": flat_cursor + ln,
                    "curriculum_weight": cw,
                }
            )
            flat_cursor += ln
            flat_cu.append(flat_cursor)
            b_acc += ln
            b_cu.append(b_acc)
            total_comp += c_len
            if ln > max_seqlen:
                max_seqlen = ln

        serialized_bins.append(b_dicts)
        bin_cu_list.append(b_cu)

    return {
        "bins": serialized_bins,
        "cu_seqlens": flat_cu,
        "bin_cu_seqlens": bin_cu_list,
        "max_seqlen": max_seqlen,
        "loss_mask_spans": loss_mask_spans,
        "total_completion_tokens": total_comp,
    }


def packed_batch_to_posttrain_tensors(
    packed_batch: dict[str, Any],
    token_ids: Any = None,
    device: str | Any = "cpu",
    *,
    flat: bool | None = None,
) -> tuple[Any, Any, int, Any, Any]:
    """Materialize `(position_ids, cu_seqlens_int32, max_seqlen, completion_mask_bool, curriculum_weights_bf16)`.

    Ready for direct consumption by `labbench-posttrain` (`labrl` SFT / MPO / DPO / GRPO / GDPO):
      - When `flat=False` (default when `token_ids is None` or 2D `(bin_count, bin_cap)`), returns
        `(bin_count, bin_cap)` `position_ids`, `completion_mask_bool`, and `curriculum_weights_bf16`
        alongside flat `cu_seqlens_int32` (`len == num_seqs + 1`) and `max_seqlen`.
      - When `flat=True` (or `token_ids` is 1D `(total_real_tokens,)`), returns unpadded 1D
        `(total_real_tokens,)` tensors aligned directly with `cu_seqlens_int32`.
    """
    if not _HAS_TORCH_LABFILL:
        raise RuntimeError("packed_batch_to_posttrain_tensors requires PyTorch")
    if not packed_batch.get("zero_cross_seq_overlap", False):
        raise ValueError("Refusing to materialize post-training tensors from halted/invalid packed_batch")

    dev = torch.device(device)
    bin_cap = int(packed_batch["bin_cap"])
    bin_count = int(packed_batch["bin_count"])
    total_real = int(packed_batch["total_real_tokens"])
    bins_data = list(packed_batch.get("bins", []))
    cu_list = [int(x) for x in packed_batch.get("cu_seqlens", [0])]
    max_seqlen = int(packed_batch.get("max_seqlen", 0))

    use_flat = bool(flat) if flat is not None else (token_ids is not None and getattr(token_ids, "ndim", 0) == 1)
    if token_ids is not None:
        if use_flat and int(token_ids.numel()) != total_real:
            raise ValueError(f"1D token_ids length {int(token_ids.numel())} != total_real_tokens={total_real}")
        if not use_flat and tuple(token_ids.shape) != (bin_count, bin_cap):
            raise ValueError(
                f"2D token_ids shape {tuple(token_ids.shape)} != (bin_count={bin_count}, bin_cap={bin_cap})"
            )

    cu_seqlens_int32 = torch.tensor(cu_list, dtype=torch.int32, device=dev)

    if use_flat:
        pos_1d = torch.zeros((total_real,), dtype=torch.int64, device=dev)
        comp_1d = torch.zeros((total_real,), dtype=torch.bool, device=dev)
        weights_1d = torch.zeros((total_real,), dtype=torch.bfloat16, device=dev)
        cursor = 0
        for bin_slices in bins_data:
            for sl in bin_slices:
                ln = int(sl["length"])
                p_len = int(sl.get("prompt_len", 0))
                cw = float(sl.get("curriculum_weight", 0.25))
                pos_1d[cursor : cursor + ln] = torch.arange(ln, dtype=torch.int64, device=dev)
                comp_1d[cursor + p_len : cursor + ln] = True
                weights_1d[cursor : cursor + ln] = torch.tensor(cw, dtype=torch.bfloat16, device=dev)
                cursor += ln
        return pos_1d, cu_seqlens_int32, max_seqlen, comp_1d, weights_1d

    position_ids = torch.zeros((bin_count, bin_cap), dtype=torch.int64, device=dev)
    completion_mask = torch.zeros((bin_count, bin_cap), dtype=torch.bool, device=dev)
    weights_bf16 = torch.zeros((bin_count, bin_cap), dtype=torch.bfloat16, device=dev)

    for b_idx, bin_slices in enumerate(bins_data):
        for sl in bin_slices:
            off = int(sl["offset"])
            ln = int(sl["length"])
            p_len = int(sl.get("prompt_len", 0))
            cw = float(sl.get("curriculum_weight", 0.25))
            end = off + ln
            position_ids[b_idx, off:end] = torch.arange(ln, dtype=torch.int64, device=dev)
            completion_mask[b_idx, off + p_len : end] = True
            weights_bf16[b_idx, off:end] = torch.tensor(cw, dtype=torch.bfloat16, device=dev)

    return position_ids, cu_seqlens_int32, max_seqlen, completion_mask, weights_bf16


def process_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Process an `autolab-wire/v1` packet through `labfill` Layer L3 BF16 sequence packing."""
    out: dict[str, Any] = copy.deepcopy(packet) if packet is not None else {}
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    if out["halted"]:
        return out

    error_twins = out.get("error_twins")
    if not isinstance(error_twins, dict):
        error_twins = {}

    raw_lengths = error_twins.get("token_lengths", [96, 144, 180, 72])
    token_lengths = [int(x) for x in list(raw_lengths)]
    num_raw = len(token_lengths)

    explicit_bin_cap = packet is not None and "bin_cap" in packet
    bin_cap = int(out.get("bin_cap", DEFAULT_WIRE_BIN_CAP))
    max_bin_cap = int(out.get("max_bin_cap", DEFAULT_MAX_BIN_CAP))
    raw_policy = out.get("overflow_policy")
    if raw_policy is None:
        overflow_policy = "halt" if explicit_bin_cap else "auto_lean_cap"
    else:
        overflow_policy = str(raw_policy)

    force_overlap = out.get("force_overlapping_slices")
    drop_zero_var = bool(out.get("drop_zero_variance", error_twins.get("drop_zero_variance", False)))

    # Validate prompt_lengths & curriculum weights upfront
    precheck_violation: str | None = None
    if overflow_policy not in ("halt", "auto_lean_cap", "promote_bin_cap", "quarantine_oversized", "chunk"):
        precheck_violation = f"unsupported_overflow_policy:{overflow_policy}"

    if "prompt_lengths" in error_twins and error_twins["prompt_lengths"] is not None:
        prompt_lengths = [int(x) for x in list(error_twins["prompt_lengths"])]
        if len(prompt_lengths) != num_raw:
            precheck_violation = precheck_violation or "prompt_lengths_length_mismatch"
        else:
            for p_val, t_val in zip(prompt_lengths, token_lengths, strict=True):
                if p_val < 0 or p_val > t_val:
                    precheck_violation = precheck_violation or "prompt_len_out_of_bounds"
                    break
    else:
        prompt_lengths = [0] * num_raw

    curriculum_weights, cw_err = _extract_curriculum_weights(error_twins, num_raw)
    if cw_err is not None:
        precheck_violation = precheck_violation or cw_err

    # Filter zero-variance sequences if requested
    zero_var_dropped = 0
    quarantined_seq_indices: list[int] = []
    work_sids: list[int] = []
    work_lens: list[int] = []
    work_prompts: list[int] = []
    work_weights: list[float] = []

    if precheck_violation is None:
        for idx in range(num_raw):
            if drop_zero_var and curriculum_weights[idx] <= 0.0:
                zero_var_dropped += 1
                continue
            work_sids.append(idx)
            work_lens.append(token_lengths[idx])
            work_prompts.append(prompt_lengths[idx])
            work_weights.append(curriculum_weights[idx])

        if any(n <= 0 for n in work_lens):
            precheck_violation = "non_positive_token_length"
        elif work_lens:
            max_work_len = max(work_lens)
            if overflow_policy == "auto_lean_cap" and max_work_len > bin_cap:
                if max_work_len <= LEAN_DEFAULT_BIN_CAP:
                    bin_cap = LEAN_DEFAULT_BIN_CAP
                elif max_work_len <= max_bin_cap:
                    bin_cap = _next_pow2_at_least(max_work_len, floor=LEAN_DEFAULT_BIN_CAP)
            elif overflow_policy == "promote_bin_cap" and max_work_len > bin_cap:
                if max_work_len <= max_bin_cap:
                    bin_cap = _next_pow2_at_least(max_work_len, floor=max(bin_cap, 1))
            elif overflow_policy == "quarantine_oversized" and bin_cap >= 1:
                kept_sids: list[int] = []
                kept_lens: list[int] = []
                kept_prompts: list[int] = []
                kept_weights: list[float] = []
                for sid, ln, pl, cw in zip(work_sids, work_lens, work_prompts, work_weights, strict=True):
                    if ln > bin_cap:
                        quarantined_seq_indices.append(sid)
                    else:
                        kept_sids.append(sid)
                        kept_lens.append(ln)
                        kept_prompts.append(pl)
                        kept_weights.append(cw)
                work_sids, work_lens, work_prompts, work_weights = (
                    kept_sids,
                    kept_lens,
                    kept_prompts,
                    kept_weights,
                )
            elif overflow_policy == "chunk" and bin_cap >= 1 and max_work_len > bin_cap:
                chunk_sids: list[int] = []
                chunk_lens: list[int] = []
                chunk_prompts: list[int] = []
                chunk_weights: list[float] = []
                next_derived_sid = max(work_sids, default=0) + 1
                for sid, ln, pl, cw in zip(work_sids, work_lens, work_prompts, work_weights, strict=True):
                    if ln <= bin_cap:
                        chunk_sids.append(sid)
                        chunk_lens.append(ln)
                        chunk_prompts.append(pl)
                        chunk_weights.append(cw)
                    else:
                        rem_prompt = pl
                        first_chunk = True
                        for start in range(0, ln, bin_cap):
                            c_ln = min(bin_cap, ln - start)
                            c_pl = min(rem_prompt, c_ln)
                            rem_prompt = max(0, rem_prompt - c_pl)
                            chunk_sids.append(sid if first_chunk else next_derived_sid)
                            if not first_chunk:
                                next_derived_sid += 1
                            first_chunk = False
                            chunk_lens.append(c_ln)
                            chunk_prompts.append(c_pl)
                            chunk_weights.append(cw)
                work_sids, work_lens, work_prompts, work_weights = (
                    chunk_sids,
                    chunk_lens,
                    chunk_prompts,
                    chunk_weights,
                )

    expected_tokens = sum(work_lens)
    inputs_packable = (
        precheck_violation is None
        and (bin_cap >= 1)
        and all(1 <= n <= bin_cap for n in work_lens)
    )
    mask_isolated = True

    if inputs_packable:
        if len(work_lens) == 0:
            bins: list[list[Any]] = []
        elif _HAS_TORCH_LABFILL:
            bins, mask_isolated = _pack_with_labfill(
                work_lens,
                bin_cap,
                prompt_lengths=work_prompts,
                curriculum_weights=work_weights,
                seq_ids=work_sids,
            )
        else:  # pragma: no cover - minimal environment without PyTorch
            bins = pack_lengths_pure_python(
                work_lens,
                bin_cap,
                prompt_lengths=work_prompts,
                curriculum_weights=work_weights,
                seq_ids=work_sids,
            )
    else:
        if work_lens:
            bins = [[_make_slice(seq_id=idx, offset=0, length=n) for idx, n in enumerate(work_lens)]]
        else:
            bins = [[_make_slice(seq_id=0, offset=0, length=0)]]

    if isinstance(force_overlap, (list, tuple)) and len(force_overlap) > 0:
        try:
            bins = _parse_forced_bins(force_overlap)
        except ValueError:
            bins = [[_make_slice(seq_id=0, offset=0, length=0)]]
    elif bool(force_overlap):
        bins = _inject_overlap_into_bins(bins, bin_cap=bin_cap, expected_tokens=expected_tokens)

    guard_ok, violation_rule = verify_packed_bins_lean_guard(bin_cap, expected_tokens, bins)
    if precheck_violation is not None:
        guard_ok = False
        violation_rule = precheck_violation

    if not guard_ok or not mask_isolated:
        out["halted"] = True
        out["halt_reason"] = "L3_LABFILL_PACK_BLOCK"

    if out["halted"]:
        bin_count = 0
        total_tokens = 0
        useful_ratio = 0.0
        wire_meta = {
            "bins": [],
            "cu_seqlens": [0],
            "bin_cu_seqlens": [],
            "max_seqlen": 0,
            "loss_mask_spans": [],
            "total_completion_tokens": 0,
        }
    else:
        bin_count = len(bins)
        total_tokens = sum(int(s.length) for b in bins for s in b)
        allocated_slots = bin_count * bin_cap
        useful_ratio = float(total_tokens) / float(allocated_slots) if allocated_slots > 0 else 0.0
        wire_meta = _build_wire_metadata(bins)

    out["packed_batch"] = {
        "bin_cap": bin_cap,
        "bin_count": bin_count,
        "total_real_tokens": total_tokens,
        "useful_token_ratio": round(useful_ratio, 4),
        "zero_cross_seq_overlap": not out["halted"],
        **wire_meta,
        "zero_variance_dropped": zero_var_dropped,
        "quarantined_seq_indices": quarantined_seq_indices,
    }

    backend_label = "labfill.fill_gpu+bf16" if _HAS_TORCH_LABFILL else "pure_python_fallback"
    if out["halted"]:
        summary = (
            f"BLOCK ({out['halt_reason']}): Lean packed-bin guard rejected layout "
            f"(rule={violation_rule or 'cross_seq_mask_leak'}, bin_cap={bin_cap})."
        )
    else:
        summary = (
            f"Packed {len(work_lens)} protocol/twin sequences ({total_tokens} tokens, "
            f"{wire_meta['total_completion_tokens']} completion tokens) into "
            f"{bin_count} non-overlapping BF16 bin(s) (bin_cap={bin_cap}, "
            f"useful_ratio={useful_ratio:.4f}, backend={backend_label})."
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": "BLOCK" if out["halted"] else "PASS",
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "invariant": "checkPackedBin_conserves_tokens",
            "first_principles": (
                "Packed BF16 sequence bins reset position_ids per sequence, enforce pairwise disjoint "
                "[offset, offset + len) intervals with unique seq_id per bin, emit exact cu_seqlens "
                "and prompt/completion loss_mask_spans, and conserve exact real token counts."
            ),
        }
    )
    return out


if __name__ == "__main__":
    sample_in = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(process_packet(sample_in), indent=2))
