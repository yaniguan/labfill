"""Recipe 2: Seeded-error twins, verifier-gated candidate generation, and batched single-edit repair."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from labfill.checker import CheckResult, check_protocols_batched
from labfill.protocol import (
    ERR_BAD_PIPETTE,
    ERR_BAD_TEMP,
    ERR_FROZEN,
    ERR_NAMES,
    ERR_NO_TIP,
    ERR_OK,
    ERR_OVERFLOW,
    ERR_SEALED,
    OP_CENTRIFUGE,
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_MIX,
    OP_NAMES,
    OP_NOOP,
    OP_PICK_TIP,
    OP_SEAL,
    OP_THAW,
    OP_TRANSFER,
    OP_UNSEAL,
    PIP_P2,
    PIP_P20,
    PIP_P200,
    PIP_P1000,
    PIPETTE_NAMES,
    PIPETTE_RANGES_TENTHS_UL,
    LabStateBatch,
    ProtocolBatch,
    default_device,
    make_random_valid_protocols,
)

TWIN_ERROR_TYPES: tuple[str, ...] = (
    "overflow",
    "wrong_pipette",
    "missing_tip",
    "skip_thaw",
    "sealed_plate",
    "bad_temp",
    "swapped_steps",
)

EXPECTED_PRIMARY_CODES: dict[str, int] = {
    "overflow": ERR_OVERFLOW,
    "wrong_pipette": ERR_BAD_PIPETTE,
    "missing_tip": ERR_NO_TIP,
    "skip_thaw": ERR_FROZEN,
    "sealed_plate": ERR_SEALED,
    "bad_temp": ERR_BAD_TEMP,
}


def _extend_steps(proto: ProtocolBatch, extra: int = 1) -> ProtocolBatch:
    """Append `extra` padded NOOP slots to the end of a ProtocolBatch."""
    b, _ = proto.op.shape
    dev = proto.device
    z = torch.zeros((b, extra), dtype=torch.int64, device=dev)
    p = torch.ones((b, extra), dtype=torch.bool, device=dev)
    return ProtocolBatch(
        op=torch.cat([proto.op, z], dim=1),
        src=torch.cat([proto.src, z], dim=1),
        dst=torch.cat([proto.dst, z], dim=1),
        vol=torch.cat([proto.vol, z], dim=1),
        temp=torch.cat([proto.temp, z], dim=1),
        duration=torch.cat([proto.duration, z], dim=1),
        pipette=torch.cat([proto.pipette, z], dim=1),
        rpm=torch.cat([proto.rpm, z], dim=1),
        pad_mask=torch.cat([proto.pad_mask, p], dim=1),
    )


def _insert_step_before(
    proto: ProtocolBatch,
    pos: Tensor,
    new_op: Tensor,
    new_src: Tensor | None = None,
    new_dst: Tensor | None = None,
    new_vol: Tensor | None = None,
    new_temp: Tensor | None = None,
    new_dur: Tensor | None = None,
    new_pip: Tensor | None = None,
    new_rpm: Tensor | None = None,
) -> ProtocolBatch:
    """Vectorized insertion of one step at `pos[b]` (in `0..max_steps`) into an extended (B, T+1) batch."""
    ext = _extend_steps(proto, extra=1)
    b, t_plus = ext.op.shape
    dev = ext.device
    cols = torch.arange(t_plus, device=dev).unsqueeze(0).expand(b, t_plus)
    pos_u = pos.clamp(0, t_plus - 1).unsqueeze(1)

    # For c < pos: take ext[:, c]; for c == pos: inserted; for c > pos: take ext[:, c - 1]
    src_gather = torch.where(cols < pos_u, cols, (cols - 1).clamp(min=0))

    out_op = ext.op.gather(1, src_gather)
    out_src = ext.src.gather(1, src_gather)
    out_dst = ext.dst.gather(1, src_gather)
    out_vol = ext.vol.gather(1, src_gather)
    out_temp = ext.temp.gather(1, src_gather)
    out_dur = ext.duration.gather(1, src_gather)
    out_pip = ext.pipette.gather(1, src_gather)
    out_rpm = ext.rpm.gather(1, src_gather)
    out_pad = ext.pad_mask.gather(1, src_gather)

    at_pos = cols == pos_u
    z = torch.zeros_like(new_op)
    s_val = new_src if new_src is not None else z
    d_val = new_dst if new_dst is not None else z
    v_val = new_vol if new_vol is not None else z
    t_val = new_temp if new_temp is not None else z
    dur_val = new_dur if new_dur is not None else z
    pip_val = new_pip if new_pip is not None else z
    rpm_val = new_rpm if new_rpm is not None else z

    out_op = torch.where(at_pos, new_op.unsqueeze(1), out_op)
    out_src = torch.where(at_pos, s_val.unsqueeze(1), out_src)
    out_dst = torch.where(at_pos, d_val.unsqueeze(1), out_dst)
    out_vol = torch.where(at_pos, v_val.unsqueeze(1), out_vol)
    out_temp = torch.where(at_pos, t_val.unsqueeze(1), out_temp)
    out_dur = torch.where(at_pos, dur_val.unsqueeze(1), out_dur)
    out_pip = torch.where(at_pos, pip_val.unsqueeze(1), out_pip)
    out_rpm = torch.where(at_pos, rpm_val.unsqueeze(1), out_rpm)
    out_pad = torch.where(at_pos, torch.zeros_like(at_pos), out_pad)

    return ProtocolBatch(
        op=out_op,
        src=out_src,
        dst=out_dst,
        vol=out_vol,
        temp=out_temp,
        duration=out_dur,
        pipette=out_pip,
        rpm=out_rpm,
        pad_mask=out_pad,
    )


def _first_matching_step(mask: Tensor) -> Tensor:
    """Return first True index per row along dim=1 (0 if none)."""
    # argmax on int gives first occurrence of max (1)
    return mask.to(torch.int64).argmax(dim=1)


def _require_step_mask(mask: Tensor, error_type: str, desc: str) -> None:
    if not bool(mask.any(dim=1).all().item()):
        raise ValueError(
            f"Cannot construct {error_type!r} error twin: at least one protocol has no {desc} step"
        )


def make_error_twin(
    init_state: LabStateBatch,
    valid_proto: ProtocolBatch,
    error_type: str,
    seed: int = 0,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Create a corrupted twin for each valid protocol in the batch with one known error.

    All 7 twin types return normalized step length `valid_proto.max_steps + 1`.
    Raises ValueError if any protocol in the batch lacks the required target operation.
    """
    if error_type not in TWIN_ERROR_TYPES:
        raise ValueError(f"Unknown error_type={error_type!r}; expected one of {TWIN_ERROR_TYPES}")
    if init_state.batch_size != valid_proto.batch_size:
        raise ValueError(
            f"Batch size mismatch: init_state={init_state.batch_size} != valid_proto={valid_proto.batch_size}"
        )
    if init_state.device != valid_proto.device:
        raise ValueError(
            f"Device mismatch: init_state on {init_state.device} != valid_proto on {valid_proto.device}"
        )

    st = init_state.clone()
    pr = valid_proto.clone()
    b = pr.batch_size
    dev = pr.device
    if b == 0:
        return st, _extend_steps(pr, extra=1)

    b_idx = torch.arange(b, device=dev)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1337)

    active = (~pr.pad_mask) & (pr.op != OP_NOOP)
    is_xfer = active & (pr.op == OP_TRANSFER)
    first_xfer = _first_matching_step(is_xfer)

    if error_type == "overflow":
        _require_step_mask(is_xfer, error_type, "OP_TRANSFER")
        # Pick first transfer; set vol above destination capacity while keeping pipette in range (P1000: 1000..10000)
        d_c = pr.dst[b_idx, first_xfer]
        d_cap = st.capacity[b_idx, d_c]
        over_vol = (d_cap + 500).clamp(1000, 7500)
        pr.vol[b_idx, first_xfer] = over_vol
        pr.pipette[b_idx, first_xfer] = PIP_P1000
        return st, _extend_steps(pr, extra=1)

    if error_type == "wrong_pipette":
        _require_step_mask(is_xfer, error_type, "OP_TRANSFER")
        cur_vol = pr.vol[b_idx, first_xfer]
        # Switch to a pipette whose range definitely excludes cur_vol
        bad_pip = torch.where(cur_vol < 1000, PIP_P1000, PIP_P2)
        pr.pipette[b_idx, first_xfer] = bad_pip
        return st, _extend_steps(pr, extra=1)

    if error_type == "missing_tip":
        is_pick = active & (pr.op == OP_PICK_TIP)
        _require_step_mask(is_pick, error_type, "OP_PICK_TIP")
        first_pick = _first_matching_step(is_pick)
        pr.op[b_idx, first_pick] = OP_NOOP
        return st, _extend_steps(pr, extra=1)

    if error_type == "skip_thaw":
        _require_step_mask(is_xfer, error_type, "OP_TRANSFER")
        is_thaw = active & (pr.op == OP_THAW)
        _require_step_mask(is_thaw, error_type, "OP_THAW")
        thaw_step = _first_matching_step(is_thaw)
        if not bool((thaw_step < first_xfer).all().item()):
            raise ValueError(
                "Cannot construct 'skip_thaw' error twin: OP_THAW must precede first OP_TRANSFER"
            )
        thaw_c = pr.src[b_idx, thaw_step]
        pr.op[b_idx, thaw_step] = OP_NOOP
        st.frozen[b_idx, thaw_c] = True
        pr.src[b_idx, first_xfer] = thaw_c
        return st, _extend_steps(pr, extra=1)

    if error_type == "sealed_plate":
        _require_step_mask(is_xfer, error_type, "OP_TRANSFER")
        # Insert OP_SEAL(dst) right before the pick_tip preceding first_xfer
        insert_pos = (first_xfer - 1).clamp(min=0)
        target_dst = pr.dst[b_idx, first_xfer]
        op_seal = torch.full((b,), OP_SEAL, dtype=torch.int64, device=dev)
        pr_inserted = _insert_step_before(pr, insert_pos, new_op=op_seal, new_src=target_dst)
        return st, pr_inserted

    if error_type == "bad_temp":
        is_inc = active & (pr.op == OP_INCUBATE)
        _require_step_mask(is_inc, error_type, "OP_INCUBATE")
        inc_step = _first_matching_step(is_inc)
        too_hot = torch.randint(0, 2, (b,), generator=g).to(dev) == 0
        pr.temp[b_idx, inc_step] = torch.where(
            too_hot,
            torch.full((b,), 120, dtype=torch.int64, device=dev),
            torch.full((b,), -10, dtype=torch.int64, device=dev),
        )
        return st, _extend_steps(pr, extra=1)

    if error_type == "swapped_steps":
        _require_step_mask(is_xfer, error_type, "OP_TRANSFER")
        if not bool((first_xfer > 0).all().item()):
            raise ValueError(
                "Cannot construct 'swapped_steps' error twin: first OP_TRANSFER is at step 0 with no preceding step"
            )
        # Swap pick_tip and transfer at (first_xfer - 1, first_xfer)
        p1 = first_xfer - 1
        p2 = first_xfer
        for field in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
            arr = getattr(pr, field)
            v1 = arr[b_idx, p1].clone()
            v2 = arr[b_idx, p2].clone()
            arr[b_idx, p1] = v2
            arr[b_idx, p2] = v1
        return st, _extend_steps(pr, extra=1)

    raise AssertionError("unreachable")


@dataclass
class TwinRecallReport:
    per_error_recall: dict[str, float]
    per_error_primary_code_match: dict[str, float]  # only error types in EXPECTED_PRIMARY_CODES
    false_positive_rate_on_valid: float
    batch_size_per_type: int


def measure_twin_recall(
    batch_size: int = 512,
    max_steps: int = 24,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> TwinRecallReport:
    """Measure batched checker detection recall across all 7 seeded error types."""
    st_valid, pr_valid = make_random_valid_protocols(
        batch_size=batch_size, max_steps=max_steps, seed=seed, device=device
    )
    res_valid = check_protocols_batched(st_valid, pr_valid)
    fp_rate = float((~res_valid.ok).float().mean().item())

    recalls: dict[str, float] = {}
    code_matches: dict[str, float] = {}
    for idx, etype in enumerate(TWIN_ERROR_TYPES):
        st_tw, pr_tw = make_error_twin(st_valid, pr_valid, etype, seed=seed + idx)
        res = check_protocols_batched(st_tw, pr_tw)
        detected = ~res.ok
        recalls[etype] = float(detected.float().mean().item())
        # swapped_steps has no single expected code, so it gets no code-match entry
        # (reporting recall again under another name would double-count it)
        if etype in EXPECTED_PRIMARY_CODES:
            exp = EXPECTED_PRIMARY_CODES[etype]
            code_matches[etype] = float((res.err_code == exp).float().mean().item())

    return TwinRecallReport(
        per_error_recall=recalls,
        per_error_primary_code_match=code_matches,
        false_positive_rate_on_valid=fp_rate,
        batch_size_per_type=batch_size,
    )


@dataclass
class GenerateKeepResult:
    num_candidates: int
    num_kept: int
    kept_fraction: float
    elapsed_sec: float
    candidates_per_sec: float
    rejection_counts: dict[str, int]
    kept_state: LabStateBatch
    kept_protocols: ProtocolBatch


def sample_noisy_candidates(
    num_candidates: int,
    max_steps: int = 24,
    num_containers: int = 8,
    mutation_rate: float = 0.25,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Stochastic candidate generator (simulating uncalibrated sampler proposals)."""
    st, pr = make_random_valid_protocols(
        batch_size=num_candidates,
        max_steps=max_steps,
        num_containers=num_containers,
        seed=seed,
        device="cpu",
    )
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 4242)

    # Mutate a 1 - (1 - mutation_rate)^3 fraction of candidates; some mutations leave the
    # protocol valid, so the kept fraction is higher than the unmutated fraction
    mutate = torch.rand((num_candidates,), generator=g) < (1.0 - (1.0 - mutation_rate) ** 3)
    m_idx = mutate.nonzero(as_tuple=False).squeeze(1)
    n_m = int(m_idx.numel())
    if n_m > 0:
        kind = torch.randint(0, 6, (n_m,), generator=g)
        step = torch.randint(0, max_steps - 4, (n_m,), generator=g)

        # 0: noisy volume jitter
        k0 = m_idx[kind == 0]
        if k0.numel() > 0:
            s0 = step[kind == 0]
            pr.vol[k0, s0] = torch.randint(5, 4500, (k0.numel(),), generator=g)

        # 1: random pipette choice
        k1 = m_idx[kind == 1]
        if k1.numel() > 0:
            s1 = step[kind == 1]
            pr.pipette[k1, s1] = torch.randint(1, 5, (k1.numel(),), generator=g)

        # 2: random source well
        k2 = m_idx[kind == 2]
        if k2.numel() > 0:
            s2 = step[kind == 2]
            pr.src[k2, s2] = torch.randint(0, num_containers, (k2.numel(),), generator=g)

        # 3: noisy temperature
        k3 = m_idx[kind == 3]
        if k3.numel() > 0:
            is_inc = (~pr.pad_mask[k3]) & (pr.op[k3] == OP_INCUBATE)
            has_inc = is_inc.any(dim=1)
            k3_valid = k3[has_inc]
            if k3_valid.numel() > 0:
                inc_pos = _first_matching_step(is_inc[has_inc])
                pr.temp[k3_valid, inc_pos] = torch.randint(-5, 115, (k3_valid.numel(),), generator=g)

        # 4: drop a random step
        k4 = m_idx[kind == 4]
        if k4.numel() > 0:
            s4 = step[kind == 4]
            pr.op[k4, s4] = OP_NOOP

        # 5: swap two adjacent active steps (including pad_mask so active steps are never hidden)
        k5 = m_idx[kind == 5]
        if k5.numel() > 0:
            active_len = (~pr.pad_mask[k5]).sum(dim=1)
            max_s5 = (active_len - 2).clamp(min=0, max=max_steps - 2)
            s5 = torch.minimum(step[kind == 5], max_s5)
            for fname in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
                arr = getattr(pr, fname)
                tmp = arr[k5, s5].clone()
                arr[k5, s5] = arr[k5, s5 + 1]
                arr[k5, s5 + 1] = tmp

    dev = st.device if device is None else torch.device(device)
    return st.to(dev), pr.to(dev)


def generate_check_keep(
    num_candidates: int = 4096,
    max_steps: int = 24,
    mutation_rate: float = 0.25,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> GenerateKeepResult:
    """Generate N candidates, run batched checker, and keep only valid protocols."""
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    st, pr = sample_noisy_candidates(
        num_candidates=num_candidates,
        max_steps=max_steps,
        mutation_rate=mutation_rate,
        seed=seed,
        device=device,
    )
    res: CheckResult = check_protocols_batched(st, pr)
    keep_idx = res.ok.nonzero(as_tuple=False).squeeze(1)
    kept_st = st.index_select(keep_idx)
    kept_pr = pr.index_select(keep_idx)

    if st.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)

    codes_cpu = res.err_code.to("cpu")
    rej_counts: dict[str, int] = {}
    for code_val, name in ERR_NAMES.items():
        if code_val == ERR_OK:
            continue
        cnt = int((codes_cpu == code_val).sum().item())
        if cnt > 0:
            rej_counts[name] = cnt

    n_kept = int(keep_idx.numel())
    return GenerateKeepResult(
        num_candidates=num_candidates,
        num_kept=n_kept,
        kept_fraction=n_kept / max(num_candidates, 1),
        elapsed_sec=elapsed,
        candidates_per_sec=num_candidates / elapsed,
        rejection_counts=rej_counts,
        kept_state=kept_st,
        kept_protocols=kept_pr,
    )


@dataclass
class RepairResult:
    """Outcome of `batched_repair`.

    `num_repaired` / `repair_rate` count only intent-preserving repairs: edits that never delete
    or redirect a transfer, mix, incubate, thaw or centrifuge step ("passes checker after 1 edit").
    Protocols that only an intent-changing edit could make pass are counted in
    `num_fixed_only_by_intent_change` and are not in `repaired_mask`.
    When `reference_protocols` is supplied to `batched_repair`, `exact_recovery_rate` and
    `mean_field_edit_distance` measure exact recovery against the pre-corruption protocols.
    """

    num_broken: int
    candidates_per_protocol: int
    num_repaired: int
    repair_rate: float
    elapsed_sec: float
    protocols_repaired_per_sec: float
    repaired_protocols: ProtocolBatch
    repaired_mask: Tensor
    chosen_edit: Tensor  # (M,) int64 winning edit id, -1 if not repaired
    edit_type_counts: dict[str, int]  # winning edit kind -> number of repaired protocols
    num_fixed_only_by_intent_change: int
    num_exact_recovered: int | None = None
    exact_recovery_rate: float | None = None
    mean_field_edit_distance: float | None = None
    field_edit_distances: list[int] | None = None


# Ops whose deletion (or redirection) changes what the protocol does, not just how it does it
INTENT_OPS: tuple[int, ...] = (OP_TRANSFER, OP_MIX, OP_INCUBATE, OP_THAW, OP_CENTRIFUGE)
COMMON_ASSAY_TEMPS_C: tuple[int, ...] = (37, 72, 25, 55, 95, 4)
_STEP_COMPARE_FIELDS: tuple[str, ...] = (
    "op",
    "src",
    "dst",
    "vol",
    "temp",
    "duration",
    "pipette",
    "rpm",
)


def repair_edit_kind(edit_id: int) -> str:
    """Name of the single-edit family proposed at slot `edit_id` by `_propose_k_edits`."""
    if edit_id < 4:
        return "set_pipette"
    if edit_id < 8:
        return "clamp_volume"
    if edit_id < 14:
        return "fix_incubate"
    if edit_id == 14:
        return "delete_step"
    if edit_id < 17:
        return "swap_adjacent"
    if edit_id == 17:
        return "delete_prior_seal"
    if edit_id < 24:
        return "insert_step"
    return "redirect_dst"


def _propose_k_edits(
    init_state: LabStateBatch,
    broken_proto: ProtocolBatch,
    first_err_step: Tensor,
    num_edits: int = 26,
    state_at_err: LabStateBatch | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Expand M broken protocols into M * K single-edit repair candidates of length T + 1.

    `state_at_err` is the state just before each protocol's failing step (the checker's
    `final_state`); volume clamps use it to keep as much of the intended volume as fits.
    """
    m = broken_proto.batch_size
    c = init_state.num_containers
    dev = broken_proto.device
    k = num_edits

    ext_base = _extend_steps(broken_proto, extra=1)
    t_len = ext_base.max_steps
    st_rep = init_state.repeat_batch(k)
    cand = ext_base.repeat_batch(k)

    # Map row r in 0..m*k-1 to (item_idx, edit_id)
    item_idx = torch.arange(m, device=dev).repeat_interleave(k)
    edit_id = torch.arange(k, device=dev).repeat(m)
    err_t = first_err_step[item_idx].clamp(0, broken_proto.max_steps - 1)

    def rows_for(eid: int) -> tuple[Tensor, Tensor]:
        if eid >= k:
            empty = torch.zeros((0,), dtype=torch.int64, device=dev)
            return empty, empty
        mask = edit_id == eid
        r = mask.nonzero(as_tuple=False).squeeze(1)
        return r, err_t[r]

    # Edits 0..3: set pipette at err_t to P2, P20, P200, P1000
    for eid, pip in enumerate((PIP_P2, PIP_P20, PIP_P200, PIP_P1000)):
        r, stp = rows_for(eid)
        if r.numel() > 0:
            cand.pipette[r, stp] = pip

    # Edits 4..7: reduce volume at err_t only as far as needed (fixes overflow / not_enough).
    # vol = min(original, free dst capacity, src volume, pipette max); pipettes are tried largest
    # first so the lowest passing edit id keeps the most of the intended volume.
    if state_at_err is None:
        state_at_err = check_protocols_batched(init_state, broken_proto).final_state
    b_idx_m0 = torch.arange(m, device=dev)
    t_m0 = first_err_step.clamp(0, broken_proto.max_steps - 1)
    s_m0 = broken_proto.src[b_idx_m0, t_m0].clamp(0, c - 1)
    d_m0 = broken_proto.dst[b_idx_m0, t_m0].clamp(0, c - 1)
    op_m0 = broken_proto.op[b_idx_m0, t_m0]
    src_avail = state_at_err.volume[b_idx_m0, s_m0]
    dst_free = state_at_err.capacity[b_idx_m0, d_m0] - state_at_err.volume[b_idx_m0, d_m0]
    fit_vol = torch.minimum(broken_proto.vol[b_idx_m0, t_m0], src_avail)
    fit_vol = torch.where(op_m0 == OP_TRANSFER, torch.minimum(fit_vol, dst_free), fit_vol)
    for offset, pip in enumerate((PIP_P1000, PIP_P200, PIP_P20, PIP_P2)):
        r, stp = rows_for(4 + offset)
        if r.numel() > 0:
            cand.vol[r, stp] = fit_vol[item_idx[r]].clamp(max=PIPETTE_RANGES_TENTHS_UL[pip][1])
            cand.pipette[r, stp] = pip

    # Edits 8..13: fix bad incubation temperature / duration across canonical assay/PCR temps (4/25/37/55/72/95 C)
    for offset, tc in enumerate(COMMON_ASSAY_TEMPS_C):
        r, stp = rows_for(8 + offset)
        if r.numel() > 0:
            cand.temp[r, stp] = tc
            cand.duration[r, stp] = cand.duration[r, stp].clamp(min=30)

    # Edit 14: delete offending step (e.g. extra pick_tip, drop_tip, seal).
    # batched_repair does not count it when the deleted step is one of INTENT_OPS.
    r, stp = rows_for(14)
    if r.numel() > 0:
        cand.op[r, stp] = OP_NOOP

    # Edit 15: swap (err_t - 1, err_t)
    r, stp = rows_for(15)
    if r.numel() > 0:
        prev = (stp - 1).clamp(min=0)
        for fname in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
            arr = getattr(cand, fname)
            v1 = arr[r, prev].clone()
            v2 = arr[r, stp].clone()
            arr[r, prev] = v2
            arr[r, stp] = v1

    # Edit 16: swap (err_t, err_t + 1)
    r, stp = rows_for(16)
    if r.numel() > 0:
        nxt = (stp + 1).clamp(max=t_len - 1)
        for fname in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
            arr = getattr(cand, fname)
            v1 = arr[r, stp].clone()
            v2 = arr[r, nxt].clone()
            arr[r, stp] = v2
            arr[r, nxt] = v1

    b_idx_m = torch.arange(m, device=dev)
    t_m = first_err_step.clamp(0, broken_proto.max_steps - 1)
    src_at_err = broken_proto.src[b_idx_m, t_m].clamp(0, c - 1)
    dst_at_err = broken_proto.dst[b_idx_m, t_m].clamp(0, c - 1)
    sealed_target = torch.where(state_at_err.sealed[b_idx_m, src_at_err], src_at_err, dst_at_err)
    unsealed_target = torch.where(~state_at_err.sealed[b_idx_m, src_at_err], src_at_err, dst_at_err)

    # Edit 17: delete earliest prior OP_SEAL on the sealed offending container (tried BEFORE inserting unseal!)
    if k > 17:
        r, _ = rows_for(17)
        cols = torch.arange(broken_proto.max_steps, device=dev).unsqueeze(0)
        prior_seal = (
            (cols < t_m.unsqueeze(1))
            & (~broken_proto.pad_mask)
            & (broken_proto.op == OP_SEAL)
            & (broken_proto.src == sealed_target.unsqueeze(1))
        )
        seal_pos = _first_matching_step(prior_seal)
        has_prior = prior_seal.any(dim=1)
        cand.op[r[has_prior], seal_pos[has_prior]] = OP_NOOP

    # Edit 18: insert OP_THAW(src_at_err) at step 0 (restores pre-protocol stock thaw before pick_tip)
    if k > 18:
        r, _ = rows_for(18)
        z_pos = torch.zeros((m,), dtype=torch.int64, device=dev)
        op_vec = torch.full((m,), OP_THAW, dtype=torch.int64, device=dev)
        inserted = _insert_step_before(broken_proto, z_pos, new_op=op_vec, new_src=src_at_err)
        for fname in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
            getattr(cand, fname)[r] = getattr(inserted, fname)

    # Edits 19..23: single-step insertions immediately before err_t
    insert_specs: list[tuple[int, int, Tensor | None]] = [
        (19, OP_PICK_TIP, None),
        (20, OP_DROP_TIP, None),
        (21, OP_THAW, src_at_err),
        (22, OP_UNSEAL, sealed_target),
        (23, OP_SEAL, unsealed_target),
    ]
    for eid, ins_op, ins_src in insert_specs:
        if eid >= k:
            continue
        op_vec = torch.full((m,), ins_op, dtype=torch.int64, device=dev)
        inserted = _insert_step_before(broken_proto, t_m, new_op=op_vec, new_src=ins_src)
        r, _ = rows_for(eid)
        for fname in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
            getattr(cand, fname)[r] = getattr(inserted, fname)

    # Edits 24..k-1: redirect dst at err_t to alternate empty/reaction container
    for eid in range(24, k):
        r, stp = rows_for(eid)
        if r.numel() > 0:
            alt_dst = (3 + (eid - 24)) % max(c, 1)
            cand.dst[r, stp] = alt_dst

    return st_rep, cand


def protocol_field_edit_distance(
    steps_a: Sequence[dict[str, Any]],
    steps_b: Sequence[dict[str, Any]],
) -> int:
    """Minimum changed step-field count aligning two serialized step sequences (8 fields per step)."""
    n, m = len(steps_a), len(steps_b)
    n_fields = len(_STEP_COMPARE_FIELDS)
    if n == 0:
        return m * n_fields
    if m == 0:
        return n * n_fields

    # Fast exact path when lengths match and ops align 1-to-1
    if n == m and all(a["op"] == b["op"] for a, b in zip(steps_a, steps_b, strict=True)):
        return sum(
            1
            for a, b in zip(steps_a, steps_b, strict=True)
            for f in _STEP_COMPARE_FIELDS
            if a[f] != b[f]
        )

    prev = [j * n_fields for j in range(m + 1)]
    for i in range(1, n + 1):
        sa = steps_a[i - 1]
        cur = [i * n_fields] + [0] * m
        for j in range(1, m + 1):
            sb = steps_b[j - 1]
            sub_cost = sum(1 for f in _STEP_COMPARE_FIELDS if sa[f] != sb[f])
            cur[j] = min(
                prev[j] + n_fields,
                cur[j - 1] + n_fields,
                prev[j - 1] + sub_cost,
            )
        prev = cur
    return prev[m]


def batched_repair(
    init_state: LabStateBatch,
    broken_protocols: ProtocolBatch,
    num_edits: int = 26,
    reference_protocols: ProtocolBatch | None = None,
) -> RepairResult:
    """Try K single-edit fixes per broken protocol in one batched checker pass.

    The lowest-numbered passing edit wins, but only among intent-preserving edits: deleting a
    step in INTENT_OPS (edit 14) and redirecting a destination (edits 24+) never count as repairs.
    When `reference_protocols` is supplied, also computes `exact_recovery_rate` and
    `mean_field_edit_distance` against the pre-corruption protocols.
    """
    dev = init_state.device
    if reference_protocols is not None and reference_protocols.batch_size != broken_protocols.batch_size:
        raise ValueError(
            f"reference_protocols.batch_size={reference_protocols.batch_size} != "
            f"broken_protocols.batch_size={broken_protocols.batch_size}"
        )
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    m = broken_protocols.batch_size
    if m == 0:
        elapsed = max(time.perf_counter() - t0, 1e-9)
        return RepairResult(
            num_broken=0,
            candidates_per_protocol=num_edits,
            num_repaired=0,
            repair_rate=0.0,
            elapsed_sec=elapsed,
            protocols_repaired_per_sec=0.0,
            repaired_protocols=_extend_steps(broken_protocols, extra=1),
            repaired_mask=torch.zeros((0,), dtype=torch.bool, device=dev),
            chosen_edit=torch.zeros((0,), dtype=torch.int64, device=dev),
            edit_type_counts={},
            num_fixed_only_by_intent_change=0,
            num_exact_recovered=0 if reference_protocols is not None else None,
            exact_recovery_rate=0.0 if reference_protocols is not None else None,
            mean_field_edit_distance=0.0 if reference_protocols is not None else None,
            field_edit_distances=[] if reference_protocols is not None else None,
        )
    init_check = check_protocols_batched(init_state, broken_protocols)
    st_cand, pr_cand = _propose_k_edits(
        init_state,
        broken_protocols,
        init_check.first_err_step,
        num_edits=num_edits,
        state_at_err=init_check.final_state,
    )
    cand_check = check_protocols_batched(st_cand, pr_cand)

    ok_grid = cand_check.ok.view(m, num_edits)

    # Mask out intent-changing edits
    allowed = torch.ones((m, num_edits), dtype=torch.bool, device=dev)
    if num_edits > 14:
        t_err = init_check.first_err_step.clamp(0, broken_protocols.max_steps - 1)
        op_err = broken_protocols.op[torch.arange(m, device=dev), t_err]
        deletes_intent = torch.zeros_like(op_err, dtype=torch.bool)
        for op_id in INTENT_OPS:
            deletes_intent |= op_err == op_id
        allowed[:, 14] = ~deletes_intent
    if num_edits > 24:
        allowed[:, 24:] = False

    honest_ok = ok_grid & allowed
    repaired_mask = honest_ok.any(dim=1)
    intent_only = (ok_grid & ~allowed).any(dim=1) & ~repaired_mask
    best_edit = honest_ok.to(torch.int64).argmax(dim=1)
    chosen_flat = torch.arange(m, device=dev) * num_edits + best_edit
    best_protos = pr_cand.index_select(chosen_flat)
    chosen_edit = torch.where(repaired_mask, best_edit, torch.full_like(best_edit, -1))

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - t0, 1e-9)

    edit_type_counts: dict[str, int] = {}
    for eid in chosen_edit[repaired_mask].tolist():
        kind = repair_edit_kind(int(eid))
        edit_type_counts[kind] = edit_type_counts.get(kind, 0) + 1

    n_rep = int(repaired_mask.sum().item())
    num_exact: int | None = None
    exact_rate: float | None = None
    mean_dist: float | None = None
    dists: list[int] | None = None
    if reference_protocols is not None:
        rep_mask_list = repaired_mask.to("cpu").tolist()
        best_cpu = best_protos.to("cpu")
        ref_cpu = reference_protocols.to("cpu")
        brok_cpu = broken_protocols.to("cpu")
        dists = []
        exact_cnt = 0
        for row_i in range(m):
            ref_steps = protocol_row_to_steps(ref_cpu, row_i)
            cand_steps = (
                protocol_row_to_steps(best_cpu, row_i)
                if rep_mask_list[row_i]
                else protocol_row_to_steps(brok_cpu, row_i)
            )
            d_val = protocol_field_edit_distance(cand_steps, ref_steps)
            dists.append(d_val)
            if rep_mask_list[row_i] and d_val == 0 and cand_steps == ref_steps:
                exact_cnt += 1
        num_exact = exact_cnt
        exact_rate = exact_cnt / max(m, 1)
        mean_dist = sum(dists) / max(m, 1)

    return RepairResult(
        num_broken=m,
        candidates_per_protocol=num_edits,
        num_repaired=n_rep,
        repair_rate=n_rep / max(m, 1),
        elapsed_sec=elapsed,
        protocols_repaired_per_sec=m / elapsed,
        repaired_protocols=best_protos,
        repaired_mask=repaired_mask,
        chosen_edit=chosen_edit,
        edit_type_counts=edit_type_counts,
        num_fixed_only_by_intent_change=int(intent_only.sum().item()),
        num_exact_recovered=num_exact,
        exact_recovery_rate=exact_rate,
        mean_field_edit_distance=mean_dist,
        field_edit_distances=dists,
    )


# ---------------------------------------------------------------------------
# JSONL export/import bridge to twinstep and labbench-posttrain DPO/RLVR
# ---------------------------------------------------------------------------

_NAME_TO_OP: dict[str, int] = {name: code for code, name in OP_NAMES.items()}
_NAME_TO_PIP: dict[str, int] = {name: code for code, name in PIPETTE_NAMES.items()}


def protocol_row_to_steps_with_map(
    proto: ProtocolBatch,
    row: int = 0,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Serialize one protocol row's active steps into dicts and map raw tensor slot -> compact step index."""
    steps: list[dict[str, Any]] = []
    raw_to_compact: dict[int, int] = {}
    for t in range(proto.max_steps):
        if bool(proto.pad_mask[row, t].item()):
            continue
        op_code = int(proto.op[row, t].item())
        if op_code == OP_NOOP:
            continue
        raw_to_compact[t] = len(steps)
        steps.append(
            {
                "op": OP_NAMES.get(op_code, str(op_code)),
                "src": int(proto.src[row, t].item()),
                "dst": int(proto.dst[row, t].item()),
                "vol": int(proto.vol[row, t].item()),
                "temp": int(proto.temp[row, t].item()),
                "duration": int(proto.duration[row, t].item()),
                "pipette": PIPETTE_NAMES.get(int(proto.pipette[row, t].item()), "NONE"),
                "rpm": int(proto.rpm[row, t].item()),
            }
        )
    return steps, raw_to_compact


def protocol_row_to_steps(proto: ProtocolBatch, row: int = 0) -> list[dict[str, Any]]:
    """Serialize one protocol row's active (unpadded, non-NOOP) steps into plain dicts."""
    steps, _ = protocol_row_to_steps_with_map(proto, row=row)
    return steps


def map_raw_step_to_compact(proto: ProtocolBatch, row: int, raw_step: int) -> int | None:
    """Map a checker's raw tensor `first_err_step` onto the compacted `protocol_row_to_steps` index."""
    if raw_step < 0:
        return None
    steps, raw_to_compact = protocol_row_to_steps_with_map(proto, row=row)
    if raw_step in raw_to_compact:
        return raw_to_compact[raw_step]
    if raw_step == 0 and steps:
        return 0
    return None


def steps_dicts_to_protocol_batch(
    serialized_protocols: Sequence[Sequence[dict[str, Any]]],
    max_steps: int | None = None,
    device: str | torch.device | None = None,
) -> ProtocolBatch:
    """Deserialize step dicts produced by `protocol_row_to_steps` back into a ProtocolBatch."""
    numeric_lists: list[list[dict[str, int]]] = []
    for proto_steps in serialized_protocols:
        row_steps: list[dict[str, int]] = []
        for st in proto_steps:
            raw_op = st.get("op", "noop")
            op_code = _NAME_TO_OP[raw_op] if isinstance(raw_op, str) else int(raw_op)
            raw_pip = st.get("pipette", "NONE")
            pip_code = _NAME_TO_PIP[raw_pip] if isinstance(raw_pip, str) else int(raw_pip)
            row_steps.append(
                {
                    "op": op_code,
                    "src": int(st.get("src", 0)),
                    "dst": int(st.get("dst", 0)),
                    "vol": int(st.get("vol", 0)),
                    "temp": int(st.get("temp", 0)),
                    "duration": int(st.get("duration", 0)),
                    "pipette": pip_code,
                    "rpm": int(st.get("rpm", 0)),
                }
            )
        numeric_lists.append(row_steps)
    return ProtocolBatch.from_steps_list(numeric_lists, max_steps=max_steps, device=device)


def state_row_to_dict(state: LabStateBatch, row: int = 0) -> dict[str, Any]:
    """Serialize a single row of `LabStateBatch` into a JSON-compatible dictionary."""
    return {
        "volume": state.volume[row].to("cpu").tolist(),
        "capacity": state.capacity[row].to("cpu").tolist(),
        "sealed": state.sealed[row].to("cpu").tolist(),
        "frozen": state.frozen[row].to("cpu").tolist(),
        "reagent_mask": state.reagent_mask[row].to("cpu").tolist(),
        "has_tip": bool(state.has_tip[row].item()),
        "tip_mask": int(state.tip_mask[row].item()),
        "diluent_mask": int(state.diluent_mask),
    }


def state_batch_from_dicts(
    records: Sequence[dict[str, Any]],
    device: str | torch.device | None = None,
) -> LabStateBatch:
    """Reconstruct a `LabStateBatch` from dictionaries produced by `state_row_to_dict`."""
    if not records:
        raise ValueError("records must be non-empty to reconstruct a LabStateBatch")
    dev = default_device(device)
    dil_masks = {int(r.get("diluent_mask", 0)) for r in records}
    if len(dil_masks) > 1:
        raise ValueError(f"Mixed diluent_mask values across batch records: {sorted(dil_masks)}")
    return LabStateBatch(
        volume=torch.tensor([r["volume"] for r in records], dtype=torch.int64, device=dev),
        capacity=torch.tensor([r["capacity"] for r in records], dtype=torch.int64, device=dev),
        sealed=torch.tensor([r["sealed"] for r in records], dtype=torch.bool, device=dev),
        frozen=torch.tensor([r["frozen"] for r in records], dtype=torch.bool, device=dev),
        reagent_mask=torch.tensor([r["reagent_mask"] for r in records], dtype=torch.int64, device=dev),
        has_tip=torch.tensor([bool(r.get("has_tip", False)) for r in records], dtype=torch.bool, device=dev),
        tip_mask=torch.tensor([int(r.get("tip_mask", 0)) for r in records], dtype=torch.int64, device=dev),
        diluent_mask=next(iter(dil_masks)),
    )


def _to_twinstep_step_dicts_with_map(
    steps: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Project labfill steps onto twinstep's generic liquid-handling schema and map compact -> twinstep index."""
    projected: list[dict[str, Any]] = []
    compact_to_twinstep: dict[int, int] = {}
    for idx, st in enumerate(steps):
        op_name = st["op"]
        if op_name == "transfer" and st["vol"] > 0:
            compact_to_twinstep[idx] = len(projected)
            projected.append(
                {
                    "type": "transfer",
                    "source": f"C{st['src']}",
                    "dest": f"C{st['dst']}",
                    "volume": round(st["vol"] / 10.0, 2),
                    "unit": "uL",
                }
            )
        elif op_name == "mix" and st["vol"] > 0:
            compact_to_twinstep[idx] = len(projected)
            projected.append(
                {
                    "type": "mix",
                    "well": f"C{st['src']}",
                    "volume": round(st["vol"] / 10.0, 2),
                    "unit": "uL",
                    "cycles": 3,
                }
            )
        elif op_name == "incubate" and st["duration"] > 0:
            compact_to_twinstep[idx] = len(projected)
            projected.append(
                {
                    "type": "incubate",
                    "plate": f"C{st['src']}",
                    "temperature": float(st["temp"]),
                    "time": float(st["duration"]),
                    "time_unit": "s",
                }
            )
    return projected, compact_to_twinstep


def _to_twinstep_step_dicts(steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project labfill steps onto twinstep's generic liquid-handling step schema where applicable."""
    projected, _ = _to_twinstep_step_dicts_with_map(steps)
    return projected


def to_twinstep_records(
    init_state: LabStateBatch,
    broken: ProtocolBatch,
    repaired: ProtocolBatch | None = None,
    error_type: str = "corrupted",
    protocol_id_prefix: str = "labfill",
) -> list[dict[str, Any]]:
    """Export `(init_state, broken[, repaired])` into `twinstep` evaluation records."""
    chk_bad = check_protocols_batched(init_state, broken)
    chk_rep = check_protocols_batched(init_state, repaired) if repaired is not None else None
    b = broken.batch_size
    out: list[dict[str, Any]] = []

    for i in range(b):
        bad_steps, raw_to_compact = protocol_row_to_steps_with_map(broken, i)
        ts_steps, compact_to_ts = _to_twinstep_step_dicts_with_map(bad_steps)
        good_steps = protocol_row_to_steps(repaired, i) if repaired is not None else []
        is_ok = bool(chk_bad.ok[i].item())
        err_c = int(chk_bad.err_code[i].item())
        err_s = int(chk_bad.first_err_step[i].item())
        if is_ok:
            compact_step: int | None = None
            raw_step: int | None = None
            ts_step: int | None = None
        else:
            raw_step = err_s
            compact_step = raw_to_compact.get(err_s, 0 if (err_s == 0 and bad_steps) else None)
            ts_step = compact_to_ts.get(compact_step) if compact_step is not None else None
        label = "clean" if is_ok else error_type
        rep_ok = bool(chk_rep.ok[i].item()) if chk_rep is not None else False
        out.append(
            {
                "id": f"{protocol_id_prefix}_{i:04d}",
                "task": "error_detection",
                "protocol_id": f"{protocol_id_prefix}_{i:04d}",
                "correct_steps": good_steps,
                "corrupted_steps": bad_steps,
                "twinstep_steps": ts_steps,
                "label": label,
                "error_type": label,
                "err_code": err_c,
                "err_name": ERR_NAMES.get(err_c, "unknown"),
                "step_index": compact_step,
                "raw_step_index": raw_step,
                "twinstep_step_index": ts_step,
                "repaired_verified": rep_ok,
                "init_state": state_row_to_dict(init_state, i),
                "description": (
                    "clean protocol"
                    if is_ok
                    else f"{error_type} ({ERR_NAMES.get(err_c, 'unknown')} at step {compact_step}, raw slot {err_s})"
                ),
            }
        )
    return out


def to_preference_jsonl(
    init_state: LabStateBatch,
    broken: ProtocolBatch,
    repaired: ProtocolBatch,
    path: str | Path | None = None,
    error_type: str = "corrupted",
    only_verified: bool = True,
) -> str:
    """Serialize `(init_state, broken, repaired)` pairs as DPO/RLVR preference JSONL for post-training.

    When `only_verified=True`, emits only rows where `broken` is rejected and `repaired` passes
    `check_protocols_batched`. `first_err_step` stores the compact index into `rejected_steps`,
    while `raw_first_err_step` preserves the raw `ProtocolBatch` slot index.
    """
    if broken.batch_size != repaired.batch_size or init_state.batch_size != broken.batch_size:
        raise ValueError("init_state, broken, and repaired must have matching batch_size")

    chk_bad = check_protocols_batched(init_state, broken)
    chk_good = check_protocols_batched(init_state, repaired)
    lines: list[str] = []

    for i in range(broken.batch_size):
        bad_ok = bool(chk_bad.ok[i].item())
        good_ok = bool(chk_good.ok[i].item())
        if only_verified and (bad_ok or not good_ok):
            continue

        bad_steps, raw_to_compact = protocol_row_to_steps_with_map(broken, i)
        good_steps = protocol_row_to_steps(repaired, i)
        err_c = int(chk_bad.err_code[i].item())
        err_s = int(chk_bad.first_err_step[i].item())
        if bad_ok:
            compact_err_s = -1
            raw_err_s = -1
        else:
            raw_err_s = err_s
            compact_err_s = raw_to_compact.get(err_s, 0 if (err_s == 0 and bad_steps) else -1)
        st_dict = state_row_to_dict(init_state, i)

        record = {
            "id": f"pref_{error_type}_{i:04d}",
            "task": "protocol_repair_dpo",
            "error_type": error_type,
            "err_code": err_c,
            "err_name": ERR_NAMES.get(err_c, "unknown"),
            "first_err_step": compact_err_s,
            "raw_first_err_step": raw_err_s,
            "chosen_reward": 1.0 if good_ok else 0.0,
            "rejected_reward": 1.0 if bad_ok else 0.0,
            "prompt": (
                "Audit and minimally repair the following wet-lab protocol so every physical check passes:\n"
                + json.dumps({"init_state": st_dict, "steps": bad_steps}, separators=(",", ":"))
            ),
            "chosen": json.dumps(good_steps, separators=(",", ":")),
            "rejected": json.dumps(bad_steps, separators=(",", ":")),
            "chosen_steps": good_steps,
            "rejected_steps": bad_steps,
            "init_state": st_dict,
        }
        lines.append(json.dumps(record, sort_keys=False))

    payload = "\n".join(lines) + ("\n" if lines else "")
    if path is not None:
        Path(path).write_text(payload, encoding="utf-8")
    return payload


def from_preference_jsonl(
    jsonl_or_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch, ProtocolBatch, list[dict[str, Any]]]:
    """Load preference JSONL (from file path or raw JSONL text) back into tensor batches."""
    if isinstance(jsonl_or_path, Path):
        raw_text = jsonl_or_path.read_text(encoding="utf-8")
    else:
        s = str(jsonl_or_path)
        if "\n" not in s and not s.lstrip().startswith("{") and Path(s).exists():
            raw_text = Path(s).read_text(encoding="utf-8")
        else:
            raw_text = s

    records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    if not records:
        raise ValueError("No JSONL preference records found")

    st_batch = state_batch_from_dicts([r["init_state"] for r in records], device=device)
    broken_batch = steps_dicts_to_protocol_batch([r["rejected_steps"] for r in records], device=device)
    repaired_batch = steps_dicts_to_protocol_batch([r["chosen_steps"] for r in records], device=device)
    return st_batch, broken_batch, repaired_batch, records
