"""Batched protocol checker in PyTorch and plain-Python reference checker.

Fast pre-filter for high-throughput candidate screening, planning, and RL rewards.
LabLean (Lean 4) remains the formal source of truth before physical robot execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from labfill.protocol import (
    ERR_BAD_CONTAINER,
    ERR_BAD_DURATION,
    ERR_BAD_OP,
    ERR_BAD_PIPETTE,
    ERR_BAD_RPM,
    ERR_BAD_TEMP,
    ERR_CONTAMINATED,
    ERR_FROZEN,
    ERR_NO_TIP,
    ERR_NOT_ENOUGH,
    ERR_NOT_SEALED,
    ERR_OK,
    ERR_OVERFLOW,
    ERR_SAME_CONTAINER,
    ERR_SEALED,
    ERR_TIP_ALREADY_ON,
    ERR_UNBALANCED,
    LabStateBatch,
    MAX_INCUBATE_C,
    MAX_RPM,
    MIN_INCUBATE_C,
    OP_CENTRIFUGE,
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_MIX,
    OP_NOOP,
    OP_PICK_TIP,
    OP_SEAL,
    OP_THAW,
    OP_TRANSFER,
    OP_UNSEAL,
    PIP_P2,
    PIP_P1000,
    PIPETTE_RANGES_TENTHS_UL,
    ProtocolBatch,
    ROTOR_TOL_TENTHS_UL,
)


@dataclass
class CheckResult:
    """Result of checking B protocols.

    - ok: (B,) bool, True iff protocol had no errors
    - err_code: (B,) int64, ERR_OK (0) or first failing error code
    - first_err_step: (B,) int64, -1 if valid else 0-based step index of first error
    - final_state: LabStateBatch after executing all valid steps (frozen at failure step if invalid)
    """

    ok: Tensor
    err_code: Tensor
    first_err_step: Tensor
    final_state: LabStateBatch


def _pipette_bounds_tensors(device: torch.device) -> tuple[Tensor, Tensor]:
    # Index 0..5: 0 and 5 are invalid (min > max so range check fails cleanly)
    mins = torch.tensor([1, 2, 20, 200, 1000, 1], dtype=torch.int64, device=device)
    maxs = torch.tensor([0, 20, 200, 2000, 10000, 0], dtype=torch.int64, device=device)
    return mins, maxs


def check_protocols_batched(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
) -> CheckResult:
    """Check B protocols simultaneously by scanning over T steps with unit-stride B-wide tensor ops."""
    st = init_state.clone()
    b, c = st.volume.shape
    t_steps = protocols.max_steps
    dev = st.device

    err_code = torch.zeros((b,), dtype=torch.int64, device=dev)
    first_err_step = torch.full((b,), -1, dtype=torch.int64, device=dev)
    zero_b = torch.zeros((b,), dtype=torch.int64, device=dev)
    true_b = torch.ones((b,), dtype=torch.bool, device=dev)
    false_b = torch.zeros((b,), dtype=torch.bool, device=dev)
    pip_min, pip_max = _pipette_bounds_tensors(dev)

    # Transpose (B, T) -> (T, B) contiguous once so per-step vectors have stride 1
    op_tb = protocols.op.transpose(0, 1).contiguous()
    src_tb = protocols.src.transpose(0, 1).contiguous()
    dst_tb = protocols.dst.transpose(0, 1).contiguous()
    vol_tb = protocols.vol.transpose(0, 1).contiguous()
    temp_tb = protocols.temp.transpose(0, 1).contiguous()
    dur_tb = protocols.duration.transpose(0, 1).contiguous()
    pip_tb = protocols.pipette.transpose(0, 1).contiguous()
    rpm_tb = protocols.rpm.transpose(0, 1).contiguous()
    pad_tb = protocols.pad_mask.transpose(0, 1).contiguous()

    for t in range(t_steps):
        op_t = op_tb[t]
        padded = pad_tb[t]
        active = (err_code == ERR_OK) & (~padded) & (op_t != OP_NOOP)

        src_t = src_tb[t]
        dst_t = dst_tb[t]
        vol_t = vol_tb[t]
        temp_t = temp_tb[t]
        dur_t = dur_tb[t]
        pip_t = pip_tb[t]
        rpm_t = rpm_tb[t]

        step_err = zero_b.clone()

        # 1. Valid op check (1..9 since NOOP=0 is excluded from active)
        bad_op = (op_t < OP_NOOP) | (op_t > OP_CENTRIFUGE)
        step_err = torch.where((step_err == 0) & bad_op, ERR_BAD_OP, step_err)

        # 2. Container index bounds check (op >= OP_TRANSFER uses src; TRANSFER and CENTRIFUGE use dst)
        uses_src = op_t >= OP_TRANSFER
        uses_dst = (op_t == OP_TRANSFER) | (op_t == OP_CENTRIFUGE)
        bad_c = (uses_src & ((src_t < 0) | (src_t >= c))) | (
            uses_dst & ((dst_t < 0) | (dst_t >= c))
        )
        step_err = torch.where((step_err == 0) & bad_c, ERR_BAD_CONTAINER, step_err)

        # Safe clamped 2D indices for row-wise gather / scatter (no syncs, CUDA-graph friendly)
        src_c2 = src_t.clamp(0, c - 1).unsqueeze(1)
        dst_c2 = dst_t.clamp(0, c - 1).unsqueeze(1)

        src_vol = st.volume.gather(1, src_c2).squeeze(1)
        dst_vol = st.volume.gather(1, dst_c2).squeeze(1)
        dst_cap = st.capacity.gather(1, dst_c2).squeeze(1)
        src_sealed = st.sealed.gather(1, src_c2).squeeze(1)
        dst_sealed = st.sealed.gather(1, dst_c2).squeeze(1)
        src_frozen = st.frozen.gather(1, src_c2).squeeze(1)
        src_rmask = st.reagent_mask.gather(1, src_c2).squeeze(1)
        dst_rmask = st.reagent_mask.gather(1, dst_c2).squeeze(1)

        pip_c = pip_t.clamp(0, 5)
        valid_pip = (
            (pip_t >= PIP_P2)
            & (pip_t <= PIP_P1000)
            & (vol_t >= pip_min.gather(0, pip_c))
            & (vol_t <= pip_max.gather(0, pip_c))
        )

        is_pick = op_t == OP_PICK_TIP
        is_drop = op_t == OP_DROP_TIP
        is_xfer = op_t == OP_TRANSFER
        is_mix = op_t == OP_MIX
        is_pip_op = is_xfer | is_mix

        step_err = torch.where(
            (step_err == 0) & is_pick & st.has_tip, ERR_TIP_ALREADY_ON, step_err
        )
        step_err = torch.where(
            (step_err == 0) & (is_drop | is_pip_op) & (~st.has_tip), ERR_NO_TIP, step_err
        )
        step_err = torch.where(
            (step_err == 0) & is_pip_op & (~valid_pip), ERR_BAD_PIPETTE, step_err
        )
        step_err = torch.where(
            (step_err == 0) & is_xfer & (src_t == dst_t), ERR_SAME_CONTAINER, step_err
        )
        step_err = torch.where(
            (step_err == 0) & ((is_xfer & (src_sealed | dst_sealed)) | (is_mix & src_sealed)),
            ERR_SEALED,
            step_err,
        )
        step_err = torch.where(
            (step_err == 0) & is_pip_op & src_frozen, ERR_FROZEN, step_err
        )
        step_err = torch.where(
            (step_err == 0) & is_pip_op & ((st.tip_mask & (~src_rmask)) != 0),
            ERR_CONTAMINATED,
            step_err,
        )
        step_err = torch.where(
            (step_err == 0) & is_pip_op & (src_vol < vol_t), ERR_NOT_ENOUGH, step_err
        )
        step_err = torch.where(
            (step_err == 0) & is_xfer & (dst_vol + vol_t > dst_cap), ERR_OVERFLOW, step_err
        )

        # OP_INCUBATE
        is_inc = op_t == OP_INCUBATE
        step_err = torch.where(
            (step_err == 0) & is_inc & ((temp_t < MIN_INCUBATE_C) | (temp_t > MAX_INCUBATE_C)),
            ERR_BAD_TEMP,
            step_err,
        )
        step_err = torch.where(
            (step_err == 0) & is_inc & (dur_t <= 0), ERR_BAD_DURATION, step_err
        )

        # OP_SEAL / OP_UNSEAL
        is_seal = op_t == OP_SEAL
        is_unseal = op_t == OP_UNSEAL
        step_err = torch.where((step_err == 0) & is_seal & src_sealed, ERR_SEALED, step_err)
        step_err = torch.where(
            (step_err == 0) & is_unseal & (~src_sealed), ERR_NOT_SEALED, step_err
        )

        # OP_CENTRIFUGE
        is_spin = op_t == OP_CENTRIFUGE
        step_err = torch.where(
            (step_err == 0) & is_spin & (src_t == dst_t), ERR_SAME_CONTAINER, step_err
        )
        step_err = torch.where(
            (step_err == 0) & is_spin & ((rpm_t <= 0) | (rpm_t > MAX_RPM)),
            ERR_BAD_RPM,
            step_err,
        )
        step_err = torch.where(
            (step_err == 0) & is_spin & (dur_t <= 0), ERR_BAD_DURATION, step_err
        )
        step_err = torch.where(
            (step_err == 0) & is_spin & ((~src_sealed) | (~dst_sealed)),
            ERR_NOT_SEALED,
            step_err,
        )
        step_err = torch.where(
            (step_err == 0) & is_spin & ((src_vol - dst_vol).abs() > ROTOR_TOL_TENTHS_UL),
            ERR_UNBALANCED,
            step_err,
        )

        failed_now = active & (step_err != ERR_OK)
        err_code = torch.where(failed_now, step_err, err_code)
        first_err_step = torch.where(failed_now, t, first_err_step)

        # Sync-free state transitions on active lanes that passed all checks
        ok_now = active & (step_err == ERR_OK)
        pick_ok = ok_now & is_pick
        drop_ok = ok_now & is_drop
        xfer_ok = ok_now & is_xfer
        mix_ok = ok_now & is_mix
        thaw_ok = ok_now & (op_t == OP_THAW)
        seal_ok = ok_now & is_seal
        unseal_ok = ok_now & is_unseal

        st.has_tip = torch.where(pick_ok, true_b, torch.where(drop_ok, false_b, st.has_tip))
        st.tip_mask = torch.where(pick_ok | drop_ok, zero_b, st.tip_mask)
        st.tip_mask = torch.where(
            xfer_ok, st.tip_mask | src_rmask | dst_rmask, st.tip_mask
        )
        st.tip_mask = torch.where(mix_ok, st.tip_mask | src_rmask, st.tip_mask)

        delta_v = torch.where(xfer_ok, vol_t, zero_b).unsqueeze(1)
        st.volume.scatter_add_(1, src_c2, -delta_v)
        st.volume.scatter_add_(1, dst_c2, delta_v)

        new_dst_rm = torch.where(xfer_ok, dst_rmask | src_rmask, dst_rmask).unsqueeze(1)
        st.reagent_mask.scatter_(1, dst_c2, new_dst_rm)

        new_frozen = torch.where(thaw_ok, false_b, src_frozen).unsqueeze(1)
        st.frozen.scatter_(1, src_c2, new_frozen)

        new_sealed = torch.where(
            seal_ok, true_b, torch.where(unseal_ok, false_b, src_sealed)
        ).unsqueeze(1)
        st.sealed.scatter_(1, src_c2, new_sealed)

    return CheckResult(
        ok=(err_code == ERR_OK),
        err_code=err_code,
        first_err_step=first_err_step,
        final_state=st,
    )


def export_python_inputs(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
) -> dict[str, Any]:
    """Convert tensor batch to plain Python nested lists for pure-Python baseline timing."""
    st_cpu = init_state.to("cpu")
    pr_cpu = protocols.to("cpu")
    return {
        "volume": st_cpu.volume.tolist(),
        "capacity": st_cpu.capacity.tolist(),
        "sealed": st_cpu.sealed.tolist(),
        "frozen": st_cpu.frozen.tolist(),
        "reagent_mask": st_cpu.reagent_mask.tolist(),
        "has_tip": st_cpu.has_tip.tolist(),
        "tip_mask": st_cpu.tip_mask.tolist(),
        "op": pr_cpu.op.tolist(),
        "src": pr_cpu.src.tolist(),
        "dst": pr_cpu.dst.tolist(),
        "vol": pr_cpu.vol.tolist(),
        "temp": pr_cpu.temp.tolist(),
        "duration": pr_cpu.duration.tolist(),
        "pipette": pr_cpu.pipette.tolist(),
        "rpm": pr_cpu.rpm.tolist(),
        "pad_mask": pr_cpu.pad_mask.tolist(),
    }


def check_protocols_python_lists(
    py_data: dict[str, Any],
) -> tuple[list[bool], list[int], list[int], list[list[int]]]:
    """Plain-Python reference loop checking one protocol at a time over Python lists."""
    b = len(py_data["volume"])
    c = len(py_data["volume"][0]) if b > 0 else 0
    t_steps = len(py_data["op"][0]) if b > 0 else 0

    oks: list[bool] = []
    errs: list[int] = []
    err_steps: list[int] = []
    final_vols: list[list[int]] = []

    uses_src_ops = {
        OP_TRANSFER,
        OP_MIX,
        OP_INCUBATE,
        OP_THAW,
        OP_SEAL,
        OP_UNSEAL,
        OP_CENTRIFUGE,
    }
    uses_dst_ops = {OP_TRANSFER, OP_CENTRIFUGE}

    for i in range(b):
        vol = list(py_data["volume"][i])
        cap = py_data["capacity"][i]
        sealed = list(py_data["sealed"][i])
        frozen = list(py_data["frozen"][i])
        rmask = list(py_data["reagent_mask"][i])
        has_tip = bool(py_data["has_tip"][i])
        tip_mask = int(py_data["tip_mask"][i])

        ops_i = py_data["op"][i]
        srcs_i = py_data["src"][i]
        dsts_i = py_data["dst"][i]
        vols_i = py_data["vol"][i]
        temps_i = py_data["temp"][i]
        durs_i = py_data["duration"][i]
        pips_i = py_data["pipette"][i]
        rpms_i = py_data["rpm"][i]
        pads_i = py_data["pad_mask"][i]

        e_code = ERR_OK
        e_step = -1

        for t in range(t_steps):
            if pads_i[t]:
                continue
            op = ops_i[t]
            if op == OP_NOOP:
                continue

            if op < OP_NOOP or op > OP_CENTRIFUGE:
                e_code, e_step = ERR_BAD_OP, t
                break

            s = srcs_i[t]
            d = dsts_i[t]
            if op in uses_src_ops and (s < 0 or s >= c):
                e_code, e_step = ERR_BAD_CONTAINER, t
                break
            if op in uses_dst_ops and (d < 0 or d >= c):
                e_code, e_step = ERR_BAD_CONTAINER, t
                break

            if op == OP_PICK_TIP:
                if has_tip:
                    e_code, e_step = ERR_TIP_ALREADY_ON, t
                    break
                has_tip = True
                tip_mask = 0

            elif op == OP_DROP_TIP:
                if not has_tip:
                    e_code, e_step = ERR_NO_TIP, t
                    break
                has_tip = False
                tip_mask = 0

            elif op == OP_TRANSFER:
                v = vols_i[t]
                p = pips_i[t]
                if not has_tip:
                    e_code, e_step = ERR_NO_TIP, t
                    break
                if p not in PIPETTE_RANGES_TENTHS_UL or not (
                    PIPETTE_RANGES_TENTHS_UL[p][0] <= v <= PIPETTE_RANGES_TENTHS_UL[p][1]
                ):
                    e_code, e_step = ERR_BAD_PIPETTE, t
                    break
                if s == d:
                    e_code, e_step = ERR_SAME_CONTAINER, t
                    break
                if sealed[s] or sealed[d]:
                    e_code, e_step = ERR_SEALED, t
                    break
                if frozen[s]:
                    e_code, e_step = ERR_FROZEN, t
                    break
                if (tip_mask & (~rmask[s])) != 0:
                    e_code, e_step = ERR_CONTAMINATED, t
                    break
                if vol[s] < v:
                    e_code, e_step = ERR_NOT_ENOUGH, t
                    break
                if vol[d] + v > cap[d]:
                    e_code, e_step = ERR_OVERFLOW, t
                    break

                old_s_rm = rmask[s]
                old_d_rm = rmask[d]
                vol[s] -= v
                vol[d] += v
                rmask[d] = old_d_rm | old_s_rm
                tip_mask = tip_mask | old_s_rm | old_d_rm

            elif op == OP_MIX:
                v = vols_i[t]
                p = pips_i[t]
                if not has_tip:
                    e_code, e_step = ERR_NO_TIP, t
                    break
                if p not in PIPETTE_RANGES_TENTHS_UL or not (
                    PIPETTE_RANGES_TENTHS_UL[p][0] <= v <= PIPETTE_RANGES_TENTHS_UL[p][1]
                ):
                    e_code, e_step = ERR_BAD_PIPETTE, t
                    break
                if sealed[s]:
                    e_code, e_step = ERR_SEALED, t
                    break
                if frozen[s]:
                    e_code, e_step = ERR_FROZEN, t
                    break
                if (tip_mask & (~rmask[s])) != 0:
                    e_code, e_step = ERR_CONTAMINATED, t
                    break
                if vol[s] < v:
                    e_code, e_step = ERR_NOT_ENOUGH, t
                    break
                tip_mask = tip_mask | rmask[s]

            elif op == OP_INCUBATE:
                tc = temps_i[t]
                dur = durs_i[t]
                if tc < MIN_INCUBATE_C or tc > MAX_INCUBATE_C:
                    e_code, e_step = ERR_BAD_TEMP, t
                    break
                if dur <= 0:
                    e_code, e_step = ERR_BAD_DURATION, t
                    break

            elif op == OP_THAW:
                frozen[s] = False

            elif op == OP_SEAL:
                if sealed[s]:
                    e_code, e_step = ERR_SEALED, t
                    break
                sealed[s] = True

            elif op == OP_UNSEAL:
                if not sealed[s]:
                    e_code, e_step = ERR_NOT_SEALED, t
                    break
                sealed[s] = False

            elif op == OP_CENTRIFUGE:
                rpm = rpms_i[t]
                dur = durs_i[t]
                if s == d:
                    e_code, e_step = ERR_SAME_CONTAINER, t
                    break
                if rpm <= 0 or rpm > MAX_RPM:
                    e_code, e_step = ERR_BAD_RPM, t
                    break
                if dur <= 0:
                    e_code, e_step = ERR_BAD_DURATION, t
                    break
                if (not sealed[s]) or (not sealed[d]):
                    e_code, e_step = ERR_NOT_SEALED, t
                    break
                if abs(vol[s] - vol[d]) > ROTOR_TOL_TENTHS_UL:
                    e_code, e_step = ERR_UNBALANCED, t
                    break

        oks.append(e_code == ERR_OK)
        errs.append(e_code)
        err_steps.append(e_step)
        final_vols.append(vol)

    return oks, errs, err_steps, final_vols


def check_protocols_python(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
) -> CheckResult:
    """Plain-Python reference checker returning a CheckResult matching check_protocols_batched."""
    dev = init_state.device
    py_data = export_python_inputs(init_state, protocols)
    oks, errs, err_steps, final_vols = check_protocols_python_lists(py_data)

    final_st = init_state.clone()
    final_st.volume = torch.tensor(final_vols, dtype=torch.int64, device=dev)
    return CheckResult(
        ok=torch.tensor(oks, dtype=torch.bool, device=dev),
        err_code=torch.tensor(errs, dtype=torch.int64, device=dev),
        first_err_step=torch.tensor(err_steps, dtype=torch.int64, device=dev),
        final_state=final_st,
    )
