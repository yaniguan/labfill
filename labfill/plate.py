"""Recipe 5: Batched 96-well plate liquid-handling rehearsal.

Uses LabLean's row-major well indexing (0-based 0..95 for A1..H12; 1-based label(65) == 'F5'),
reuses `check_protocols_batched` for validation (plus a volume-conservation sanity check), and tracks exact
final (B, 8, 12) well volumes plus approximate solute concentrations across B plates at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from labfill.checker import CheckResult, check_protocols_batched
from labfill.protocol import (
    OP_DROP_TIP,
    OP_MIX,
    OP_PICK_TIP,
    OP_TRANSFER,
    LabStateBatch,
    ProtocolBatch,
    default_device,
    pick_pipette_for_vol,
    ul,
)

PLATE_ROWS = 8
PLATE_COLS = 12
NUM_WELLS_96 = PLATE_ROWS * PLATE_COLS  # 96


def index_of_well(label: str) -> int:
    """Convert 'A1'..'H12' to 0-based row-major index 0..95."""
    s = label.strip().upper()
    if len(s) < 2:
        raise ValueError(f"Invalid well label: {label!r}")
    row = ord(s[0]) - ord("A")
    col = int(s[1:]) - 1
    if not (0 <= row < PLATE_ROWS and 0 <= col < PLATE_COLS):
        raise ValueError(f"Well out of 96-well range: {label!r}")
    return row * PLATE_COLS + col


def well_of_index(idx: int) -> str:
    """Convert 0-based row-major index 0..95 to 'A1'..'H12'."""
    if not (0 <= idx < NUM_WELLS_96):
        raise ValueError(f"Well index out of 0..95 range: {idx}")
    row = idx // PLATE_COLS
    col = idx % PLATE_COLS + 1
    return f"{chr(ord('A') + row)}{col}"


def label_1based(n: int) -> str:
    """LabLean 1-based well label convention: 1='A1', 12='A12', 13='B1', 65='F5', 96='H12'."""
    if 1 <= n <= NUM_WELLS_96:
        return well_of_index(n - 1)
    return "?"


@dataclass
class PlateRehearsalResult:
    check: CheckResult
    well_volumes_tenths_ul: Tensor  # (B, 8, 12) int64 in 0.1 uL
    well_volumes_ul: Tensor         # (B, 8, 12) float32 in uL
    solute_conc: Tensor             # (B, 8, 12) float32 relative concentration (0..1)
    # (B,) bool: total container volume unchanged. Sanity check only: the checker's sole volume
    # update is a paired -v/+v scatter_add, so this is true by construction.
    volume_conserved: Tensor
    elapsed_sec: float


def rehearse_96well_protocols(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
    initial_solute_conc: Tensor | None = None,
) -> PlateRehearsalResult:
    """Simulate B liquid-handling protocols on a 96-well plate + reagent reservoirs.

    Containers 0..95 are the 96 plate wells (row-major A1..H12).
    Containers 96..C-1 are reservoirs / tubes.
    """
    b, c = init_state.volume.shape
    if c < NUM_WELLS_96:
        raise ValueError(f"Expected at least {NUM_WELLS_96} containers for a 96-well plate, got {c}")

    dev = init_state.device
    if initial_solute_conc is not None:
        if initial_solute_conc.ndim != 2 or initial_solute_conc.shape[0] != b or initial_solute_conc.shape[1] not in (
            c,
            NUM_WELLS_96,
        ):
            raise ValueError(
                f"initial_solute_conc shape {tuple(initial_solute_conc.shape)} must match "
                f"(B, C)=({b}, {c}) or (B, 96)=({b}, {NUM_WELLS_96})"
            )
        if b > 0 and bool((initial_solute_conc < 0).any().item()):
            raise ValueError("initial_solute_conc must be non-negative (>= 0)")

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    init_total = init_state.volume.sum(dim=1)
    chk = check_protocols_batched(init_state, protocols)
    final_total = chk.final_state.volume.sum(dim=1)
    conserved = init_total == final_total

    if b == 0:
        empty_i = torch.zeros((0, PLATE_ROWS, PLATE_COLS), dtype=torch.int64, device=dev)
        empty_f = torch.zeros((0, PLATE_ROWS, PLATE_COLS), dtype=torch.float32, device=dev)
        return PlateRehearsalResult(
            check=chk,
            well_volumes_tenths_ul=empty_i,
            well_volumes_ul=empty_f,
            solute_conc=empty_f.clone(),
            volume_conserved=conserved,
            elapsed_sec=max(time.perf_counter() - t0, 0.0),
        )

    # Track exact int64 volume alongside float64 solute mass to prevent float32/int64 drift
    cur_vol = init_state.volume.clone()
    if initial_solute_conc is not None:
        if initial_solute_conc.shape[1] == c:
            conc64 = initial_solute_conc.to(device=dev, dtype=torch.float64)
        else:
            conc64 = torch.zeros((b, c), dtype=torch.float64, device=dev)
            conc64[:, :NUM_WELLS_96] = initial_solute_conc.to(device=dev, dtype=torch.float64)
            if c > NUM_WELLS_96 + 1:
                conc64[:, NUM_WELLS_96 + 1] = 1.0
    else:
        conc64 = torch.zeros((b, c), dtype=torch.float64, device=dev)
        if c > NUM_WELLS_96 + 1:
            # Default convention: reservoir 97 is 1.0x stock solute, 96 is 0.0x diluent
            conc64[:, NUM_WELLS_96 + 1] = 1.0

    max_conc = conc64.max(dim=1).values.clamp(min=1.0).view(b, 1, 1)
    solute_mass = cur_vol.to(torch.float64) * conc64
    zero_i = torch.zeros((b,), dtype=torch.int64, device=dev)
    zero_f64 = torch.zeros((b,), dtype=torch.float64, device=dev)

    op_tb = protocols.op.transpose(0, 1).contiguous()
    src_tb = protocols.src.transpose(0, 1).contiguous()
    dst_tb = protocols.dst.transpose(0, 1).contiguous()
    vol_tb = protocols.vol.transpose(0, 1).contiguous()
    pad_tb = protocols.pad_mask.transpose(0, 1).contiguous()

    for t in range(protocols.max_steps):
        # Only step lanes that are completely valid or have not yet reached first_err_step
        lane_valid = chk.ok | ((chk.first_err_step >= 0) & (t < chk.first_err_step))
        xfer = lane_valid & (~pad_tb[t]) & (op_tb[t] == OP_TRANSFER)
        src_c2 = src_tb[t].clamp(0, c - 1).unsqueeze(1)
        dst_c2 = dst_tb[t].clamp(0, c - 1).unsqueeze(1)

        s_v = cur_vol.gather(1, src_c2).squeeze(1)
        s_m = solute_mass.gather(1, src_c2).squeeze(1)
        v_i = torch.where(xfer, vol_tb[t], zero_i)
        frac = torch.where(
            s_v > 0,
            (v_i.to(torch.float64) / s_v.clamp(min=1).to(torch.float64)).clamp(0.0, 1.0),
            zero_f64,
        )
        moved_mass = s_m * frac

        cur_vol.scatter_add_(1, src_c2, -v_i.unsqueeze(1))
        cur_vol.scatter_add_(1, dst_c2, v_i.unsqueeze(1))
        solute_mass.scatter_add_(1, src_c2, -moved_mass.unsqueeze(1))
        solute_mass.scatter_add_(1, dst_c2, moved_mass.unsqueeze(1))

    well_vols_tenths = chk.final_state.volume[:, :NUM_WELLS_96].view(b, PLATE_ROWS, PLATE_COLS)
    well_vols_ul = well_vols_tenths.to(torch.float32) / 10.0
    well_mass = solute_mass[:, :NUM_WELLS_96].view(b, PLATE_ROWS, PLATE_COLS)
    well_conc64 = torch.where(
        well_vols_tenths > 0,
        (well_mass / well_vols_tenths.to(torch.float64).clamp(min=1.0)).clamp(min=0.0),
        torch.zeros_like(well_mass),
    )
    well_conc = torch.minimum(well_conc64, max_conc).to(torch.float32)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return PlateRehearsalResult(
        check=chk,
        well_volumes_tenths_ul=well_vols_tenths,
        well_volumes_ul=well_vols_ul,
        solute_conc=well_conc,
        volume_conserved=conserved,
        elapsed_sec=elapsed,
    )


def make_serial_dilution_plate_batch(
    batch_size: int = 128,
    num_rows_active: int = 4,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Generate B 96-well serial dilution protocols across `num_rows_active` rows (cols 1..8).

    Containers:
      0..95 : 96-well plate (cap 300 uL, start 0 uL)
      96    : Diluent buffer trough (start 5,000 uL, cap 10,000 uL, reagent bit 0, diluent)
      97    : Stock analyte tube (start 2,000 uL, cap 5,000 uL, reagent bit 1)
      98    : Liquid waste trough (start 0 uL, cap 10,000 uL)
    """
    if batch_size < 0:
        raise ValueError(f"batch_size must be >= 0, got {batch_size}")
    if not (0 <= num_rows_active <= PLATE_ROWS):
        raise ValueError(f"num_rows_active must be in 0..{PLATE_ROWS}, got {num_rows_active}")

    dev = default_device(device)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    c_total = NUM_WELLS_96 + 3
    vol = torch.zeros((batch_size, c_total), dtype=torch.int64)
    cap = torch.full((batch_size, c_total), ul(300.0), dtype=torch.int64)
    vol[:, 96] = ul(5000.0)
    cap[:, 96] = ul(10000.0)
    vol[:, 97] = ul(2000.0)
    cap[:, 97] = ul(5000.0)
    cap[:, 98] = ul(10000.0)

    rmask = torch.zeros((batch_size, c_total), dtype=torch.int64)
    rmask[:, 96] = 1 << 0
    rmask[:, 97] = 1 << 1

    state = LabStateBatch(
        volume=vol,
        capacity=cap,
        sealed=torch.zeros((batch_size, c_total), dtype=torch.bool),
        frozen=torch.zeros((batch_size, c_total), dtype=torch.bool),
        reagent_mask=rmask,
        diluent_mask=1 << 0,
    )

    # Vary dilution transfer volume per protocol in batch (50 uL, 75 uL, or 100 uL 1:2 dilution)
    dil_choices = torch.tensor([ul(50.0), ul(75.0), ul(100.0)], dtype=torch.int64)
    pip_d_choices = torch.tensor(
        [pick_pipette_for_vol(int(v)) for v in dil_choices.tolist()], dtype=torch.int64
    )
    pip_2d_choices = torch.tensor(
        [pick_pipette_for_vol(int(2 * v)) for v in dil_choices.tolist()], dtype=torch.int64
    )
    choice_idx = torch.randint(0, 3, (batch_size,), generator=g)
    dv_per_b = dil_choices[choice_idx]          # (B,)
    pip_d_per_b = pip_d_choices[choice_idx]     # (B,)
    pip_2d_per_b = pip_2d_choices[choice_idx]   # (B,)

    # Build the shared 1-plate schedule template once and broadcast across B
    n_cols = 8
    t_ops: list[int] = []
    t_srcs: list[int] = []
    t_dsts: list[int] = []
    t_vmult: list[int] = []  # 0 = none, 1 = 1*dv (pip_d), 2 = 2*dv (pip_2d)

    def _add(op_code: int, s: int = 0, d: int = 0, mult: int = 0) -> None:
        t_ops.append(op_code)
        t_srcs.append(s)
        t_dsts.append(d)
        t_vmult.append(mult)

    # 1. Pre-fill diluent into cols 2..8 for active rows using one clean tip
    _add(OP_PICK_TIP)
    for r in range(num_rows_active):
        for col in range(1, n_cols):
            _add(OP_TRANSFER, s=96, d=r * PLATE_COLS + col, mult=1)
    _add(OP_DROP_TIP)

    # 2. Dispense 2*dv of 1.0x stock into col 1 (index 0) for active rows
    _add(OP_PICK_TIP)
    for r in range(num_rows_active):
        _add(OP_TRANSFER, s=97, d=r * PLATE_COLS + 0, mult=2)
    _add(OP_DROP_TIP)

    # 3. Serial 2-fold transfers col 0 -> 1 -> ... -> 6 -> 7, then discard dv from col 7 to waste (98)
    for r in range(num_rows_active):
        _add(OP_PICK_TIP)
        for col in range(n_cols - 1):
            w_src = r * PLATE_COLS + col
            w_dst = r * PLATE_COLS + (col + 1)
            _add(OP_TRANSFER, s=w_src, d=w_dst, mult=1)
            _add(OP_MIX, s=w_dst, d=0, mult=1)
        _add(OP_TRANSFER, s=r * PLATE_COLS + (n_cols - 1), d=98, mult=1)
        _add(OP_DROP_TIP)

    t_len = len(t_ops)
    op = torch.tensor(t_ops, dtype=torch.int64).unsqueeze(0).expand(batch_size, t_len).clone()
    src = torch.tensor(t_srcs, dtype=torch.int64).unsqueeze(0).expand(batch_size, t_len).clone()
    dst = torch.tensor(t_dsts, dtype=torch.int64).unsqueeze(0).expand(batch_size, t_len).clone()
    vmult = torch.tensor(t_vmult, dtype=torch.int64).unsqueeze(0)  # (1, T)

    vol_steps = dv_per_b.unsqueeze(1) * vmult
    zero_bt = torch.zeros((batch_size, t_len), dtype=torch.int64)
    pip_steps = torch.where(
        vmult == 1,
        pip_d_per_b.unsqueeze(1),
        torch.where(vmult == 2, pip_2d_per_b.unsqueeze(1), zero_bt),
    )
    pad_mask = torch.zeros((batch_size, t_len), dtype=torch.bool)

    proto = ProtocolBatch(
        op=op,
        src=src,
        dst=dst,
        vol=vol_steps,
        temp=zero_bt.clone(),
        duration=zero_bt.clone(),
        pipette=pip_steps,
        rpm=zero_bt.clone(),
        pad_mask=pad_mask,
    ).to(dev)
    return state.to(dev), proto
