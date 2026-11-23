"""Recipe 5: Batched 96-well plate liquid-handling rehearsal.

Uses LabLean's row-major well indexing (0-based 0..95 for A1..H12; 1-based label(65) == 'F5'),
reuses `check_protocols_batched` for validation (plus a volume-conservation sanity check), and tracks exact
final (B, 8, 12) well volumes plus approximate solute concentrations across B plates at once.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor

from labfill.checker import CheckResult, check_protocols_batched
from labfill.protocol import (
    ERR_OK,
    LabStateBatch,
    OP_DROP_TIP,
    OP_MIX,
    OP_NOOP,
    OP_PICK_TIP,
    OP_TRANSFER,
    PIP_P20,
    PIP_P200,
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
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    init_total = init_state.volume.sum(dim=1)
    chk = check_protocols_batched(init_state, protocols)
    final_total = chk.final_state.volume.sum(dim=1)
    conserved = init_total == final_total

    # Track solute mass across transfers for valid protocols to compute final concentrations
    cur_vol = init_state.volume.to(torch.float32).clone()
    if initial_solute_conc is not None:
        conc = initial_solute_conc.to(device=dev, dtype=torch.float32)
    else:
        conc = torch.zeros((b, c), dtype=torch.float32, device=dev)
        if c > NUM_WELLS_96 + 1:
            # Default convention: reservoir 97 is 1.0x stock solute, 96 is 0.0x diluent
            conc[:, NUM_WELLS_96 + 1] = 1.0

    solute_mass = cur_vol * conc
    zero_f = torch.zeros((b,), dtype=torch.float32, device=dev)

    op_tb = protocols.op.transpose(0, 1).contiguous()
    src_tb = protocols.src.transpose(0, 1).contiguous()
    dst_tb = protocols.dst.transpose(0, 1).contiguous()
    vol_tb = protocols.vol.transpose(0, 1).contiguous()
    pad_tb = protocols.pad_mask.transpose(0, 1).contiguous()

    for t in range(protocols.max_steps):
        # Only step lanes that have not failed at or before t
        lane_valid = (chk.first_err_step < 0) | (t < chk.first_err_step)
        xfer = lane_valid & (~pad_tb[t]) & (op_tb[t] == OP_TRANSFER)
        src_c2 = src_tb[t].clamp(0, c - 1).unsqueeze(1)
        dst_c2 = dst_tb[t].clamp(0, c - 1).unsqueeze(1)

        s_v = cur_vol.gather(1, src_c2).squeeze(1)
        s_m = solute_mass.gather(1, src_c2).squeeze(1)
        v_f = torch.where(xfer, vol_tb[t].to(torch.float32), zero_f)
        frac = torch.where(s_v > 0.0, (v_f / s_v.clamp(min=1e-6)).clamp(0.0, 1.0), zero_f)
        moved_mass = s_m * frac

        cur_vol.scatter_add_(1, src_c2, -v_f.unsqueeze(1))
        cur_vol.scatter_add_(1, dst_c2, v_f.unsqueeze(1))
        solute_mass.scatter_add_(1, src_c2, -moved_mass.unsqueeze(1))
        solute_mass.scatter_add_(1, dst_c2, moved_mass.unsqueeze(1))

    well_vols_tenths = chk.final_state.volume[:, :NUM_WELLS_96].view(b, PLATE_ROWS, PLATE_COLS)
    well_vols_ul = well_vols_tenths.to(torch.float32) / 10.0
    well_mass = solute_mass[:, :NUM_WELLS_96].view(b, PLATE_ROWS, PLATE_COLS)
    well_conc = torch.where(
        well_vols_tenths > 0,
        well_mass / well_vols_tenths.to(torch.float32).clamp(min=1.0),
        torch.zeros_like(well_vols_ul),
    )

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
      96    : Diluent buffer trough (start 5,000 uL, cap 10,000 uL, reagent bit 0)
      97    : Stock analyte tube (start 2,000 uL, cap 5,000 uL, reagent bit 1)
      98    : Liquid waste trough (start 0 uL, cap 10,000 uL)
    """
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
    )

    # Vary dilution transfer volume per protocol in batch (e.g. 50 uL, 75 uL, or 100 uL 1:2 dilution)
    dil_choices = torch.tensor([ul(50.0), ul(75.0), ul(100.0)], dtype=torch.int64)
    dil_v_per_b = dil_choices[torch.randint(0, 3, (batch_size,), generator=g)].tolist()

    n_cols = 8
    all_protos: list[list[dict[str, int]]] = []
    for b_i in range(batch_size):
        dv = int(dil_v_per_b[b_i])
        pip_d = pick_pipette_for_vol(dv)
        pip_2d = pick_pipette_for_vol(2 * dv)
        steps: list[dict[str, int]] = []

        # 1. Pre-fill diluent into cols 2..8 for active rows using one clean tip
        steps.append({"op": OP_PICK_TIP})
        for r in range(num_rows_active):
            for col in range(1, n_cols):
                w = r * PLATE_COLS + col
                steps.append({"op": OP_TRANSFER, "pipette": pip_d, "src": 96, "dst": w, "vol": dv})
        steps.append({"op": OP_DROP_TIP})

        # 2. Dispense 2*dv of 1.0x stock into col 1 (index 0) for active rows
        steps.append({"op": OP_PICK_TIP})
        for r in range(num_rows_active):
            w0 = r * PLATE_COLS + 0
            steps.append({"op": OP_TRANSFER, "pipette": pip_2d, "src": 97, "dst": w0, "vol": 2 * dv})
        steps.append({"op": OP_DROP_TIP})

        # 3. Serial 2-fold transfers col 0 -> 1 -> ... -> 6 -> 7, then discard dv from col 7 to waste (98)
        for r in range(num_rows_active):
            steps.append({"op": OP_PICK_TIP})
            for col in range(n_cols - 1):
                w_src = r * PLATE_COLS + col
                w_dst = r * PLATE_COLS + (col + 1)
                steps.append({"op": OP_TRANSFER, "pipette": pip_d, "src": w_src, "dst": w_dst, "vol": dv})
                steps.append({"op": OP_MIX, "pipette": pip_d, "src": w_dst, "vol": dv})
            # Equalize final well volume by discarding dv from last column into waste
            w_last = r * PLATE_COLS + (n_cols - 1)
            steps.append({"op": OP_TRANSFER, "pipette": pip_d, "src": w_last, "dst": 98, "vol": dv})
            steps.append({"op": OP_DROP_TIP})

        all_protos.append(steps)

    proto = ProtocolBatch.from_steps_list(all_protos, device=dev)
    return state.to(dev), proto
