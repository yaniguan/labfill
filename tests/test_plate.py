from __future__ import annotations

import pytest
import torch

from labfill.plate import (
    NUM_WELLS_96,
    PLATE_COLS,
    index_of_well,
    label_1based,
    make_serial_dilution_plate_batch,
    rehearse_96well_protocols,
    well_of_index,
)
from labfill.protocol import OP_DROP_TIP, OP_MIX, OP_PICK_TIP, OP_TRANSFER, ProtocolBatch, pick_pipette_for_vol, ul


def test_well_indexing_matches_lablean() -> None:
    assert label_1based(1) == "A1"
    assert label_1based(12) == "A12"
    assert label_1based(13) == "B1"
    assert label_1based(65) == "F5"
    assert label_1based(96) == "H12"
    assert label_1based(0) == "?"
    assert label_1based(97) == "?"

    for i in range(NUM_WELLS_96):
        assert index_of_well(well_of_index(i)) == i


def test_batched_96well_serial_dilution_rehearsal() -> None:
    st, pr = make_serial_dilution_plate_batch(batch_size=16, num_rows_active=3, seed=0)
    res = rehearse_96well_protocols(st, pr)
    assert bool(res.check.ok.all().item()) is True
    # Sanity check only: true by construction (transfers move volume, never create it)
    assert bool(res.volume_conserved.all().item()) is True
    assert res.well_volumes_ul.shape == (16, 8, 12)

    expected_conc = torch.tensor([2.0 ** (-k) for k in range(8)], dtype=torch.float32)
    assert torch.allclose(res.solute_conc[0, 0, :8].cpu(), expected_conc, atol=1e-4)
    # Inactive rows (3..7) remain completely empty
    assert float(res.well_volumes_ul[:, 3:, :].sum().item()) == 0.0


def test_serial_dilution_expected_final_volumes() -> None:
    rows = 3
    st, pr = make_serial_dilution_plate_batch(batch_size=16, num_rows_active=rows, seed=0)
    res = rehearse_96well_protocols(st, pr)
    assert bool(res.check.ok.all().item()) is True

    # Dilution volume dv (0.1 uL) is the first diluent transfer of each plate
    dv = pr.vol[:, 1]
    assert set(dv.tolist()) <= {ul(50.0), ul(75.0), ul(100.0)}
    assert len(set(dv.tolist())) > 1

    # Cols 2..8 get dv diluent, col 1 gets 2*dv stock, each serial step moves dv right and
    # the last column discards dv: every active well ends at exactly dv
    expected = torch.zeros((16, 8, 12), dtype=torch.int64)
    expected[:, :rows, :8] = dv.view(16, 1, 1)
    assert torch.equal(res.well_volumes_tenths_ul, expected)

    final = res.check.final_state.volume
    assert torch.equal(final[:, 96], ul(5000.0) - rows * 7 * dv)  # diluent trough
    assert torch.equal(final[:, 97], ul(2000.0) - rows * 2 * dv)  # stock tube
    assert torch.equal(final[:, 98], rows * dv)  # waste


def test_plate_rehearsal_custom_conc_validation_and_failed_lanes() -> None:
    st, pr = make_serial_dilution_plate_batch(batch_size=4, num_rows_active=2, seed=2)

    # Wrong shape raises ValueError
    with pytest.raises(ValueError, match="initial_solute_conc shape"):
        rehearse_96well_protocols(st, pr, initial_solute_conc=torch.zeros((4, 10)))
    with pytest.raises(ValueError, match="non-negative"):
        bad_c = torch.zeros((4, st.num_containers))
        bad_c[0, 0] = -0.1
        rehearse_96well_protocols(st, pr, initial_solute_conc=bad_c)

    # (B, 96) plate-only concentration tensor is accepted and padded with reservoir defaults
    plate_96 = torch.zeros((4, NUM_WELLS_96), dtype=torch.float32)
    res96 = rehearse_96well_protocols(st, pr, initial_solute_conc=plate_96)
    assert bool(((res96.solute_conc >= 0.0) & (res96.solute_conc <= 1.0)).all().item())

    # Inject an error on lane 1 at step 0 (drop_tip with no tip) -> lane 1 wells stay at 0 conc & 0 vol
    pr_bad = pr.clone()
    pr_bad.op[1, 0] = OP_DROP_TIP
    res_bad = rehearse_96well_protocols(st, pr_bad)
    assert bool(res_bad.check.ok[0].item()) is True
    assert bool(res_bad.check.ok[1].item()) is False
    assert float(res_bad.well_volumes_ul[1].sum().item()) == 0.0
    assert float(res_bad.solute_conc[1].sum().item()) == 0.0


def test_vectorized_serial_dilution_matches_step_dict_reference() -> None:
    b, rows, seed = 12, 3, 0
    _, pr_vec = make_serial_dilution_plate_batch(batch_size=b, num_rows_active=rows, seed=seed)

    g = torch.Generator(device="cpu").manual_seed(seed)
    dil_choices = torch.tensor([ul(50.0), ul(75.0), ul(100.0)], dtype=torch.int64)
    dil_v = dil_choices[torch.randint(0, 3, (b,), generator=g)].tolist()

    all_protos: list[list[dict[str, int]]] = []
    for dv in dil_v:
        pip_d = pick_pipette_for_vol(int(dv))
        pip_2d = pick_pipette_for_vol(int(2 * dv))
        steps: list[dict[str, int]] = [{"op": OP_PICK_TIP}]
        for r in range(rows):
            for col in range(1, 8):
                steps.append({"op": OP_TRANSFER, "pipette": pip_d, "src": 96, "dst": r * PLATE_COLS + col, "vol": dv})
        steps.append({"op": OP_DROP_TIP})
        steps.append({"op": OP_PICK_TIP})
        for r in range(rows):
            steps.append({"op": OP_TRANSFER, "pipette": pip_2d, "src": 97, "dst": r * PLATE_COLS, "vol": 2 * dv})
        steps.append({"op": OP_DROP_TIP})
        for r in range(rows):
            steps.append({"op": OP_PICK_TIP})
            for col in range(7):
                w_s, w_d = r * PLATE_COLS + col, r * PLATE_COLS + col + 1
                steps.append({"op": OP_TRANSFER, "pipette": pip_d, "src": w_s, "dst": w_d, "vol": dv})
                steps.append({"op": OP_MIX, "pipette": pip_d, "src": w_d, "vol": dv})
            steps.append({"op": OP_TRANSFER, "pipette": pip_d, "src": r * PLATE_COLS + 7, "dst": 98, "vol": dv})
            steps.append({"op": OP_DROP_TIP})
        all_protos.append(steps)

    pr_ref = ProtocolBatch.from_steps_list(all_protos, device=pr_vec.device)
    for fname in ("op", "src", "dst", "vol", "temp", "duration", "pipette", "rpm", "pad_mask"):
        assert torch.equal(getattr(pr_vec, fname), getattr(pr_ref, fname)), fname
