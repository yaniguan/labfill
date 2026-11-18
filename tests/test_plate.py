from __future__ import annotations

import torch

from labfill.plate import (
    NUM_WELLS_96,
    index_of_well,
    label_1based,
    make_serial_dilution_plate_batch,
    rehearse_96well_protocols,
    well_of_index,
)


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
    assert bool(res.volume_conserved.all().item()) is True
    assert res.well_volumes_ul.shape == (16, 8, 12)

    expected_conc = torch.tensor([2.0 ** (-k) for k in range(8)], dtype=torch.float32)
    assert torch.allclose(res.solute_conc[0, 0, :8].cpu(), expected_conc, atol=1e-4)
    # Inactive rows (3..7) remain completely empty
    assert float(res.well_volumes_ul[:, 3:, :].sum().item()) == 0.0
