#!/usr/bin/env python3
"""Recipe 5: Batched 96-well plate liquid-handling rehearsal."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labfill.plate import (
    index_of_well,
    label_1based,
    make_serial_dilution_plate_batch,
    rehearse_96well_protocols,
    well_of_index,
)
from labfill.protocol import default_device


def main() -> None:
    dev = default_device()
    print(f"=== Recipe 5: Batched 96-Well Plate Rehearsal (device={dev}) ===\n")

    print(
        f"Well index check (LabLean convention): label(1)={label_1based(1)}, "
        f"label(65)={label_1based(65)}, label(96)={label_1based(96)}, "
        f"roundtrip('F5')={well_of_index(index_of_well('F5'))}"
    )

    st, pr = make_serial_dilution_plate_batch(batch_size=128, num_rows_active=4, seed=0, device=dev)
    res = rehearse_96well_protocols(st, pr)

    print(f"Plates rehearsed         : {pr.batch_size} (steps={pr.max_steps})")
    print(f"All valid                : {bool(res.check.ok.all().item())}")
    print(f"Volume conserved         : {bool(res.volume_conserved.all().item())}")
    print(f"Throughput               : {pr.batch_size / res.elapsed_sec:,.0f} plates/s ({res.elapsed_sec * 1000:.1f} ms)")
    concs = [round(c, 4) for c in res.solute_conc[0, 0, :8].tolist()]
    vols = res.well_volumes_ul[0, 0, :8].tolist()
    print(f"Plate 0 Row A1..A8 (uL)  : {vols}")
    print(f"Plate 0 Row A1..A8 conc  : {concs}")


if __name__ == "__main__":
    main()
