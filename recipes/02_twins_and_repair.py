#!/usr/bin/env python3
"""Recipe 2: Seeded-error twins, verifier-gated candidate generation, and batched repair."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labfill.protocol import default_device, make_random_valid_protocols
from labfill.twins import (
    TWIN_ERROR_TYPES,
    batched_repair,
    generate_check_keep,
    make_error_twin,
    measure_twin_recall,
)


def main() -> None:
    dev = default_device()
    print(f"=== Recipe 2: Seeded-Error Twins, Generate/Check/Keep, and Repair (device={dev}) ===\n")

    # 1. Twin recall
    report = measure_twin_recall(batch_size=512, seed=0, device=dev)
    print(f"False-positive rate on valid protocols: {report.false_positive_rate_on_valid:.2%}")
    print(f"{'Error Type':<18s} | {'Detection Recall':>16s} | {'Exact Primary Code':>18s}")
    print("-" * 60)
    for et in TWIN_ERROR_TYPES:
        code_match = report.per_error_primary_code_match.get(et)
        code_str = f"{code_match:17.1%}" if code_match is not None else f"{'n/a':>17s}"
        print(f"{et:<18s} | {report.per_error_recall[et]:15.1%} | {code_str}")

    # 2. Verifier-gated candidate generation (token-factory pattern)
    gk = generate_check_keep(num_candidates=4096, mutation_rate=0.25, seed=7, device=dev)
    print(
        f"\nGenerate/Check/Keep: kept {gk.num_kept}/{gk.num_candidates} ({gk.kept_fraction:.1%}) "
        f"at {gk.candidates_per_sec:,.0f} candidates/s"
    )
    print(f"Rejection breakdown: {gk.rejection_counts}\n")

    # 3. Batched K-edit repair search (deleting or redirecting a transfer/mix step is not a repair)
    st_base, pr_base = make_random_valid_protocols(128, seed=11, device=dev)
    print(
        f"{'Broken Twin Type':<18s} | {'Repaired (K=24)':>16s} | {'Throughput (proto/s)':>20s} | "
        f"{'Intent-change only':>18s} | Winning edits"
    )
    print("-" * 112)
    for et in TWIN_ERROR_TYPES:
        st_b, pr_b = make_error_twin(st_base, pr_base, et, seed=19)
        rr = batched_repair(st_b, pr_b, num_edits=24)
        print(
            f"{et:<18s} | {rr.num_repaired:4d}/{rr.num_broken:<4d} ({rr.repair_rate:5.1%}) | "
            f"{rr.protocols_repaired_per_sec:20,.0f} | {rr.num_fixed_only_by_intent_change:18d} | "
            f"{rr.edit_type_counts}"
        )


if __name__ == "__main__":
    main()
