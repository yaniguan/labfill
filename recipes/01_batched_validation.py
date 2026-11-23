#!/usr/bin/env python3
"""Recipe 1: Batched protocol validation vs plain-Python reference checker."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labfill.checker import (
    check_protocols_batched,
    check_protocols_python,
    check_protocols_python_lists,
    export_python_inputs,
)
from labfill.protocol import default_device, make_pcr_protocol, make_random_protocols


def main() -> None:
    dev = default_device()
    print(f"=== Recipe 1: Batched Protocol Validation (device={dev}) ===")

    # 1. PCR master-mix example (mirrors LabLean PCR.lean)
    st_pcr, pr_pcr = make_pcr_protocol(device=dev)
    res_pcr = check_protocols_batched(st_pcr, pr_pcr)
    wells_ul = [v / 10.0 for v in res_pcr.final_state.volume[0, 8:12].tolist()]
    print(f"PCR protocol valid: {bool(res_pcr.ok.item())} | Reaction wells A1..A4 (uL): {wells_ul}")

    # 2. Differential check against plain-Python reference on 2,000 random protocols
    st_mix, pr_mix = make_random_protocols(2000, max_steps=24, error_prob=0.6, seed=42, device=dev)
    rb = check_protocols_batched(st_mix, pr_mix)
    rp = check_protocols_python(st_mix, pr_mix)
    exact_match = bool(
        (rb.ok == rp.ok).all()
        and (rb.err_code == rp.err_code).all()
        and (rb.first_err_step == rp.first_err_step).all()
        and (rb.final_state.volume == rp.final_state.volume).all()
    )
    print(f"Differential check (2,000 random+corrupted): exact_match={exact_match}\n")

    # 3. Throughput benchmark for B in {1, 64, 1024, 16384}
    print(f"{'Batch B':>8s} | {'Python (proto/s)':>18s} | {'Torch CPU (proto/s)':>20s} | {'Torch/Py':>9s}")
    print("-" * 66)
    for b in (1, 64, 1024, 16384):
        st, pr = make_random_protocols(b, max_steps=24, error_prob=0.5, seed=b, device=dev)
        py_in = export_python_inputs(st, pr)
        check_protocols_batched(st, pr)  # warmup

        py_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            check_protocols_python_lists(py_in)
            py_times.append(time.perf_counter() - t0)

        th_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            check_protocols_batched(st, pr)
            th_times.append(time.perf_counter() - t0)

        py_rate = b / statistics.median(py_times)
        th_rate = b / statistics.median(th_times)
        print(f"{b:8d} | {py_rate:18,.0f} | {th_rate:20,.0f} | {th_rate / py_rate:8.2f}x")


if __name__ == "__main__":
    main()
