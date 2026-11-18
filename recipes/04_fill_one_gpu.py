#!/usr/bin/env python3
"""Recipe 4: Fill One GPU — throughput ablation on variable-length protocol sequences."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labfill.fill_gpu import run_fill_gpu_ablation
from labfill.protocol import default_device


def main() -> None:
    dev = default_device()
    print(f"=== Recipe 4: Fill One GPU Ablation (device={dev}) ===\n")

    rows, prof_ops, flex_info = run_fill_gpu_ablation(
        num_sequences=96,
        batch_size=16,
        pack_len=512,
        pack_rows_per_batch=2,
        slow_work_iters=500,
        seed=0,
        device=dev,
    )

    header = (
        f"{'Config':<26s} | {'Real tok/s':>10s} | {'Useful %':>8s} | "
        f"{'Step ms':>8s} | {'Wait %':>7s} | {'Speedup':>8s} | {'MFU':>6s} | Status / Notes"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.status == "ok":
            print(
                f"{r.config:<26s} | {r.real_tokens_per_sec:10,.1f} | "
                f"{r.useful_token_fraction:7.1%} | {r.step_time_ms:8.1f} | "
                f"{r.data_wait_fraction:6.1%} | {r.speedup_vs_baseline:7.2f}x | "
                f"{r.mfu:>6s} | {r.notes or 'ok'}"
            )
        else:
            print(
                f"{r.config:<26s} | {'—':>10s} | {'—':>8s} | "
                f"{'—':>8s} | {'—':>7s} | {'—':>8s} | {r.mfu:>6s} | {r.status}: {r.notes}"
            )

    print("\nTop torch.profiler CPU operators (1 baseline step):")
    for op in prof_ops:
        print(f"  {op['op']:<42s} calls={op['calls']:<4d} self_cpu_ms={op['self_cpu_ms']:.2f}")
    print(f"\nFlexAttention probe: {flex_info}")


if __name__ == "__main__":
    main()
