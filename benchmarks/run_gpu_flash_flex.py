#!/usr/bin/env python3
"""Ready-to-launch single-A100/H100 GPU benchmark suite for labfill (Issue #35).

Executes Recipes 1-5 on CUDA with:
  - Calibrated NVIDIA A100 / H100 / H200 / L40S / RTX 4090 BF16 Tensor Core peak TFLOPS & MFU
  - BF16 `flash_attn_varlen_func` / jagged FlashAttention SDPA + `torch.compile`
  - Prebuilt cached `BlockMask` + compiled `FlexAttention` (`W=64 + G=4`)
  - Ding et al. (2024) `O(N log N)` Segment-Tree OBFD sequence packing
  - Static-shape CUDA Graph capture/replay & CUDA device-time profiler summaries

Usage on an A100/H100 node:
    python benchmarks/run_gpu_flash_flex.py --tag a100
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.run_all import (
    assemble_payload,
    build_recipe_runners,
    format_markdown_report,
    read_cpu_info,
    resolve_part_dir,
)
from labfill.fill_gpu import _FLASH_ATTN_VARLEN_FUNC, probe_flex_attention


def main() -> None:
    ap = argparse.ArgumentParser(description="Run single-GPU FlashAttention/FlexAttention/OBFD benchmark suite.")
    ap.add_argument("--tag", default="gpu", help="report tag suffix (e.g. a100 or h100)")
    ap.add_argument("--repeats", type=int, default=5, help="number of seeds for Recipe 4 spread (default: 5)")
    ap.add_argument("--peak-tflops", type=float, default=None, help="optional manual BF16 TFLOPS override")
    ap.add_argument("--check-only", action="store_true", help="print GPU readiness diagnostics and exit")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(
            "STATUS: PENDING_GPU — torch.cuda.is_available() is False on this host.\n"
            "Launch this script on a single NVIDIA A100/H100 GPU node to populate "
            "benchmarks/results/gpu_<date>.{json,md} and close Issue #35."
        )
        return

    dev = torch.device("cuda")
    machine = read_cpu_info(dev)
    flex = probe_flex_attention(dev)
    print(f"Detected GPU: {machine['gpu_name']} (CC {machine['gpu_compute_capability']}, {machine['gpu_vram_mb']} MB)")
    print(f"flash_attn_varlen_func available: {_FLASH_ATTN_VARLEN_FUNC is not None}")
    print(f"FlexAttention compiled backward ok: {flex.get('compiled_ok')} ({flex.get('note')})")
    if args.check_only:
        return

    runners = build_recipe_runners(
        peak_tflops_override=args.peak_tflops,
        smoke=False,
        recipe4_repeats=args.repeats,
    )
    out_dir = REPO_ROOT / "benchmarks" / "results"
    part_dir = resolve_part_dir(out_dir, dev, tag=args.tag)
    part_dir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for idx, (key, (title, fn)) in enumerate(runners.items(), 1):
        print(f"  [{idx}/{len(runners)}] {title}...", flush=True)
        t0 = time.perf_counter()
        res = fn(dev)
        entry = {"machine": machine, "wall_sec": round(time.perf_counter() - t0, 2), "result": res}
        (part_dir / f"{key}.json").write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
        loaded[key] = entry

    payload = assemble_payload(loaded)
    stamp = dt.date.today().isoformat()
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = out_dir / f"gpu_{stamp}{suffix}.json"
    md_path = out_dir / f"gpu_{stamp}{suffix}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_markdown_report(payload), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
