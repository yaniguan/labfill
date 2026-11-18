#!/usr/bin/env python3
"""Run all labfill CPU benchmarks and write JSON + Markdown tables to benchmarks/results/."""

from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from labfill.checker import (
    check_protocols_batched,
    check_protocols_python_lists,
    export_python_inputs,
)
from labfill.fill_gpu import AblationRow, run_fill_gpu_ablation
from labfill.planner import (
    batched_beam_plan_dag,
    make_assay_dag_problem,
    make_lab_schedule_durations,
    sample_random_topological_orders,
    schedule_batched_sa,
    schedule_greedy_list,
)
from labfill.plate import make_serial_dilution_plate_batch, rehearse_96well_protocols
from labfill.protocol import default_device, make_random_protocols, make_random_valid_protocols
from labfill.twins import (
    TWIN_ERROR_TYPES,
    batched_repair,
    generate_check_keep,
    make_error_twin,
    measure_twin_recall,
)


def read_cpu_info() -> dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    flags_found: list[str] = []
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        text = cpuinfo_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
        for line in text.splitlines():
            if line.lower().startswith("flags") and ":" in line:
                raw_flags = set(line.split(":", 1)[1].strip().split())
                for interesting in ("avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_tile"):
                    if interesting in raw_flags:
                        flags_found.append(interesting)
                break

    return {
        "date": dt.date.today().isoformat(),
        "cpu_model": cpu_model,
        "cpu_cores": os.cpu_count() or 1,
        "torch_num_threads": torch.get_num_threads(),
        "cpu_isa_highlights": flags_found,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "load_avg_1_5_15_at_start": list(os.getloadavg()),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)",
    }


def bench_recipe1(dev: torch.device, repeats: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b in (1, 64, 1024, 16384):
        st, pr = make_random_protocols(b, max_steps=24, error_prob=0.5, seed=100 + b, device=dev)
        py_in = export_python_inputs(st, pr)

        # Warmup
        check_protocols_batched(st, pr)
        check_protocols_python_lists(py_in)

        py_reps = 3 if b == 16384 else repeats
        py_times: list[float] = []
        for _ in range(py_reps):
            t0 = time.perf_counter()
            check_protocols_python_lists(py_in)
            py_times.append(time.perf_counter() - t0)

        th_times: list[float] = []
        for _ in range(repeats):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            check_protocols_batched(st, pr)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            th_times.append(time.perf_counter() - t0)

        py_med = statistics.median(py_times)
        th_med = statistics.median(th_times)
        py_rate = b / py_med
        th_rate = b / th_med
        rows.append(
            {
                "batch_size": b,
                "python_proto_per_sec": round(py_rate, 1),
                "python_ms": round(py_med * 1000.0, 3),
                "torch_cpu_proto_per_sec": round(th_rate, 1) if dev.type == "cpu" else "not yet measured",
                "torch_cpu_ms": round(th_med * 1000.0, 3) if dev.type == "cpu" else "not yet measured",
                "torch_vs_python_ratio": round(th_rate / py_rate, 3),
                "torch_gpu_proto_per_sec": round(th_rate, 1) if dev.type == "cuda" else "not yet measured",
            }
        )
    return rows


def bench_recipe2(dev: torch.device, repeats: int = 3) -> dict[str, Any]:
    recall_rep = measure_twin_recall(batch_size=512, seed=0, device=dev)

    gk_rates: list[float] = []
    gk_kept_frac = 0.0
    gk_rej: dict[str, int] = {}
    for r in range(repeats):
        gk = generate_check_keep(num_candidates=4096, mutation_rate=0.25, seed=10 + r, device=dev)
        gk_rates.append(gk.candidates_per_sec)
        if r == 0:
            gk_kept_frac = gk.kept_fraction
            gk_rej = gk.rejection_counts

    st_base, pr_base = make_random_valid_protocols(128, seed=22, device=dev)
    repair_rows: list[dict[str, Any]] = []
    for idx, et in enumerate(TWIN_ERROR_TYPES):
        st_b, pr_b = make_error_twin(st_base, pr_base, et, seed=30 + idx)
        rates: list[float] = []
        rep_frac = 0.0
        for _ in range(repeats):
            rr = batched_repair(st_b, pr_b, num_edits=24)
            rates.append(rr.protocols_repaired_per_sec)
            rep_frac = rr.repair_rate
        repair_rows.append(
            {
                "error_type": et,
                "detection_recall": recall_rep.per_error_recall[et],
                "primary_code_match": recall_rep.per_error_primary_code_match[et],
                "repair_rate_k24": round(rep_frac, 4),
                "repair_cpu_proto_per_sec": round(statistics.median(rates), 1),
                "repair_gpu_proto_per_sec": "not yet measured",
            }
        )

    return {
        "false_positive_rate_on_valid": recall_rep.false_positive_rate_on_valid,
        "generate_check_keep": {
            "num_candidates": 4096,
            "kept_fraction": round(gk_kept_frac, 4),
            "cpu_candidates_per_sec": round(statistics.median(gk_rates), 1),
            "gpu_candidates_per_sec": "not yet measured",
            "sample_rejection_counts": gk_rej,
        },
        "twins_and_repair": repair_rows,
    }


def bench_recipe3(dev: torch.device, repeats: int = 3) -> dict[str, Any]:
    dag = make_assay_dag_problem(device=dev)
    _, rand_topo_pass = sample_random_topological_orders(dag, num_samples=512, seed=0)

    beam_times: list[float] = []
    plan_res = batched_beam_plan_dag(dag, beam_width=32)  # warmup
    for _ in range(repeats):
        plan_res = batched_beam_plan_dag(dag, beam_width=32)
        beam_times.append(plan_res.elapsed_sec * 1000.0)

    durs = make_lab_schedule_durations(
        num_jobs=96, num_instruments=8, heterogeneous=True, seed=0, device=dev
    )

    sched_rows: list[dict[str, Any]] = []
    for name, fn in (
        ("greedy_arrival", lambda: schedule_greedy_list(durs, 8, sort_lpt=False)),
        ("greedy_lpt", lambda: schedule_greedy_list(durs, 8, sort_lpt=True)),
        ("batched_sa_r256", lambda: schedule_batched_sa(durs, 8, num_restarts=256, num_steps=300, seed=0)),
    ):
        fn()  # warmup
        ms_times: list[float] = []
        best_mk = 0.0
        for _ in range(repeats):
            sr = fn()
            ms_times.append(sr.elapsed_sec * 1000.0)
            best_mk = sr.makespan
        sched_rows.append(
            {
                "method": name,
                "makespan_sec": round(best_mk, 1),
                "cpu_time_ms": round(statistics.median(ms_times), 2),
                "gpu_time_ms": "not yet measured",
            }
        )

    return {
        "dag_beam_search": {
            "num_steps": dag.steps.max_steps,
            "beam_width": 32,
            "random_topological_valid_rate": round(rand_topo_pass, 4),
            "beam_found_valid": plan_res.found,
            "beam_cost": round(plan_res.best_cost, 3),
            "candidates_checked": plan_res.total_candidates_checked,
            "cpu_time_ms": round(statistics.median(beam_times), 2),
            "gpu_time_ms": "not yet measured",
        },
        "instrument_scheduling": sched_rows,
    }


def bench_recipe4(dev: torch.device, repeats: int = 3) -> dict[str, Any]:
    runs: list[list[AblationRow]] = []
    prof_ops: list[dict[str, Any]] = []
    flex_info: dict[str, Any] = {}

    for r in range(repeats):
        rows, p_ops, f_info = run_fill_gpu_ablation(
            num_sequences=96,
            batch_size=16,
            pack_len=512,
            pack_rows_per_batch=2,
            d_model=256,
            n_layers=4,
            n_heads=4,
            slow_work_iters=8000,
            peak_flops=None,
            seed=r,
            device=dev,
        )
        runs.append(rows)
        if r == 0:
            prof_ops = p_ops
            flex_info = f_info

    # Aggregate median across repeats per config
    num_cfgs = len(runs[0])
    agg_rows: list[dict[str, Any]] = []
    base_tps = statistics.median([runs[r][0].real_tokens_per_sec for r in range(repeats)])

    for c_i in range(num_cfgs):
        sample = runs[0][c_i]
        if sample.status != "ok":
            d = asdict(sample)
            d["gpu_tokens_per_sec"] = "not yet measured"
            agg_rows.append(d)
            continue

        med_tps = statistics.median([runs[r][c_i].real_tokens_per_sec for r in range(repeats)])
        med_utf = statistics.median([runs[r][c_i].useful_token_fraction for r in range(repeats)])
        med_step = statistics.median([runs[r][c_i].step_time_ms for r in range(repeats)])
        med_wait_ms = statistics.median([runs[r][c_i].data_wait_ms for r in range(repeats)])
        med_comp_ms = statistics.median([runs[r][c_i].compute_ms for r in range(repeats)])
        med_wait_frac = statistics.median([runs[r][c_i].data_wait_fraction for r in range(repeats)])
        med_u_tflops = statistics.median([runs[r][c_i].useful_tflops_per_sec for r in range(repeats)])
        med_e_tflops = statistics.median([runs[r][c_i].executed_tflops_per_sec for r in range(repeats)])
        med_loss = statistics.median([runs[r][c_i].final_loss for r in range(repeats)])

        agg_rows.append(
            {
                "config": sample.config,
                "status": "ok",
                "real_tokens_per_sec": round(med_tps, 1),
                "useful_token_fraction": round(med_utf, 4),
                "step_time_ms": round(med_step, 1),
                "data_wait_ms": round(med_wait_ms, 2),
                "compute_ms": round(med_comp_ms, 1),
                "data_wait_fraction": round(med_wait_frac, 4),
                "useful_tflops_per_sec": round(med_u_tflops, 4),
                "executed_tflops_per_sec": round(med_e_tflops, 4),
                "mfu": sample.mfu,
                "speedup_vs_baseline": round(med_tps / max(base_tps, 1e-6), 2),
                "final_loss": round(med_loss, 4),
                "gpu_tokens_per_sec": "not yet measured",
                "notes": sample.notes,
            }
        )

    return {
        "ablation": agg_rows,
        "profiler_top_cpu_ops": prof_ops,
        "flex_attention_probe": flex_info,
    }


def bench_recipe5(dev: torch.device, repeats: int = 3) -> dict[str, Any]:
    st, pr = make_serial_dilution_plate_batch(batch_size=128, num_rows_active=4, seed=0, device=dev)
    rehearse_96well_protocols(st, pr)  # warmup
    times: list[float] = []
    res = rehearse_96well_protocols(st, pr)
    for _ in range(repeats):
        res = rehearse_96well_protocols(st, pr)
        times.append(res.elapsed_sec)
    med_s = statistics.median(times)
    return {
        "batch_plates": 128,
        "steps_per_plate": pr.max_steps,
        "all_valid": bool(res.check.ok.all().item()),
        "volume_conserved": bool(res.volume_conserved.all().item()),
        "cpu_plates_per_sec": round(128.0 / med_s, 1),
        "cpu_time_ms": round(med_s * 1000.0, 2),
        "gpu_plates_per_sec": "not yet measured",
    }


def format_markdown_report(payload: dict[str, Any]) -> str:
    m = payload["machine"]
    lines: list[str] = []
    lines.append(f"# labfill Benchmark Report ({m['date']})\n")
    lines.append("## Machine Info\n")
    lines.append(f"- **Date**: {m['date']}")
    lines.append(f"- **CPU**: {m['cpu_model']} ({m['cpu_cores']} cores, PyTorch threads={m['torch_num_threads']})")
    lines.append(f"- **ISA highlights**: `{', '.join(m['cpu_isa_highlights']) or 'standard'}`")
    lines.append(f"- **Software**: Python {m['python_version']}, PyTorch {m['torch_version']}, NumPy {m['numpy_version']}")
    lines.append(f"- **GPU**: {m['gpu_name']} (all GPU columns below: `not yet measured`)\n")

    # Recipe 1
    lines.append("## Recipe 1: Batched Protocol Validation (T=24 steps, C=8 containers)\n")
    lines.append("| Batch B | Plain Python (proto/s) | Batched Torch CPU (proto/s) | Torch CPU / Python | Batched Torch GPU (proto/s) |")
    lines.append("|---:|---:|---:|---:|---|")
    for r in payload["recipe1_validation"]:
        lines.append(
            f"| {r['batch_size']} | {r['python_proto_per_sec']:,.1f} | "
            f"{r['torch_cpu_proto_per_sec']:,.1f} | {r['torch_vs_python_ratio']:.2f}x | "
            f"{r['torch_gpu_proto_per_sec']} |"
        )
    lines.append("")

    # Recipe 2
    r2 = payload["recipe2_twins_and_factory"]
    gk = r2["generate_check_keep"]
    lines.append("## Recipe 2: Seeded-Error Twins, Generate/Check/Keep, and Batched Repair\n")
    lines.append(
        f"- **False-positive rate on valid protocols**: `{r2['false_positive_rate_on_valid']:.1%}`"
    )
    lines.append(
        f"- **Generate-N / Check / Keep (`N={gk['num_candidates']}`)**: kept `{gk['kept_fraction']:.1%}` "
        f"at `{gk['cpu_candidates_per_sec']:,.1f}` candidates/s on CPU (GPU: `{gk['gpu_candidates_per_sec']}`)\n"
    )
    lines.append("| Error Type | Detection Recall | Repair Rate (K=24) | Repair CPU (proto/s) | Repair GPU (proto/s) |")
    lines.append("|---|---:|---:|---:|---|")
    for tr in r2["twins_and_repair"]:
        lines.append(
            f"| `{tr['error_type']}` | {tr['detection_recall']:.1%} | "
            f"{tr['repair_rate_k24']:.1%} | {tr['repair_cpu_proto_per_sec']:,.1f} | "
            f"{tr['repair_gpu_proto_per_sec']} |"
        )
    lines.append("")

    # Recipe 3
    r3 = payload["recipe3_planning_and_scheduling"]
    dag = r3["dag_beam_search"]
    lines.append("## Recipe 3: Batched Step-DAG Planning & Instrument Makespan Scheduling\n")
    lines.append(
        f"- **DAG Beam Search (`S={dag['num_steps']}`, `beam={dag['beam_width']}`)**: "
        f"random topological pass rate `{dag['random_topological_valid_rate']:.1%}` vs checker-gated beam "
        f"found=`{dag['beam_found_valid']}` (`{dag['candidates_checked']}` candidates checked in "
        f"`{dag['cpu_time_ms']:.2f}` ms CPU; GPU: `{dag['gpu_time_ms']}`)\n"
    )
    lines.append("| Scheduler (96 jobs, 8 instruments) | Makespan (s) | CPU Time (ms) | GPU Time (ms) |")
    lines.append("|---|---:|---:|---|")
    for sr in r3["instrument_scheduling"]:
        lines.append(
            f"| `{sr['method']}` | {sr['makespan_sec']:.1f} | {sr['cpu_time_ms']:.2f} | {sr['gpu_time_ms']} |"
        )
    lines.append("")

    # Recipe 4
    r4 = payload["recipe4_fill_one_gpu"]
    lines.append("## Recipe 4: Fill One GPU Ablation (4-layer d=256 Transformer, lognormal 8..512 tokens)\n")
    lines.append(
        "| Config | Real tok/s (CPU) | Useful-token % | Step Time (ms) | Data-wait % | Speedup vs (a) | MFU | GPU tok/s | Notes |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for row in r4["ablation"]:
        if row["status"] == "ok":
            lines.append(
                f"| `{row['config']}` | {row['real_tokens_per_sec']:,.1f} | "
                f"{row['useful_token_fraction'] * 100:.1f}% | {row['step_time_ms']:.1f} | "
                f"{row['data_wait_fraction'] * 100:.1f}% | {row['speedup_vs_baseline']:.2f}x | "
                f"{row['mfu']} | {row['gpu_tokens_per_sec']} | {row['notes'] or 'ok'} |"
            )
        else:
            lines.append(
                f"| `{row['config']}` | — | — | — | — | — | {row['mfu']} | "
                f"{row['gpu_tokens_per_sec']} | `{row['status']}`: {row['notes']} |"
            )
    lines.append("")

    # Recipe 5
    r5 = payload["recipe5_plate_rehearsal"]
    lines.append("## Recipe 5: Batched 96-Well Plate Rehearsal\n")
    lines.append("| Batch Plates | Steps / Plate | Volume Conserved | CPU (plates/s) | CPU Time (ms) | GPU (plates/s) |")
    lines.append("|---:|---:|---|---:|---:|---|")
    lines.append(
        f"| {r5['batch_plates']} | {r5['steps_per_plate']} | `{r5['volume_conserved']}` | "
        f"{r5['cpu_plates_per_sec']:,.1f} | {r5['cpu_time_ms']:.2f} | {r5['gpu_plates_per_sec']} |"
    )
    lines.append("")
    return "\n".join(lines)


RECIPES = {
    "recipe1_validation": ("Recipe 1: Batched protocol validation", lambda dev: bench_recipe1(dev, repeats=5)),
    "recipe2_twins_and_factory": ("Recipe 2: Seeded-error twins, generate/check/keep, batched repair", lambda dev: bench_recipe2(dev, repeats=3)),
    "recipe3_planning_and_scheduling": ("Recipe 3: Step-DAG planning and instrument scheduling", lambda dev: bench_recipe3(dev, repeats=3)),
    "recipe4_fill_one_gpu": ("Recipe 4: Fill one GPU ablation (median of 3 repeats)", lambda dev: bench_recipe4(dev, repeats=3)),
    "recipe5_plate_rehearsal": ("Recipe 5: Batched 96-well plate rehearsal", lambda dev: bench_recipe5(dev, repeats=3)),
}


def main() -> None:
    """Run each recipe and save its result right away, so a crash does not lose finished parts.

    Usage: python benchmarks/run_all.py [--only recipe4_fill_one_gpu] [--fresh]
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run one recipe key, then assemble if all parts exist")
    ap.add_argument("--fresh", action="store_true", help="ignore saved parts")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (use 1 on a shared, busy machine)")
    ap.add_argument("--tag", default="", help="suffix for the parts folder and report names, e.g. t1")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    dev = default_device()
    machine = read_cpu_info()
    out_dir = REPO_ROOT / "benchmarks" / "results"
    part_dir = out_dir / ("parts" + (f"_{args.tag}" if args.tag else ""))
    part_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running labfill benchmarks on {machine['cpu_model']} ({machine['cpu_cores']} cores)...")

    keys = [args.only] if args.only else list(RECIPES)
    for i, key in enumerate(keys, 1):
        part = part_dir / f"{key}.json"
        if part.exists() and not args.fresh:
            print(f"  [{i}/{len(keys)}] {key}: saved part found, skipping")
            continue
        title, fn = RECIPES[key]
        print(f"  [{i}/{len(keys)}] {title}...", flush=True)
        t0 = time.perf_counter()
        res = fn(dev)
        part.write_text(json.dumps({"machine": machine, "wall_sec": round(time.perf_counter() - t0, 2), "result": res}, indent=2) + "\n", encoding="utf-8")

    parts = {k: part_dir / f"{k}.json" for k in RECIPES}
    if not all(p.exists() for p in parts.values()):
        print("Not all recipe parts exist yet; run the missing ones.")
        return
    loaded = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in parts.items()}
    payload = {
        "machine": loaded["recipe1_validation"]["machine"],
        "machine_per_recipe": {k: v["machine"] for k, v in loaded.items()},
        "total_benchmark_wall_sec": round(sum(v["wall_sec"] for v in loaded.values()), 2),
    }
    payload.update({k: v["result"] for k, v in loaded.items()})

    date_tag = payload["machine"]["date"]
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = out_dir / f"cpu_{date_tag}{suffix}.json"
    md_path = out_dir / f"cpu_{date_tag}{suffix}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_report = format_markdown_report(payload)
    md_path.write_text(md_report, encoding="utf-8")
    print(f"Wrote JSON : {json_path}")
    print(f"Wrote Table: {md_path}\n")
    print(md_report)


if __name__ == "__main__":
    main()
