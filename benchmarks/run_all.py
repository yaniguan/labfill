#!/usr/bin/env python3
"""Run all labfill CPU/GPU benchmarks and write JSON + Markdown tables to benchmarks/results/."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
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
from labfill.fill_gpu import (
    _FLASH_ATTN_VARLEN_FUNC,
    AblationRow,
    detect_gpu_peak_flops,
    run_fill_gpu_ablation,
)
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


def _query_nvidia_smi() -> dict[str, str] | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=driver_version,clocks.current.sm,clocks.max.sm,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        if len(parts) >= 5:
            return {
                "driver_version": parts[0],
                "sm_clock_mhz": parts[1],
                "sm_max_clock_mhz": parts[2],
                "power_draw_w": parts[3],
                "power_limit_w": parts[4],
            }
    except Exception:
        return None
    return None


def read_cpu_info(dev: torch.device | None = None) -> dict[str, Any]:
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

    load_avg = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    cuda_avail = torch.cuda.is_available()
    eff_type = dev.type if dev is not None else ("cuda" if cuda_avail else "cpu")
    use_cuda_meta = eff_type == "cuda" and cuda_avail

    gpu_name = torch.cuda.get_device_name(0) if use_cuda_meta else "none (CPU only)"
    gpu_cc: str | None = None
    gpu_vram_mb: int | None = None
    if use_cuda_meta:
        props = torch.cuda.get_device_properties(0)
        gpu_cc = f"{props.major}.{props.minor}"
        gpu_vram_mb = int(props.total_memory // (1024 * 1024))

    return {
        "date": dt.date.today().isoformat(),
        "device_type": eff_type,
        "cpu_model": cpu_model,
        "cpu_cores": os.cpu_count() or 1,
        "torch_num_threads": torch.get_num_threads(),
        "cpu_isa_highlights": flags_found,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flash_attn_available": _FLASH_ATTN_VARLEN_FUNC is not None,
        "numpy_version": np.__version__,
        "load_avg_1_5_15_at_start": load_avg,
        "cuda_available": use_cuda_meta,
        "gpu_name": gpu_name,
        "gpu_compute_capability": gpu_cc,
        "gpu_vram_mb": gpu_vram_mb,
        "nvidia_smi_telemetry": _query_nvidia_smi() if use_cuda_meta else None,
    }


def bench_recipe1(
    dev: torch.device,
    repeats: int = 5,
    batch_sizes: tuple[int, ...] = (1, 64, 1024, 16384),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for b in batch_sizes:
        st, pr = make_random_protocols(b, max_steps=24, error_prob=0.5, seed=100 + b, device=dev)
        py_in = export_python_inputs(st, pr)

        # Warmup
        check_protocols_batched(st, pr)
        check_protocols_python_lists(py_in)

        py_reps = min(3, repeats) if b == 16384 else repeats
        py_times: list[float] = []
        for _ in range(py_reps):
            t0 = time.perf_counter()
            check_protocols_python_lists(py_in)
            py_times.append(time.perf_counter() - t0)

        th_times: list[float] = []
        for _ in range(repeats):
            if dev.type == "cuda":
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_ev.record()
                check_protocols_batched(st, pr)
                end_ev.record()
                torch.cuda.synchronize()
                th_times.append(max(float(start_ev.elapsed_time(end_ev)) / 1000.0, 1e-9))
            else:
                t0 = time.perf_counter()
                check_protocols_batched(st, pr)
                th_times.append(max(time.perf_counter() - t0, 1e-9))

        py_med = max(statistics.median(py_times), 1e-9)
        th_med = max(statistics.median(th_times), 1e-9)
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
                "torch_gpu_ms": round(th_med * 1000.0, 3) if dev.type == "cuda" else "not yet measured",
            }
        )
    return rows


def bench_recipe2(
    dev: torch.device,
    repeats: int = 3,
    num_candidates: int = 4096,
    base_batch: int = 128,
) -> dict[str, Any]:
    recall_rep = measure_twin_recall(batch_size=min(512, max(base_batch, 32)), seed=0, device=dev)

    gk_rates: list[float] = []
    gk_kept_frac = 0.0
    gk_rej: dict[str, int] = {}
    for r in range(repeats):
        gk = generate_check_keep(num_candidates=num_candidates, mutation_rate=0.25, seed=10 + r, device=dev)
        gk_rates.append(gk.candidates_per_sec)
        if r == 0:
            gk_kept_frac = gk.kept_fraction
            gk_rej = gk.rejection_counts

    st_base, pr_base = make_random_valid_protocols(base_batch, seed=22, device=dev)
    repair_rows: list[dict[str, Any]] = []
    for idx, et in enumerate(TWIN_ERROR_TYPES):
        st_b, pr_b = make_error_twin(st_base, pr_base, et, seed=30 + idx)
        rates: list[float] = []
        rep_frac = 0.0
        for _ in range(repeats):
            rr = batched_repair(st_b, pr_b, num_edits=24)
            rates.append(rr.protocols_repaired_per_sec)
            rep_frac = rr.repair_rate
        med_rate = round(statistics.median(rates), 1)
        repair_rows.append(
            {
                "error_type": et,
                "detection_recall": recall_rep.per_error_recall[et],
                "primary_code_match": recall_rep.per_error_primary_code_match.get(et, "n/a"),
                "repair_rate_k24": round(rep_frac, 4),
                "repair_edit_type_counts": rr.edit_type_counts,
                "fixed_only_by_intent_change": rr.num_fixed_only_by_intent_change,
                "repair_cpu_proto_per_sec": med_rate if dev.type == "cpu" else "not yet measured",
                "repair_gpu_proto_per_sec": med_rate if dev.type == "cuda" else "not yet measured",
            }
        )

    gk_med = round(statistics.median(gk_rates), 1)
    return {
        "false_positive_rate_on_valid": recall_rep.false_positive_rate_on_valid,
        "generate_check_keep": {
            "num_candidates": num_candidates,
            "kept_fraction": round(gk_kept_frac, 4),
            "cpu_candidates_per_sec": gk_med if dev.type == "cpu" else "not yet measured",
            "gpu_candidates_per_sec": gk_med if dev.type == "cuda" else "not yet measured",
            "sample_rejection_counts": gk_rej,
        },
        "twins_and_repair": repair_rows,
    }


def bench_recipe3(dev: torch.device, repeats: int = 3, sa_steps: int = 300) -> dict[str, Any]:
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
        ("batched_sa_r256", lambda: schedule_batched_sa(durs, 8, num_restarts=256, num_steps=sa_steps, seed=0)),
    ):
        fn()  # warmup
        ms_times: list[float] = []
        best_mk = 0.0
        for _ in range(repeats):
            sr = fn()
            ms_times.append(sr.elapsed_sec * 1000.0)
            best_mk = sr.makespan
        med_ms = round(statistics.median(ms_times), 2)
        sched_rows.append(
            {
                "method": name,
                "makespan_sec": round(best_mk, 1),
                "cpu_time_ms": med_ms if dev.type == "cpu" else "not yet measured",
                "gpu_time_ms": med_ms if dev.type == "cuda" else "not yet measured",
            }
        )

    beam_med = round(statistics.median(beam_times), 2)
    return {
        "dag_beam_search": {
            "num_steps": dag.steps.max_steps,
            "beam_width": 32,
            "random_topological_valid_rate": round(rand_topo_pass, 4),
            "beam_found_valid": plan_res.found,
            "beam_cost": round(plan_res.best_cost, 3),
            "candidates_checked": plan_res.total_candidates_checked,
            "cpu_time_ms": beam_med if dev.type == "cpu" else "not yet measured",
            "gpu_time_ms": beam_med if dev.type == "cuda" else "not yet measured",
        },
        "instrument_scheduling": sched_rows,
    }


def resolve_benchmark_peak_flops(
    dev: torch.device,
    peak_tflops_override: float | None = None,
    gpu_name: str | None = None,
) -> tuple[float | None, str]:
    """Resolve calibrated hardware peak FLOP/s and diagnostic label for Recipe 4 MFU."""
    if peak_tflops_override is not None:
        return detect_gpu_peak_flops(gpu_name=gpu_name, override_tflops=peak_tflops_override)
    if dev.type == "cuda":
        inferred_name = gpu_name
        if inferred_name is None and torch.cuda.is_available():
            inferred_name = torch.cuda.get_device_name(0)
        return detect_gpu_peak_flops(inferred_name, precision="bf16")
    return None, "N/A"


def bench_recipe4(
    dev: torch.device,
    repeats: int = 3,
    peak_flops: float | None = None,
    peak_tflops_override: float | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    if peak_flops is not None:
        resolved_peak = peak_flops
        peak_desc = f"explicit ({peak_flops / 1e12:.2f} TFLOPS)"
    else:
        resolved_peak, peak_desc = resolve_benchmark_peak_flops(
            dev, peak_tflops_override=peak_tflops_override
        )

    mfu_na_label = peak_desc if (resolved_peak is None and dev.type == "cuda") else "N/A"

    if smoke:
        num_sequences = 24
        batch_size = 8
        pack_rows_per_batch = 1
        d_model = 64
        n_layers = 2
        slow_work_iters = 20
        include_varlen_bf16 = True
    elif dev.type == "cuda":
        # GPU-scale batch sizing to saturate A100/H100 SMs and warm Inductor/Triton kernels
        num_sequences = 512
        batch_size = 32
        pack_rows_per_batch = 8
        d_model = 256
        n_layers = 4
        slow_work_iters = 8000
        include_varlen_bf16 = True
    else:
        num_sequences = 96
        batch_size = 16
        pack_rows_per_batch = 2
        d_model = 256
        n_layers = 4
        slow_work_iters = 8000
        include_varlen_bf16 = False

    runs: list[list[AblationRow]] = []
    prof_ops: list[dict[str, Any]] = []
    flex_info: dict[str, Any] = {}

    for r in range(repeats):
        rows, p_ops, f_info = run_fill_gpu_ablation(
            num_sequences=num_sequences,
            batch_size=batch_size,
            pack_len=512,
            pack_rows_per_batch=pack_rows_per_batch,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            slow_work_iters=slow_work_iters,
            peak_flops=resolved_peak,
            seed=r,
            device=dev,
            include_varlen_bf16=include_varlen_bf16,
            mfu_na_label=mfu_na_label,
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
            d["real_tokens_per_sec"] = "not yet measured" if dev.type == "cuda" else 0.0
            d["gpu_tokens_per_sec"] = "—" if dev.type == "cuda" else "not yet measured"
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

        mfu_val = (
            f"{100.0 * (med_u_tflops * 1e12) / resolved_peak:.2f}%"
            if resolved_peak is not None
            else sample.mfu
        )

        agg_rows.append(
            {
                "config": sample.config,
                "status": "ok",
                "real_tokens_per_sec": round(med_tps, 1) if dev.type == "cpu" else "not yet measured",
                "useful_token_fraction": round(med_utf, 4),
                "step_time_ms": round(med_step, 1),
                "data_wait_ms": round(med_wait_ms, 2),
                "compute_ms": round(med_comp_ms, 1),
                "data_wait_fraction": round(med_wait_frac, 4),
                "useful_tflops_per_sec": round(med_u_tflops, 4),
                "executed_tflops_per_sec": round(med_e_tflops, 4),
                "mfu": mfu_val,
                "speedup_vs_baseline": round(med_tps / max(base_tps, 1e-6), 2),
                "final_loss": round(med_loss, 4),
                "gpu_tokens_per_sec": round(med_tps, 1) if dev.type == "cuda" else "not yet measured",
                "notes": sample.notes,
            }
        )

    return {
        "ablation": agg_rows,
        "peak_flops_per_sec": resolved_peak,
        "peak_flops_source": peak_desc,
        "profiler_top_ops": prof_ops,
        "profiler_top_cpu_ops": prof_ops,
        "flex_attention_probe": flex_info,
    }


def bench_recipe5(dev: torch.device, repeats: int = 3, batch_plates: int = 128) -> dict[str, Any]:
    st, pr = make_serial_dilution_plate_batch(batch_size=batch_plates, num_rows_active=4, seed=0, device=dev)
    rehearse_96well_protocols(st, pr)  # warmup
    times: list[float] = []
    res = rehearse_96well_protocols(st, pr)
    for _ in range(repeats):
        if dev.type == "cuda":
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_ev.record()
            res = rehearse_96well_protocols(st, pr)
            end_ev.record()
            torch.cuda.synchronize()
            times.append(max(float(start_ev.elapsed_time(end_ev)) / 1000.0, 1e-9))
        else:
            res = rehearse_96well_protocols(st, pr)
            times.append(max(res.elapsed_sec, 1e-9))
    med_s = statistics.median(times)
    rate = round(float(batch_plates) / med_s, 1)
    ms = round(med_s * 1000.0, 2)
    return {
        "batch_plates": batch_plates,
        "steps_per_plate": pr.max_steps,
        "all_valid": bool(res.check.ok.all().item()),
        "volume_conserved": bool(res.volume_conserved.all().item()),
        "cpu_plates_per_sec": rate if dev.type == "cpu" else "not yet measured",
        "cpu_time_ms": ms if dev.type == "cpu" else "not yet measured",
        "gpu_plates_per_sec": rate if dev.type == "cuda" else "not yet measured",
        "gpu_time_ms": ms if dev.type == "cuda" else "not yet measured",
    }


def _fmt_num(val: Any, fmt: str = ",.1f", suffix: str = "") -> str:
    """Format numeric benchmark cells while preserving string placeholders such as 'not yet measured'."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return f"{val:{fmt}}{suffix}"
    return str(val)


def format_markdown_report(payload: dict[str, Any]) -> str:
    m = payload["machine"]
    gpu_measured = any(
        isinstance(r.get("torch_gpu_proto_per_sec"), (int, float))
        and not isinstance(r.get("torch_gpu_proto_per_sec"), bool)
        for r in payload.get("recipe1_validation", [])
    )
    lines: list[str] = []
    lines.append(f"# labfill Benchmark Report ({m['date']})\n")
    lines.append("## Machine Info\n")
    lines.append(f"- **Date**: {m['date']}")
    lines.append(f"- **CPU**: {m['cpu_model']} ({m['cpu_cores']} cores, PyTorch threads={m['torch_num_threads']})")
    lines.append(f"- **ISA highlights**: `{', '.join(m['cpu_isa_highlights']) or 'standard'}`")
    lines.append(f"- **Software**: Python {m['python_version']}, PyTorch {m['torch_version']}, NumPy {m['numpy_version']}")
    if gpu_measured:
        gpu_extra = ""
        if m.get("cuda_version"):
            gpu_extra += f", CUDA {m['cuda_version']}"
        if m.get("gpu_vram_mb"):
            gpu_extra += f", {m['gpu_vram_mb']} MB VRAM"
        lines.append(f"- **GPU**: {m['gpu_name']}{gpu_extra}\n")
    else:
        lines.append(f"- **GPU**: {m['gpu_name']} (all GPU columns below: `not yet measured`)\n")

    per_recipe = payload.get("machine_per_recipe", {})
    if per_recipe:
        lines.append("| Recipe | PyTorch threads | Load avg 1/5/15 min at start |")
        lines.append("|---|---:|---|")
        for key, pm in per_recipe.items():
            raw_load = pm.get("load_avg_1_5_15_at_start")
            load = " / ".join(f"{x:.1f}" for x in raw_load) if raw_load else "N/A"
            lines.append(f"| `{key}` | {pm['torch_num_threads']} | {load} |")
        lines.append("")

    # Recipe 1
    lines.append("## Recipe 1: Batched Protocol Validation (T=24 steps, C=8 containers)\n")
    lines.append("| Batch B | Plain Python (proto/s) | Batched Torch CPU (proto/s) | Torch / Python | Batched Torch GPU (proto/s) |")
    lines.append("|---:|---:|---:|---:|---|")
    for r in payload["recipe1_validation"]:
        lines.append(
            f"| {r['batch_size']} | {_fmt_num(r['python_proto_per_sec'], ',.1f')} | "
            f"{_fmt_num(r['torch_cpu_proto_per_sec'], ',.1f')} | "
            f"{_fmt_num(r['torch_vs_python_ratio'], '.2f', 'x')} | "
            f"{_fmt_num(r['torch_gpu_proto_per_sec'], ',.1f')} |"
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
        f"at `{_fmt_num(gk['cpu_candidates_per_sec'], ',.1f')}` candidates/s on CPU "
        f"(GPU: `{_fmt_num(gk['gpu_candidates_per_sec'], ',.1f')}`)\n"
    )
    lines.append(
        "Repair Rate counts only edits that keep every transfer and mix step; deleting or "
        "redirecting one is reported separately and not counted.\n"
    )
    lines.append("| Error Type | Detection Recall | Repair Rate (K=24) | Winning Edits | Intent-Change Only | Repair CPU (proto/s) | Repair GPU (proto/s) |")
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for tr in r2["twins_and_repair"]:
        edits = ", ".join(f"{k}: {v}" for k, v in tr.get("repair_edit_type_counts", {}).items()) or "—"
        lines.append(
            f"| `{tr['error_type']}` | {tr['detection_recall']:.1%} | "
            f"{tr['repair_rate_k24']:.1%} | {edits} | {tr.get('fixed_only_by_intent_change', 'n/a')} | "
            f"{_fmt_num(tr['repair_cpu_proto_per_sec'], ',.1f')} | "
            f"{_fmt_num(tr['repair_gpu_proto_per_sec'], ',.1f')} |"
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
        f"`{_fmt_num(dag['cpu_time_ms'], '.2f')}` ms CPU; GPU: `{_fmt_num(dag['gpu_time_ms'], '.2f')}`)\n"
    )
    lines.append("| Scheduler (96 jobs, 8 instruments) | Makespan (s) | CPU Time (ms) | GPU Time (ms) |")
    lines.append("|---|---:|---:|---|")
    for sr in r3["instrument_scheduling"]:
        lines.append(
            f"| `{sr['method']}` | {_fmt_num(sr['makespan_sec'], '.1f')} | "
            f"{_fmt_num(sr['cpu_time_ms'], '.2f')} | {_fmt_num(sr['gpu_time_ms'], '.2f')} |"
        )
    lines.append("")

    # Recipe 4
    r4 = payload["recipe4_fill_one_gpu"]
    lines.append("## Recipe 4: Fill One GPU Ablation (4-layer d=256 Transformer, lognormal 8..512 tokens)\n")
    if r4.get("peak_flops_source") and r4.get("peak_flops_source") != "N/A":
        lines.append(f"- **MFU Peak Reference**: `{r4['peak_flops_source']}`\n")
    lines.append(
        "| Config | Real tok/s (CPU) | Useful-token % | Step Time (ms) | Data-wait % | Speedup vs (a) | MFU | GPU tok/s | Notes |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for row in r4["ablation"]:
        if row["status"] == "ok":
            lines.append(
                f"| `{row['config']}` | {_fmt_num(row['real_tokens_per_sec'], ',.1f')} | "
                f"{row['useful_token_fraction'] * 100:.1f}% | {_fmt_num(row['step_time_ms'], '.1f')} | "
                f"{row['data_wait_fraction'] * 100:.1f}% | {_fmt_num(row['speedup_vs_baseline'], '.2f', 'x')} | "
                f"{row['mfu']} | {_fmt_num(row['gpu_tokens_per_sec'], ',.1f')} | {row['notes'] or 'ok'} |"
            )
        else:
            lines.append(
                f"| `{row['config']}` | — | — | — | — | — | {row['mfu']} | "
                f"{_fmt_num(row['gpu_tokens_per_sec'], ',.1f')} | `{row['status']}`: {row['notes']} |"
            )
    lines.append("")

    # Recipe 5
    r5 = payload["recipe5_plate_rehearsal"]
    lines.append("## Recipe 5: Batched 96-Well Plate Rehearsal\n")
    lines.append("| Batch Plates | Steps / Plate | Volume Conserved (sanity check, true by construction) | CPU (plates/s) | CPU Time (ms) | GPU (plates/s) |")
    lines.append("|---:|---:|---|---:|---:|---|")
    lines.append(
        f"| {r5['batch_plates']} | {r5['steps_per_plate']} | `{r5['volume_conserved']}` | "
        f"{_fmt_num(r5['cpu_plates_per_sec'], ',.1f')} | {_fmt_num(r5['cpu_time_ms'], '.2f')} | "
        f"{_fmt_num(r5['gpu_plates_per_sec'], ',.1f')} |"
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


def resolve_part_dir(out_dir: Path, dev: torch.device, tag: str = "") -> Path:
    """Return device-isolated part cache directory (`parts*` on CPU vs `parts_gpu*` on CUDA)."""
    base = "parts_gpu" if dev.type == "cuda" else "parts"
    return out_dir / (base + (f"_{tag}" if tag else ""))


def _part_device_signature(machine: dict[str, Any]) -> tuple[str, str]:
    dev_type = str(
        machine.get("device_type", "cuda" if machine.get("cuda_available", False) else "cpu")
    )
    gpu_name = str(machine.get("gpu_name", "none (CPU only)"))
    return dev_type, gpu_name


def assemble_payload(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge saved recipe parts into one report payload.

    Raises SystemExit if the parts disagree on device (`cpu` vs `cuda` / `gpu_name`) or
    `torch_num_threads`, preventing CPU/GPU cache cross-contamination.
    """
    dev_sigs = {k: _part_device_signature(v["machine"]) for k, v in loaded.items()}
    if len(set(dev_sigs.values())) > 1:
        detail = ", ".join(f"{k}={dt_name[0]}:{dt_name[1]}" for k, dt_name in dev_sigs.items())
        raise SystemExit(
            f"Saved parts disagree on device/GPU ({detail}). Refusing to assemble a mixed CPU/GPU report. "
            "Re-run with --fresh, or keep CPU and GPU parts in separate cache directories."
        )

    threads = {k: v["machine"]["torch_num_threads"] for k, v in loaded.items()}
    if len(set(threads.values())) > 1:
        detail = ", ".join(f"{k}={t}" for k, t in threads.items())
        raise SystemExit(
            f"Saved parts disagree on torch_num_threads ({detail}). Refusing to assemble a mixed report. "
            "Re-run with --fresh, or use a separate --tag for each --threads setting."
        )
    payload = {
        "machine": loaded["recipe1_validation"]["machine"],
        "machine_per_recipe": {k: v["machine"] for k, v in loaded.items()},
        "total_benchmark_wall_sec": round(sum(v["wall_sec"] for v in loaded.values()), 2),
    }
    payload.update({k: v["result"] for k, v in loaded.items()})
    return payload


def build_recipe_runners(
    peak_tflops_override: float | None = None,
    smoke: bool = False,
) -> dict[str, tuple[str, Any]]:
    if smoke:
        return {
            "recipe1_validation": (
                "Recipe 1: Batched protocol validation (smoke)",
                lambda dev: bench_recipe1(dev, repeats=1, batch_sizes=(1, 64)),
            ),
            "recipe2_twins_and_factory": (
                "Recipe 2: Seeded-error twins, generate/check/keep, batched repair (smoke)",
                lambda dev: bench_recipe2(dev, repeats=1, num_candidates=256, base_batch=16),
            ),
            "recipe3_planning_and_scheduling": (
                "Recipe 3: Step-DAG planning and instrument scheduling (smoke)",
                lambda dev: bench_recipe3(dev, repeats=1, sa_steps=30),
            ),
            "recipe4_fill_one_gpu": (
                "Recipe 4: Fill one GPU ablation (smoke)",
                lambda dev: bench_recipe4(
                    dev, repeats=1, peak_tflops_override=peak_tflops_override, smoke=True
                ),
            ),
            "recipe5_plate_rehearsal": (
                "Recipe 5: Batched 96-well plate rehearsal (smoke)",
                lambda dev: bench_recipe5(dev, repeats=1, batch_plates=16),
            ),
        }
    return {
        "recipe1_validation": ("Recipe 1: Batched protocol validation", lambda dev: bench_recipe1(dev, repeats=5)),
        "recipe2_twins_and_factory": ("Recipe 2: Seeded-error twins, generate/check/keep, batched repair", lambda dev: bench_recipe2(dev, repeats=3)),
        "recipe3_planning_and_scheduling": ("Recipe 3: Step-DAG planning and instrument scheduling", lambda dev: bench_recipe3(dev, repeats=3)),
        "recipe4_fill_one_gpu": (
            "Recipe 4: Fill one GPU ablation (median of 3 repeats)",
            lambda dev: bench_recipe4(dev, repeats=3, peak_tflops_override=peak_tflops_override),
        ),
        "recipe5_plate_rehearsal": ("Recipe 5: Batched 96-well plate rehearsal", lambda dev: bench_recipe5(dev, repeats=3)),
    }


def run_smoke_verification(
    device: str | torch.device | None = None,
    peak_tflops_override: float | None = None,
) -> dict[str, Any]:
    """Execute an honest, self-contained CPU/CUDA smoke check without mutating committed results."""
    dev = default_device(device)
    machine = read_cpu_info(dev)
    runners = build_recipe_runners(peak_tflops_override=peak_tflops_override, smoke=True)
    loaded: dict[str, dict[str, Any]] = {}
    for key, (_title, fn) in runners.items():
        t0 = time.perf_counter()
        res = fn(dev)
        loaded[key] = {
            "machine": machine,
            "wall_sec": round(time.perf_counter() - t0, 2),
            "result": res,
        }
    payload = assemble_payload(loaded)
    _ = format_markdown_report(payload)
    return payload


def main() -> None:
    """Run each recipe and save its result right away, so a crash does not lose finished parts.

    CPU and CUDA parts are isolated in separate directories (`parts*` vs `parts_gpu*`) so
    `--device cuda` never silently reuses committed CPU caches.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run one recipe key, then assemble if all parts exist")
    ap.add_argument("--fresh", action="store_true", help="ignore saved parts")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (use 1 on a shared, busy machine)")
    ap.add_argument("--tag", default="", help="suffix for the parts folder and report names, e.g. t1")
    ap.add_argument("--device", default=None, help="device override, e.g. cpu or cuda")
    ap.add_argument("--peak-tflops", type=float, default=None, help="override hardware peak BF16/FP32 TFLOPS for MFU")
    ap.add_argument("--smoke", action="store_true", help="run fast smoke verification in memory without overwriting results")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.smoke:
        payload = run_smoke_verification(device=args.device, peak_tflops_override=args.peak_tflops)
        print(format_markdown_report(payload))
        return

    dev = default_device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but torch.cuda.is_available() is False.")

    machine = read_cpu_info(dev)
    runners = build_recipe_runners(peak_tflops_override=args.peak_tflops, smoke=False)
    out_dir = REPO_ROOT / "benchmarks" / "results"
    part_dir = resolve_part_dir(out_dir, dev, tag=args.tag)
    part_dir.mkdir(parents=True, exist_ok=True)
    target_desc = f"{machine['gpu_name']} (CUDA)" if dev.type == "cuda" else f"{machine['cpu_model']} ({machine['cpu_cores']} cores)"
    print(f"Running labfill benchmarks on {target_desc}...")

    keys = [args.only] if args.only else list(runners)
    for i, key in enumerate(keys, 1):
        part = part_dir / f"{key}.json"
        if part.exists() and not args.fresh:
            saved_obj = json.loads(part.read_text(encoding="utf-8"))
            saved_machine = saved_obj["machine"]
            same_dev = _part_device_signature(saved_machine) == _part_device_signature(machine)
            same_threads = (
                args.threads is None or saved_machine.get("torch_num_threads") == torch.get_num_threads()
            )
            if same_dev and same_threads:
                saved_threads = saved_machine["torch_num_threads"]
                print(
                    f"  [{i}/{len(keys)}] {key}: saved part found (device={dev.type}, threads={saved_threads}), "
                    "skipping; pass --fresh to re-measure"
                )
                continue
        title, fn = runners[key]
        print(f"  [{i}/{len(keys)}] {title}...", flush=True)
        t0 = time.perf_counter()
        res = fn(dev)
        part.write_text(
            json.dumps(
                {"machine": machine, "wall_sec": round(time.perf_counter() - t0, 2), "result": res},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    parts = {k: part_dir / f"{k}.json" for k in runners}
    if not all(p.exists() for p in parts.values()):
        print("Not all recipe parts exist yet; run the missing ones.")
        return
    loaded = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in parts.items()}
    payload = assemble_payload(loaded)

    date_tag = payload["machine"]["date"]
    prefix = "gpu" if dev.type == "cuda" else "cpu"
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = out_dir / f"{prefix}_{date_tag}{suffix}.json"
    md_path = out_dir / f"{prefix}_{date_tag}{suffix}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_report = format_markdown_report(payload)
    md_path.write_text(md_report, encoding="utf-8")
    print(f"Wrote JSON : {json_path}")
    print(f"Wrote Table: {md_path}\n")
    print(md_report)


if __name__ == "__main__":
    main()
