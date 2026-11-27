"""Regression test preventing seeded README and saved-JSON benchmark numbers from drifting from code."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pytest

from labfill.fill_gpu import build_batches_from_items, generate_protocol_sequences
from labfill.planner import (
    batched_beam_plan_dag,
    make_assay_dag_problem,
    make_lab_schedule_durations,
    sample_random_topological_orders,
    schedule_batched_sa,
    schedule_greedy_list,
)
from labfill.plate import make_serial_dilution_plate_batch, rehearse_96well_protocols
from labfill.protocol import (
    MAX_CONTAINER_VOL_TENTHS_UL,
    LabStateBatch,
    make_random_valid_protocols,
)
from labfill.twins import (
    TWIN_ERROR_TYPES,
    batched_repair,
    generate_check_keep,
    make_error_twin,
    measure_twin_recall,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _latest_main_v2_json() -> dict:
    candidates = sorted((REPO_ROOT / "benchmarks" / "results").glob("cpu_*_main_v2.json"))
    assert candidates, "Missing benchmarks/results/cpu_*_main_v2.json fresh benchmark report"
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _parse_md_table_rows(md_text: str) -> dict[str, list[str]]:
    """Map backtick-quoted first-column key -> list of stripped cell strings across Markdown tables."""
    out: dict[str, list[str]] = {}
    for line in md_text.splitlines():
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = re.fullmatch(r"`([^`]+)`", cells[0])
        if m:
            out[m.group(1)] = cells
    return out


def test_seeded_readme_and_json_numbers_match_live_code() -> None:
    """Issue #40: recompute every seeded, machine-independent metric and assert zero README/JSON drift."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    report = _latest_main_v2_json()
    md_rows = _parse_md_table_rows(readme)

    # 1. Max safe container volume documented in both README and LabStateBatch docstring (Issue #37)
    assert LabStateBatch.__doc__ is not None
    assert "100_000_000" in LabStateBatch.__doc__ and "10,000 mL" in LabStateBatch.__doc__
    assert "10,000 mL" in readme and str(MAX_CONTAINER_VOL_TENTHS_UL) in readme.replace(",", "").replace("_", "")

    # 2. Recipe 2: recall, kept_fraction, passes_checker_after_1_edit, exact_recovery_rate, winning edits
    r2_json = report["recipe2_twins_and_factory"]
    recall_live = measure_twin_recall(batch_size=128, seed=0, device="cpu")
    assert recall_live.false_positive_rate_on_valid == r2_json["false_positive_rate_on_valid"] == 0.0

    gk_live = generate_check_keep(num_candidates=4096, mutation_rate=0.25, seed=10, device="cpu")
    kept_frac = round(gk_live.kept_fraction, 4)
    assert kept_frac == r2_json["generate_check_keep"]["kept_fraction"]
    assert f"kept `{gk_live.kept_fraction:.1%}`" in readme

    st_base, pr_base = make_random_valid_protocols(128, seed=22, device="cpu")
    r2_by_type = {row["error_type"]: row for row in r2_json["twins_and_repair"]}
    for idx, et in enumerate(TWIN_ERROR_TYPES):
        st_b, pr_b = make_error_twin(st_base, pr_base, et, seed=30 + idx)
        rr = batched_repair(st_b, pr_b, num_edits=26, reference_protocols=pr_base)
        jrow = r2_by_type[et]
        assert round(rr.repair_rate, 4) == jrow["passes_checker_after_1_edit"]
        assert round(rr.exact_recovery_rate or 0.0, 4) == jrow["exact_recovery_rate"]
        assert round(rr.mean_field_edit_distance or 0.0, 3) == jrow["mean_field_edit_distance"]
        assert rr.edit_type_counts == jrow["repair_edit_type_counts"]

        # Verify README Recipe-2 table row cells
        assert et in md_rows, f"Missing `{et}` row in README.md"
        cells = md_rows[et]
        assert cells[1] == f"{recall_live.per_error_recall[et]:.1%}"
        assert cells[2] == f"{rr.repair_rate:.1%}"
        assert cells[3] == f"{(rr.exact_recovery_rate or 0.0):.1%}"
        assert cells[4] == f"{(rr.mean_field_edit_distance or 0.0):.2f}"
        exp_edits = ", ".join(f"{k}: {v}" for k, v in rr.edit_type_counts.items())
        assert cells[5] == exp_edits

    # 3. Recipe 3: DAG random topo pass rate, beam candidate count, and 3 scheduler makespans
    r3_json = report["recipe3_planning_and_scheduling"]
    dag = make_assay_dag_problem(device="cpu")
    _, rand_topo = sample_random_topological_orders(dag, num_samples=512, seed=0)
    beam = batched_beam_plan_dag(dag, beam_width=32)
    assert round(rand_topo, 4) == r3_json["dag_beam_search"]["random_topological_valid_rate"]
    assert beam.found is True and r3_json["dag_beam_search"]["beam_found_valid"] is True
    assert beam.total_candidates_checked == r3_json["dag_beam_search"]["candidates_checked"]
    assert f"`{rand_topo:.1%}`" in readme
    assert f"`{beam.total_candidates_checked}` candidates checked" in readme

    durs = make_lab_schedule_durations(num_jobs=96, num_instruments=8, heterogeneous=True, seed=0, device="cpu")
    live_makespans = {
        "greedy_arrival": round(schedule_greedy_list(durs, 8, sort_lpt=False).makespan, 1),
        "greedy_lpt": round(schedule_greedy_list(durs, 8, sort_lpt=True).makespan, 1),
        "batched_sa_r256": round(
            schedule_batched_sa(durs, 8, num_restarts=256, num_steps=300, seed=0).makespan, 1
        ),
    }
    for srow in r3_json["instrument_scheduling"]:
        mname = srow["method"]
        assert live_makespans[mname] == srow["makespan_sec"]
        assert md_rows[mname][1] == f"{live_makespans[mname]:.1f}"

    # 4. Recipe 4: >= 5 seeds, [min, max] spread in README, and exact useful-token % match
    r4_json = report["recipe4_fill_one_gpu"]
    assert r4_json["num_seeds"] >= 5
    utf_padded: list[float] = []
    utf_packed: list[float] = []
    utf_obfd: list[float] = []
    for seed_r in range(r4_json["num_seeds"]):
        items = generate_protocol_sequences(num_sequences=96, min_len=8, max_len=512, seed=seed_r)
        pad_b = build_batches_from_items(items, "padded", batch_size=16)
        pk_b = build_batches_from_items(items, "packed", pack_len=512, pack_rows_per_batch=2)
        ob_b = build_batches_from_items(items, "packed_obfd", pack_len=512, pack_rows_per_batch=2)
        utf_padded.append(sum(b.real_tokens for b in pad_b) / sum(b.total_tokens for b in pad_b))
        utf_packed.append(sum(b.real_tokens for b in pk_b) / sum(b.total_tokens for b in pk_b))
        utf_obfd.append(sum(b.real_tokens for b in ob_b) / sum(b.total_tokens for b in ob_b))

    r4_by_cfg = {row["config"]: row for row in r4_json["ablation"]}
    assert r4_by_cfg["a_padded_baseline"]["useful_token_fraction"] == pytest.approx(
        round(statistics.median(utf_padded), 4)
    )
    assert r4_by_cfg["c_sequence_packed"]["useful_token_fraction"] == pytest.approx(
        round(statistics.median(utf_packed), 4)
    )
    assert r4_by_cfg["e1_obfd_packed_bf16"]["useful_token_fraction"] == pytest.approx(
        round(statistics.median(utf_obfd), 4)
    )

    for cfg, jrow in r4_by_cfg.items():
        assert cfg in md_rows, f"Missing `{cfg}` row in README.md"
        if jrow["status"] == "ok":
            cells = md_rows[cfg]
            assert cells[2] == f"{jrow['useful_token_fraction'] * 100:.1f}%"
            assert re.search(r"\[\d+\.\d+x,\s*\d+\.\d+x\]", cells[5]), (cfg, cells[5])
            assert jrow["speedup_min"] <= jrow["speedup_vs_baseline"] <= jrow["speedup_max"]

    # 5. Recipe 5: deterministic plate batch dimensions & volume conservation
    r5_json = report["recipe5_plate_rehearsal"]
    st5, pr5 = make_serial_dilution_plate_batch(batch_size=128, num_rows_active=4, seed=0, device="cpu")
    res5 = rehearse_96well_protocols(st5, pr5)
    assert pr5.max_steps == r5_json["steps_per_plate"]
    assert bool(res5.volume_conserved.all().item()) is True and r5_json["volume_conserved"] is True
