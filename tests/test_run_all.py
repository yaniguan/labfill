from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "benchmarks" / "results"


def _load_run_all():
    spec = importlib.util.spec_from_file_location("run_all", RESULTS.parent / "run_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_parts(folder: str) -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((RESULTS / folder).glob("*.json"))}


def test_assemble_refuses_mixed_thread_counts() -> None:
    run_all = _load_run_all()
    loaded = _load_parts("parts_t1")
    mixed = copy.deepcopy(loaded)
    mixed["recipe4_fill_one_gpu"] = _load_parts("parts")["recipe4_fill_one_gpu"]
    with pytest.raises(SystemExit, match="torch_num_threads"):
        run_all.assemble_payload(mixed)


def test_report_lists_threads_and_load_per_recipe() -> None:
    run_all = _load_run_all()
    payload = run_all.assemble_payload(_load_parts("parts_t1"))
    md = run_all.format_markdown_report(payload)
    for key in run_all.RECIPES:
        assert f"| `{key}` | 1 |" in md


def test_read_cpu_info_without_getloadavg(monkeypatch: pytest.MonkeyPatch) -> None:
    run_all = _load_run_all()
    monkeypatch.delattr(run_all.os, "getloadavg", raising=False)
    info = run_all.read_cpu_info()
    assert info["load_avg_1_5_15_at_start"] is None

    payload = run_all.assemble_payload(_load_parts("parts_t1"))
    for pm in payload["machine_per_recipe"].values():
        pm["load_avg_1_5_15_at_start"] = None
    md = run_all.format_markdown_report(payload)
    assert "| `recipe1_validation` | 1 | N/A |" in md


def test_format_markdown_report_cuda_payload() -> None:
    run_all = _load_run_all()
    payload = copy.deepcopy(run_all.assemble_payload(_load_parts("parts_t1")))
    payload["machine"]["cuda_available"] = True
    payload["machine"]["gpu_name"] = "MockCUDA H100"

    for r in payload["recipe1_validation"]:
        r["torch_cpu_proto_per_sec"] = "not yet measured"
        r["torch_gpu_proto_per_sec"] = 987654.3

    payload["recipe2_twins_and_factory"]["generate_check_keep"]["cpu_candidates_per_sec"] = "not yet measured"
    payload["recipe2_twins_and_factory"]["generate_check_keep"]["gpu_candidates_per_sec"] = 543210.9
    for tr in payload["recipe2_twins_and_factory"]["twins_and_repair"]:
        tr["repair_cpu_proto_per_sec"] = "not yet measured"
        tr["repair_gpu_proto_per_sec"] = 45678.2

    payload["recipe3_planning_and_scheduling"]["dag_beam_search"]["cpu_time_ms"] = "not yet measured"
    payload["recipe3_planning_and_scheduling"]["dag_beam_search"]["gpu_time_ms"] = 4.25
    for sr in payload["recipe3_planning_and_scheduling"]["instrument_scheduling"]:
        sr["cpu_time_ms"] = "not yet measured"
        sr["gpu_time_ms"] = 1.75

    for row in payload["recipe4_fill_one_gpu"]["ablation"]:
        if row["status"] == "ok":
            row["real_tokens_per_sec"] = "not yet measured"
            row["gpu_tokens_per_sec"] = 123456.7

    payload["recipe5_plate_rehearsal"]["cpu_plates_per_sec"] = "not yet measured"
    payload["recipe5_plate_rehearsal"]["cpu_time_ms"] = "not yet measured"
    payload["recipe5_plate_rehearsal"]["gpu_plates_per_sec"] = 84500.5

    md = run_all.format_markdown_report(payload)
    assert "MockCUDA H100" in md
    assert "all GPU columns below: `not yet measured`" not in md
    assert "987,654.3" in md
    assert "123,456.7" in md
    assert "not yet measured" in md
