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
