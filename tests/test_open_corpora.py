"""Tests for real open-corpora tokenization, OBFD truncation-free packing, and protocol lowering."""

from __future__ import annotations

import json
from pathlib import Path

from labfill.open_corpora import (
    CORPORA_NAMES,
    evaluate_corpus_records,
    format_open_corpora_markdown,
    opentrons_commands_to_protocol,
    text_procedure_to_protocol,
    tokenize_protocol_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_open_corpora_lowering_and_obfd_evaluation() -> None:
    toks = tokenize_protocol_text("Transfer 25.0 uL buffer to well A1 and mix 3x.")
    assert toks[0] == 1 and toks[-1] == 2 and len(toks) > 6

    ot_cmds = [
        {"commandType": "pickUpTip", "params": {"pipetteId": "p1", "labwareId": "rack", "wellName": "A1"}},
        {"commandType": "aspirate", "params": {"pipetteId": "p1", "labwareId": "src_lw", "wellName": "A1", "volume": 50}},
        {"commandType": "dispense", "params": {"pipetteId": "p1", "labwareId": "dst_lw", "wellName": "B1", "volume": 50}},
        {"commandType": "dropTip", "params": {"pipetteId": "p1", "labwareId": "trash", "wellName": "A1"}},
    ]
    synthetic_records = [
        {"id": "ot_0", "text": "Opentrons transfer 50 uL A1 -> B1", "commands": ot_cmds},
        {
            "id": "wlp_0",
            "text": "Thaw frozen enzyme stock. Add 20 uL buffer to reaction tube. Mix 15 uL gently. Incubate at 37 C for 5 min.",
        },
        {
            "id": "ord_0",
            "text": "Charge 0.2 mL reagent solution, stir and heat at 85 C for 10 min, then centrifuge at 4000 rpm for 30 sec.",
        },
    ] * 8

    st_ot, steps_ot = opentrons_commands_to_protocol(ot_cmds)
    assert len(st_ot["volume"]) == 12 and len(steps_ot) >= 3
    st_tx, steps_tx = text_procedure_to_protocol(synthetic_records[1]["text"])
    assert len(st_tx["volume"]) == 8 and len(steps_tx) >= 3

    metrics = evaluate_corpus_records("synthetic_open", "unit-test", synthetic_records, checker_batch_target=64)
    assert metrics.truncation_count == 0
    assert metrics.checker_exact_python_parity is True
    assert metrics.obfd_bins <= metrics.first_fit_bins
    assert metrics.valid_protocol_fraction == 1.0
    md = format_open_corpora_markdown([metrics])
    assert "`synthetic_open`" in md


def test_committed_open_corpora_summary_invariants() -> None:
    summary_path = REPO_ROOT / "benchmarks" / "results" / "open_corpora_v2.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["all_zero_truncations"] is True
    assert payload["all_checker_python_parity"] is True
    assert payload["all_obfd_le_first_fit_bins"] is True
    by_name = {c["corpus"]: c for c in payload["corpora"]}
    assert set(by_name) == set(CORPORA_NAMES)
    for cname, row in by_name.items():
        assert row["num_documents"] >= 15, cname
        assert row["obfd_bins"] <= row["first_fit_bins"], cname
        assert row["obfd_useful_pct"] >= row["first_fit_useful_pct"], cname
        assert row["batched_checker_proto_per_sec"] > 0.0, cname
