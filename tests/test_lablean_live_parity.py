"""Live conformance tests compiling 310+ exported labfill protocols against YLM LabLean."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from labfill.checker import check_protocols_batched
from labfill.lablean_io import (
    LABLEAN_DOCUMENTED_DELTAS,
    export_lablean_batch_module,
    lablean_error_to_err_code,
)
from labfill.plate import make_serial_dilution_plate_batch
from labfill.protocol import (
    ERR_BAD_DURATION,
    ERR_BAD_RPM,
    ERR_FROZEN,
    ERR_NOT_SEALED,
    ERR_OK,
    ERR_UNBALANCED,
    OP_CENTRIFUGE,
    OP_INCUBATE,
    OP_THAW,
    OP_UNSEAL,
    LabStateBatch,
    ProtocolBatch,
    make_pcr_protocol,
    make_random_protocols,
    make_random_valid_protocols,
)
from labfill.twins import TWIN_ERROR_TYPES, make_error_twin

REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_lake_exe() -> str | None:
    found = shutil.which("lake") or shutil.which("lake.exe")
    if found:
        return found
    elan_lake = Path.home() / ".elan" / "bin" / ("lake.exe" if os.name == "nt" else "lake")
    if elan_lake.exists():
        return str(elan_lake)
    return None


def _resolve_lablean_source_dir() -> Path | None:
    env_dir = os.environ.get("LABFILL_LABLEAN_DIR")
    if env_dir:
        p = Path(env_dir)
        if (p / "LabLean" / "Checker.lean").exists():
            return p
    sibling = REPO_ROOT.parent / "YLM" / "lean"
    if (sibling / "LabLean" / "Checker.lean").exists():
        return sibling
    return None


def _prepare_built_lablean_workspace(tmp_path: Path) -> tuple[str, Path]:
    """Return `(lake_exe, built_lablean_dir)`, copying to `tmp_path` if not pre-built so sibling YLM stays untouched."""
    lake_exe = _find_lake_exe()
    src_dir = _resolve_lablean_source_dir()
    if lake_exe is None or src_dir is None:
        pytest.skip(
            "Live LabLean parity test requires `lake` and `YLM/lean` "
            "(set LABFILL_LABLEAN_DIR or place YLM sibling repo next to labfill)."
        )

    # Only run directly in src_dir when LABFILL_LABLEAN_DIR was explicitly provided AND already built
    if os.environ.get("LABFILL_LABLEAN_DIR") and (src_dir / ".lake" / "build").exists():
        return lake_exe, src_dir

    work_dir = tmp_path / "ylm_lean_work"
    shutil.copytree(
        src_dir,
        work_dir,
        ignore=shutil.ignore_patterns(".lake", "build"),
    )
    build_proc = subprocess.run(
        [lake_exe, "build", "LabLean.Checker"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert build_proc.returncode == 0, (
        f"`lake build LabLean.Checker` failed (rc={build_proc.returncode}):\n"
        f"STDOUT:\n{build_proc.stdout}\nSTDERR:\n{build_proc.stderr}"
    )
    return lake_exe, work_dir


def _is_allowlisted_verdict_delta(
    st: LabStateBatch,
    pr: ProtocolBatch,
    row: int,
    lf_err: int,
    ll_err: int,
    raw_report: str,
) -> str | None:
    """Return the documented `LABLEAN_DOCUMENTED_DELTAS` key if an accept/reject mismatch is explained, else `None`."""
    ops_active = [
        (t, int(pr.op[row, t].item()))
        for t in range(pr.max_steps)
        if not bool(pr.pad_mask[row, t].item()) and int(pr.op[row, t].item()) != 0
    ]
    op_set = {op for _, op in ops_active}

    # 1. Protocol uses OP_UNSEAL (sentinel rejected by LabLean or unseal state delta)
    if OP_UNSEAL in op_set or "__LABFILL_UNSUPPORTED_UNSEAL_" in raw_report:
        return "unseal_extension"

    # 2. Per-container vs per-reagent thaw / frozen incubate / redundant thaw
    if lf_err == ERR_FROZEN and ll_err == ERR_OK:
        if OP_THAW in op_set or OP_INCUBATE in op_set:
            return "per_container_vs_per_reagent_thaw"
    if "__LABFILL_EMPTY_THAW_" in raw_report:
        return "per_container_vs_per_reagent_thaw"

    # 3. High-temp unsealed incubate, non-positive duration, or zero/negative RPM
    if lf_err in (ERR_NOT_SEALED, ERR_BAD_DURATION, ERR_BAD_RPM) and ll_err == ERR_OK:
        if OP_INCUBATE in op_set or OP_CENTRIFUGE in op_set:
            return "high_temp_seal_and_positive_duration_rpm"

    # 4. Centrifuge 5 uL volume tolerance vs 100 mg mass tolerance
    if lf_err == ERR_UNBALANCED and ll_err == ERR_OK and OP_CENTRIFUGE in op_set:
        return "rotor_volume_5ul_vs_mass_100mg"

    # Also handle multi-error corrupted protocols where an earlier step passed LabLean under one of the
    # documented relaxations (e.g. redundant thaw or 0-duration incubate) and later hit a real error or OK
    if ll_err == ERR_OK and lf_err != ERR_OK:
        for t, op in ops_active:
            if op == OP_THAW:
                return "per_container_vs_per_reagent_thaw"
            if op in (OP_INCUBATE, OP_CENTRIFUGE) and int(pr.duration[row, t].item()) <= 0:
                return "high_temp_seal_and_positive_duration_rpm"
            if op == OP_CENTRIFUGE and int(pr.rpm[row, t].item()) <= 0:
                return "high_temp_seal_and_positive_duration_rpm"
            if op == OP_INCUBATE and int(pr.temp[row, t].item()) >= 80:
                return "high_temp_seal_and_positive_duration_rpm"

    return None


def test_live_ylm_lablean_310_protocol_parity(tmp_path: Path) -> None:
    """Compile and evaluate 310 exported protocols in live YLM LabLean: 0 compile errors, 0 unexplained deltas."""
    lake_exe, work_dir = _prepare_built_lablean_workspace(tmp_path)

    # Assemble 310 protocols across 5 canonical suites:
    #   1 PCR + 4 96-well serial dilutions + 100 random valid (lablean_compatible=True, strict=True)
    #   + 100 random corrupted (error_prob=1.0, seed=8, strict=False)
    #   + 105 error twins (7 twin types x 15, strict=True)
    st_pcr, pr_pcr = make_pcr_protocol(device="cpu")
    st_dil, pr_dil = make_serial_dilution_plate_batch(batch_size=4, num_rows_active=1, seed=0, device="cpu")
    st_val, pr_val = make_random_valid_protocols(100, seed=7, device="cpu", lablean_compatible=True)
    st_cor, pr_cor = make_random_protocols(100, error_prob=1.0, seed=8, device="cpu", lablean_compatible=False)
    st_tw_base, pr_tw_base = make_random_valid_protocols(15, seed=19, device="cpu", lablean_compatible=True)

    batch_items: list[tuple[str, LabStateBatch, ProtocolBatch, int, bool]] = []
    meta: dict[str, tuple[str, LabStateBatch, ProtocolBatch, int, int]] = {}

    def add_suite(name: str, st: LabStateBatch, pr: ProtocolBatch, strict: bool) -> None:
        chk = check_protocols_batched(st, pr)
        for r in range(st.batch_size):
            tag = f"{name}_{r:03d}"
            batch_items.append((tag, st, pr, r, strict))
            meta[tag] = (name, st, pr, r, int(chk.err_code[r].item()))

    add_suite("pcr", st_pcr, pr_pcr, strict=True)
    add_suite("serial_dilution", st_dil, pr_dil, strict=True)
    add_suite("random_valid", st_val, pr_val, strict=True)
    add_suite("random_corrupted", st_cor, pr_cor, strict=False)
    for idx, et in enumerate(TWIN_ERROR_TYPES):
        st_tw, pr_tw = make_error_twin(st_tw_base, pr_tw_base, et, seed=40 + idx)
        add_suite(f"twin_{et}", st_tw, pr_tw, strict=True)

    assert len(batch_items) == 310

    lean_module_src = export_lablean_batch_module(batch_items, module_namespace="LiveParity310")
    assert "NONE" not in lean_module_src

    export_file = tmp_path / "LiveParity310.lean"
    export_file.write_text(lean_module_src, encoding="utf-8")

    eval_proc = subprocess.run(
        [lake_exe, "env", "lean", str(export_file)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert eval_proc.returncode == 0, (
        f"Lean compilation/evaluation failed on exported 310-protocol suite (rc={eval_proc.returncode}):\n"
        f"STDOUT:\n{eval_proc.stdout[:2000]}\nSTDERR:\n{eval_proc.stderr[:2000]}"
    )

    parsed_reports: dict[str, str] = {}
    for raw_line in eval_proc.stdout.splitlines():
        if "\t" in raw_line:
            tag, rep = raw_line.split("\t", 1)
            parsed_reports[tag.strip()] = rep.strip()

    assert len(parsed_reports) == 310, f"Expected 310 #eval report lines, got {len(parsed_reports)}"

    unexplained: list[dict[str, Any]] = []
    exact_code_matches_twins = 0
    valid_suite_matches = 0
    allowlist_hits: dict[str, int] = {}

    for tag, (suite, st, pr, r, lf_err) in meta.items():
        raw_rep = parsed_reports[tag]
        ll_err = lablean_error_to_err_code(raw_rep)
        lf_ok = lf_err == ERR_OK
        ll_ok = ll_err == ERR_OK

        if suite in ("pcr", "serial_dilution", "random_valid"):
            if lf_ok and ll_ok:
                valid_suite_matches += 1
            else:
                unexplained.append({"tag": tag, "lf_err": lf_err, "ll_err": ll_err, "report": raw_rep})
            continue

        if suite.startswith("twin_"):
            assert not lf_ok and not ll_ok, f"Twin {tag} unexpectedly accepted: lf={lf_err}, ll={ll_err}"
            if lf_err == ll_err:
                exact_code_matches_twins += 1
            else:
                unexplained.append({"tag": tag, "lf_err": lf_err, "ll_err": ll_err, "report": raw_rep})
            continue

        # random_corrupted: if accept/reject verdicts differ, must match an explicit documented delta
        if lf_ok != ll_ok:
            reason = _is_allowlisted_verdict_delta(st, pr, r, lf_err, ll_err, raw_rep)
            if reason is None or reason not in LABLEAN_DOCUMENTED_DELTAS:
                unexplained.append({"tag": tag, "lf_err": lf_err, "ll_err": ll_err, "report": raw_rep})
            else:
                allowlist_hits[reason] = allowlist_hits.get(reason, 0) + 1

    assert valid_suite_matches == 105  # 1 PCR + 4 serial dilution + 100 random valid
    assert exact_code_matches_twins == 105  # 7 twin types x 15 = 105/105 exact error code match
    assert unexplained == [], f"Unexplained LabLean verdict differences: {unexplained}"
