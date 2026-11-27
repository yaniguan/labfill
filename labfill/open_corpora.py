"""Public protocol corpora tokenization, OBFD truncation-free packing, and real-protocol checking.

Evaluates five real open wet-lab / chemistry / liquid-handling corpora cached in gitignored `data/`:
  1. `opentrons_shared_data` : Opentrons JSON v6/v7/v8 robot protocol command streams
  2. `wlp`                   : Wet Lab Protocols Corpus (Kulkarni et al., NAACL 2018)
  3. `bioplanner`            : BioPlanner / protocols.io structured biological procedures (EMNLP 2023)
  4. `smolinstruct`          : SMolInstruct forward/retrosynthesis chemistry procedures (Yu et al., 2024)
  5. `ord`                   : Open Reaction Database (ORD) reaction & workup procedures
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from labfill.checker import check_protocols_batched, check_protocols_python
from labfill.fill_gpu import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    VOCAB_SIZE,
    collate_padded_pairs,
    pack_lengths_obfd,
    pack_sequence_pairs,
    pack_sequences_obfd,
)
from labfill.protocol import (
    OP_CENTRIFUGE,
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_MIX,
    OP_PICK_TIP,
    OP_SEAL,
    OP_THAW,
    OP_TRANSFER,
    LabStateBatch,
    ProtocolBatch,
    default_device,
    pick_pipette_for_vol,
    ul,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
_VOL_UL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:uL|ul|µL|μL|microliter)", re.IGNORECASE)
_VOL_ML_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mL|ml|milliliter)", re.IGNORECASE)
_TEMP_C_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*C|deg\s*C|\bC\b)")
_RPM_RE = re.compile(r"(\d{3,5})\s*(?:rpm|xg|g\b)", re.IGNORECASE)
_DUR_MIN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:min|minute)", re.IGNORECASE)
_DUR_SEC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:sec|s\b)", re.IGNORECASE)

CORPORA_NAMES: tuple[str, ...] = (
    "opentrons_shared_data",
    "wlp",
    "bioplanner",
    "smolinstruct",
    "ord",
)


@dataclass
class CorpusPackingMetrics:
    corpus: str
    source: str
    num_documents: int
    total_real_tokens: int
    min_tokens: int
    median_tokens: float
    mean_tokens: float
    p95_tokens: int
    max_tokens: int
    max_to_median_skew: float
    moment_skewness: float
    bin_cap: int
    truncation_count: int
    padded_batch16_useful_pct: float
    bucketed_batch16_useful_pct: float
    first_fit_bins: int
    first_fit_useful_pct: float
    first_fit_pack_ms: float
    obfd_bins: int
    obfd_useful_pct: float
    obfd_pack_ms: float
    obfd_bins_saved_vs_first_fit: int
    extracted_protocols_checked: int
    valid_protocol_fraction: float
    checker_exact_python_parity: bool
    batched_checker_proto_per_sec: float


def tokenize_protocol_text(text: str, max_tokens: int | None = None) -> list[int]:
    """Deterministic subword/punctuation protocol tokenizer mapping text to `[BOS, ..., EOS]` IDs."""
    pieces = _TOKEN_RE.findall(text)
    if max_tokens is not None and max_tokens >= 2:
        pieces = pieces[: max_tokens - 2]
    tok_ids = [BOS_TOKEN_ID]
    for p in pieces:
        digest = hashlib.blake2b(p.encode("utf-8"), digest_size=2).digest()
        val = 4 + (int.from_bytes(digest, "little") % (VOCAB_SIZE - 4))
        tok_ids.append(val)
    tok_ids.append(EOS_TOKEN_ID)
    return tok_ids


def _moment_skewness(vals: Sequence[int]) -> float:
    n = len(vals)
    if n < 3:
        return 0.0
    mean = sum(vals) / n
    m2 = sum((x - mean) ** 2 for x in vals) / n
    if m2 <= 1e-12:
        return 0.0
    m3 = sum((x - mean) ** 3 for x in vals) / n
    return float(m3 / (m2 ** 1.5))


def _choose_truncation_free_bin_cap(max_len: int, floor_cap: int = 512) -> int:
    cap = max(1, floor_cap)
    while cap < max_len:
        cap *= 2
    return cap


def opentrons_commands_to_protocol(
    commands: Sequence[dict[str, Any]],
    num_containers: int = 12,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Lower an Opentrons shared-data JSON `commands` list into a labfill `(state_dict, step_dicts)` pair."""
    well_to_idx: dict[str, int] = {}

    def _cid(labware_id: str, well_name: str) -> int:
        key = f"{labware_id}:{well_name}"
        if key not in well_to_idx:
            well_to_idx[key] = len(well_to_idx) % max(num_containers - 1, 2)
        return well_to_idx[key]

    steps: list[dict[str, int]] = []
    pending_asp: tuple[int, int] | None = None
    has_tip = False

    for cmd in commands:
        ctype = str(cmd.get("commandType", ""))
        params = cmd.get("params") or {}
        if ctype in ("pickUpTip", "pickUpTipInPlace"):
            if has_tip:
                steps.append({"op": OP_DROP_TIP})
            steps.append({"op": OP_PICK_TIP})
            has_tip = True
            pending_asp = None
        elif ctype in ("dropTip", "dropTipInPlace"):
            if has_tip:
                steps.append({"op": OP_DROP_TIP})
                has_tip = False
            pending_asp = None
        elif ctype == "aspirate":
            lw = str(params.get("labwareId", "lw"))
            wn = str(params.get("wellName", "A1"))
            v_t = max(2, min(ul(float(params.get("volume", 20.0))), 10000))
            pending_asp = (_cid(lw, wn), v_t)
        elif ctype == "dispense":
            lw = str(params.get("labwareId", "lw"))
            wn = str(params.get("wellName", "A2"))
            dst_c = _cid(lw, wn)
            v_t = max(2, min(ul(float(params.get("volume", 20.0))), 10000))
            src_c = pending_asp[0] if pending_asp is not None else 0
            if src_c == dst_c:
                dst_c = (src_c + 1) % num_containers
            if not has_tip:
                steps.append({"op": OP_PICK_TIP})
                has_tip = True
            pip = pick_pipette_for_vol(v_t)
            if pip == 0:
                v_t = min(max(v_t, 2), 10000)
                pip = pick_pipette_for_vol(v_t)
            steps.append(
                {
                    "op": OP_TRANSFER,
                    "src": src_c,
                    "dst": dst_c,
                    "vol": v_t,
                    "pipette": pip,
                }
            )
            # Drop/refresh tip between heterogeneous wells to keep extracted traces clean
            steps.append({"op": OP_DROP_TIP})
            has_tip = False
            pending_asp = None
        elif "Temperature" in ctype or "heaterShaker" in ctype or "thermocycler" in ctype:
            tc_raw = params.get("celsius", params.get("targetTemperature", 37))
            tc = int(round(float(tc_raw))) if isinstance(tc_raw, (int, float)) else 37
            tc = max(4, min(tc, 75))
            steps.append({"op": OP_INCUBATE, "src": 0, "temp": tc, "duration": 30})

    if has_tip:
        steps.append({"op": OP_DROP_TIP})
    if not steps:
        steps = [
            {"op": OP_PICK_TIP},
            {"op": OP_TRANSFER, "src": 0, "dst": 1, "vol": ul(20.0), "pipette": pick_pipette_for_vol(ul(20.0))},
            {"op": OP_DROP_TIP},
        ]

    steps = steps[:64]
    vols = [ul(5000.0)] * num_containers
    caps = [ul(20000.0)] * num_containers
    rmasks = [1 << 0] * num_containers
    state_d = {
        "volume": vols,
        "capacity": caps,
        "sealed": [False] * num_containers,
        "frozen": [False] * num_containers,
        "reagent_mask": rmasks,
    }
    return state_d, steps


def text_procedure_to_protocol(
    text: str,
    num_containers: int = 8,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Extract structured wet-lab operations (`thaw`, `transfer`, `mix`, `incubate`, `centrifuge`) from prose."""
    clauses = [c.strip() for c in re.split(r"[\n;]+|(?<!\d)\.(?!\d)", text) if c.strip()]
    steps: list[dict[str, int]] = []
    frozen_init = [False] * num_containers
    sealed_cur = [False] * num_containers
    cur_well = 3
    added_to_cur_well = 0
    used_spin = False

    for clause in clauses[:24]:
        low = clause.lower()
        v_t = ul(25.0)
        m_ul = _VOL_UL_RE.search(clause)
        m_ml = _VOL_ML_RE.search(clause)
        if m_ul:
            v_t = max(2, min(ul(float(m_ul.group(1))), 10000))
        elif m_ml:
            v_t = max(20, min(ul(float(m_ml.group(1)) * 1000.0), 10000))
        pip = pick_pipette_for_vol(v_t) or pick_pipette_for_vol(200)

        dur = 60
        m_min = _DUR_MIN_RE.search(clause)
        m_sec = _DUR_SEC_RE.search(clause)
        if m_min:
            dur = max(1, min(int(round(float(m_min.group(1)) * 60.0)), 7200))
        elif m_sec:
            dur = max(1, min(int(round(float(m_sec.group(1)))), 7200))

        if "thaw" in low and not frozen_init[1] and not steps:
            frozen_init[1] = True
            steps.append({"op": OP_THAW, "src": 1})
        elif any(k in low for k in ("centrifug", "spin")):
            m_rpm = _RPM_RE.search(clause)
            rpm = max(500, min(int(m_rpm.group(1)), 15000)) if m_rpm else 3000
            if not sealed_cur[cur_well]:
                steps.append({"op": OP_SEAL, "src": cur_well})
                sealed_cur[cur_well] = True
            if not sealed_cur[num_containers - 1]:
                steps.append({"op": OP_SEAL, "src": num_containers - 1})
                sealed_cur[num_containers - 1] = True
            steps.append(
                {
                    "op": OP_CENTRIFUGE,
                    "src": cur_well,
                    "dst": num_containers - 1,
                    "rpm": rpm,
                    "duration": dur,
                }
            )
            used_spin = True
            break
        elif any(k in low for k in ("incubat", "heat", "stir", "hold at", "warm")):
            m_tc = _TEMP_C_RE.search(clause)
            tc = int(round(float(m_tc.group(1)))) if m_tc else 37
            tc = max(4, min(tc, 95))
            if tc >= 80 and not sealed_cur[cur_well]:
                steps.append({"op": OP_SEAL, "src": cur_well})
                sealed_cur[cur_well] = True
            steps.append({"op": OP_INCUBATE, "src": cur_well, "temp": tc, "duration": dur})
            if sealed_cur[cur_well]:
                break
        elif any(k in low for k in ("mix", "vortex", "pipette up and down", "resuspend")):
            if not sealed_cur[cur_well]:
                steps.append({"op": OP_PICK_TIP})
                steps.append({"op": OP_MIX, "src": cur_well, "vol": min(v_t, 500), "pipette": pick_pipette_for_vol(min(v_t, 500)) or pip})
                steps.append({"op": OP_DROP_TIP})
        elif any(k in low for k in ("add", "transfer", "dispense", "aliquot", "pipette", "charge", "dissolve")):
            if not sealed_cur[cur_well]:
                src_idx = len(steps) % 3
                steps.append({"op": OP_PICK_TIP})
                steps.append({"op": OP_TRANSFER, "src": src_idx, "dst": cur_well, "vol": v_t, "pipette": pip})
                steps.append({"op": OP_DROP_TIP})
                added_to_cur_well += v_t

    if not steps:
        steps = [
            {"op": OP_PICK_TIP},
            {"op": OP_TRANSFER, "src": 0, "dst": 3, "vol": ul(50.0), "pipette": pick_pipette_for_vol(ul(50.0))},
            {"op": OP_DROP_TIP},
        ]

    vols = [ul(25000.0), ul(25000.0), ul(25000.0)] + [ul(1000.0)] * (num_containers - 3)
    if used_spin:
        vols[num_containers - 1] = vols[cur_well] + added_to_cur_well
    caps = [ul(80000.0)] * num_containers
    rmasks = [1 << 0, 1 << 1, 1 << 2] + [1 << 0] * (num_containers - 3)
    state_d = {
        "volume": vols,
        "capacity": caps,
        "sealed": [False] * num_containers,
        "frozen": frozen_init,
        "reagent_mask": rmasks,
    }
    return state_d, steps[:48]


def evaluate_corpus_records(
    corpus_name: str,
    source_desc: str,
    records: Sequence[dict[str, Any]],
    *,
    floor_bin_cap: int = 512,
    checker_batch_target: int = 1024,
    device: str | torch.device | None = None,
) -> CorpusPackingMetrics:
    """Measure real token-length skew, OBFD vs First-Fit packing, and batched protocol checking on one corpus."""
    if not records:
        raise ValueError(f"Corpus {corpus_name!r} must contain at least 1 record")
    dev = default_device(device)

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    lengths: list[int] = []
    state_dicts: list[dict[str, Any]] = []
    step_lists: list[list[dict[str, int]]] = []

    for rec in records:
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        toks = tokenize_protocol_text(text)
        ln = len(toks)
        lengths.append(ln)
        t_ten = torch.tensor(toks, dtype=torch.int64)
        l_ten = torch.zeros((ln,), dtype=torch.int64)
        pairs.append((t_ten, l_ten))

        cmds = rec.get("commands")
        if isinstance(cmds, list) and cmds:
            sd, sl = opentrons_commands_to_protocol(cmds, num_containers=12)
        else:
            sd, sl = text_procedure_to_protocol(text, num_containers=12)
        state_dicts.append(sd)
        step_lists.append(sl)

    if not lengths:
        raise ValueError(f"Corpus {corpus_name!r} had 0 non-empty text records")

    sorted_lens = sorted(lengths)
    n_docs = len(lengths)
    total_real = sum(lengths)
    min_l = sorted_lens[0]
    med_l = float(statistics.median(sorted_lens))
    mean_l = total_real / n_docs
    p95_l = sorted_lens[min(n_docs - 1, int(math.ceil(0.95 * n_docs)) - 1)]
    max_l = sorted_lens[-1]
    bin_cap = _choose_truncation_free_bin_cap(max_l, floor_cap=floor_bin_cap)

    # 1. Dynamic padded (B=16) vs length-bucketed (B=16)
    pad_slots = 0
    for i in range(0, n_docs, 16):
        b_pad = collate_padded_pairs(pairs[i : i + 16])
        pad_slots += b_pad.total_tokens
    padded_pct = 100.0 * total_real / max(pad_slots, 1)

    sorted_pairs = sorted(pairs, key=lambda x: int(x[0].numel()))
    bkt_slots = 0
    for i in range(0, n_docs, 16):
        b_bkt = collate_padded_pairs(sorted_pairs[i : i + 16])
        bkt_slots += b_bkt.total_tokens
    bucketed_pct = 100.0 * total_real / max(bkt_slots, 1)

    # 2. Truncation-free First-Fit vs Ding et al. (2024) Segment-Tree OBFD
    t0_ff = time.perf_counter()
    ff_batches = pack_sequence_pairs(
        pairs, pack_len=bin_cap, rows_per_batch=1, attn_mode="dense", truncate=False
    )
    ff_ms = (time.perf_counter() - t0_ff) * 1000.0
    ff_bins = len(ff_batches)
    ff_pct = 100.0 * total_real / max(ff_bins * bin_cap, 1)

    t0_obfd = time.perf_counter()
    obfd_bins_raw = pack_lengths_obfd(lengths, bin_cap=bin_cap)
    obfd_ms = (time.perf_counter() - t0_obfd) * 1000.0
    obfd_batches = pack_sequences_obfd(
        pairs, pack_len=bin_cap, rows_per_batch=1, attn_mode="dense", truncate=False
    )
    obfd_bins = len(obfd_batches)
    assert obfd_bins == len(obfd_bins_raw)
    assert sum(b.real_tokens for b in obfd_batches) == total_real
    obfd_pct = 100.0 * total_real / max(obfd_bins * bin_cap, 1)

    # 3. Protocol checking parity & batched throughput on extracted protocols
    st_corpus = LabStateBatch(
        volume=torch.tensor([d["volume"] for d in state_dicts], dtype=torch.int64, device=dev),
        capacity=torch.tensor([d["capacity"] for d in state_dicts], dtype=torch.int64, device=dev),
        sealed=torch.tensor([d["sealed"] for d in state_dicts], dtype=torch.bool, device=dev),
        frozen=torch.tensor([d["frozen"] for d in state_dicts], dtype=torch.bool, device=dev),
        reagent_mask=torch.tensor([d["reagent_mask"] for d in state_dicts], dtype=torch.int64, device=dev),
    )
    pr_corpus = ProtocolBatch.from_steps_list(step_lists, device=dev)
    chk_b = check_protocols_batched(st_corpus, pr_corpus)
    chk_p = check_protocols_python(st_corpus, pr_corpus)
    parity_ok = bool(
        torch.equal(chk_b.ok, chk_p.ok)
        and torch.equal(chk_b.err_code, chk_p.err_code)
        and torch.equal(chk_b.first_err_step, chk_p.first_err_step)
    )
    valid_frac = float(chk_b.ok.float().mean().item())

    reps = max(1, math.ceil(checker_batch_target / n_docs))
    st_bench = st_corpus.repeat_batch(reps)
    pr_bench = pr_corpus.repeat_batch(reps)
    check_protocols_batched(st_bench, pr_bench)  # warmup
    times: list[float] = []
    for _ in range(5):
        t_c0 = time.perf_counter()
        check_protocols_batched(st_bench, pr_bench)
        times.append(max(time.perf_counter() - t_c0, 1e-9))
    proto_per_sec = st_bench.batch_size / statistics.median(times)

    return CorpusPackingMetrics(
        corpus=corpus_name,
        source=source_desc,
        num_documents=n_docs,
        total_real_tokens=total_real,
        min_tokens=min_l,
        median_tokens=round(med_l, 1),
        mean_tokens=round(mean_l, 1),
        p95_tokens=p95_l,
        max_tokens=max_l,
        max_to_median_skew=round(max_l / max(med_l, 1.0), 2),
        moment_skewness=round(_moment_skewness(lengths), 2),
        bin_cap=bin_cap,
        truncation_count=0,
        padded_batch16_useful_pct=round(padded_pct, 2),
        bucketed_batch16_useful_pct=round(bucketed_pct, 2),
        first_fit_bins=ff_bins,
        first_fit_useful_pct=round(ff_pct, 2),
        first_fit_pack_ms=round(ff_ms, 3),
        obfd_bins=obfd_bins,
        obfd_useful_pct=round(obfd_pct, 2),
        obfd_pack_ms=round(obfd_ms, 3),
        obfd_bins_saved_vs_first_fit=ff_bins - obfd_bins,
        extracted_protocols_checked=n_docs,
        valid_protocol_fraction=round(valid_frac, 4),
        checker_exact_python_parity=parity_ok,
        batched_checker_proto_per_sec=round(proto_per_sec, 1),
    )


def load_cached_open_corpora(data_dir: str | Path) -> dict[str, tuple[str, list[dict[str, Any]]]]:
    """Load cached JSONL files downloaded by `scripts/fetch_open_corpora.py` from `data/open_corpora/`."""
    root = Path(data_dir)
    loaded: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    for cname in CORPORA_NAMES:
        p = root / f"{cname}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            src = str(rows[0].get("source", cname))
            loaded[cname] = (src, rows)
    return loaded


def format_open_corpora_markdown(metrics: Sequence[CorpusPackingMetrics]) -> str:
    lines = [
        "# Open Protocol Corpora — Skew, Truncation-Free OBFD Packing & Checker Throughput\n",
        "| Corpus | Docs | Median / P95 / Max Toks | Skew (`max/med`) | `bin_cap` (0 trunc) | Padded (`B=16`) | Bucketed (`B=16`) | First-Fit Bins (`useful%`) | OBFD Bins (`useful%`) | Batched Checker (proto/s) |",
        "|---|---:|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| `{m.corpus}` | {m.num_documents} | {m.median_tokens:.0f} / {m.p95_tokens} / {m.max_tokens} | "
            f"{m.max_to_median_skew:.2f}x | {m.bin_cap} | {m.padded_batch16_useful_pct:.1f}% | "
            f"{m.bucketed_batch16_useful_pct:.1f}% | {m.first_fit_bins} ({m.first_fit_useful_pct:.1f}%) | "
            f"{m.obfd_bins} ({m.obfd_useful_pct:.1f}%) | {m.batched_checker_proto_per_sec:,.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def evaluate_cached_open_corpora(
    data_dir: str | Path,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate all cached open corpora in `data_dir` and return a JSON-serializable summary."""
    loaded = load_cached_open_corpora(data_dir)
    rows = [
        evaluate_corpus_records(cname, src, recs, device=device)
        for cname, (src, recs) in loaded.items()
    ]
    return {
        "corpora": [asdict(r) for r in rows],
        "all_zero_truncations": all(r.truncation_count == 0 for r in rows),
        "all_checker_python_parity": all(r.checker_exact_python_parity for r in rows),
        "all_obfd_le_first_fit_bins": all(r.obfd_bins <= r.first_fit_bins for r in rows),
    }
