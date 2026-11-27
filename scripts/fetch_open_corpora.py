#!/usr/bin/env python3
"""Fetch real public protocol & chemistry corpora into gitignored `data/open_corpora/` and benchmark OBFD packing.

Sources fetched (all public, open-access research corpora):
  1. Opentrons/shared-data : https://github.com/Opentrons/opentrons (MIT / Apache-2.0 JSON v6/v7/v8 fixtures)
  2. WLP                   : https://github.com/chaitanya2334/WLP-Dataset (Wet Lab Protocols Corpus, NAACL 2018)
  3. BioPlanner            : https://github.com/bioplanner/bioplanner (BioPlanner `bioprot/*.json`, EMNLP 2023)
  4. SMolInstruct          : https://huggingface.co/datasets/osunlp/SMolInstruct (HTTP Range zip slice of dev JSONL)
  5. ORD                   : https://huggingface.co/datasets/open-reaction-database/ord-data (ORD reaction records)
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import urllib.request
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labfill.open_corpora import (  # noqa: E402
    CorpusPackingMetrics,
    evaluate_cached_open_corpora,
    format_open_corpora_markdown,
)

USER_AGENT = "labfill-open-corpora/1.0"


def _http_get_bytes(url: str, byte_range: tuple[int, int] | None = None, timeout: int = 20) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_json(url: str, timeout: int = 20) -> Any:
    return json.loads(_http_get_bytes(url, timeout=timeout).decode("utf-8", errors="replace"))


def _decode_protobuf_utf8_strings(buf: bytes, min_len: int = 4) -> list[str]:
    """Recursively extract human-readable UTF-8 string fields from a Protobuf wire buffer (wire type 2)."""
    found: list[str] = []

    def _walk(data: bytes, depth: int) -> None:
        if depth > 6:
            return
        idx = 0
        n = len(data)
        while idx < n:
            tag = 0
            shift = 0
            while idx < n:
                b = data[idx]
                idx += 1
                tag |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
                if shift > 35:
                    return
            wire = tag & 0x07
            if wire == 0:  # varint
                while idx < n and (data[idx] & 0x80):
                    idx += 1
                idx += 1
            elif wire == 1:  # 64-bit
                idx += 8
            elif wire == 5:  # 32-bit
                idx += 4
            elif wire == 2:  # length-delimited
                ln = 0
                shift = 0
                while idx < n:
                    b = data[idx]
                    idx += 1
                    ln |= (b & 0x7F) << shift
                    if not (b & 0x80):
                        break
                    shift += 7
                if ln < 0 or idx + ln > n:
                    return
                sub = data[idx : idx + ln]
                idx += ln
                try:
                    s = sub.decode("utf-8")
                    if len(s) >= min_len and all(ch.isprintable() or ch in "\n\r\t" for ch in s):
                        found.append(s.strip())
                        continue
                except UnicodeDecodeError:
                    pass
                _walk(sub, depth + 1)
            else:
                return

    _walk(buf, 0)
    return [s for s in found if s]


def fetch_opentrons_shared_data(max_docs: int = 40) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ver in ("6", "7", "8"):
        if len(records) >= max_docs:
            break
        listing_url = f"https://api.github.com/repos/Opentrons/opentrons/contents/shared-data/protocol/fixtures/{ver}"
        try:
            items = _http_get_json(listing_url)
        except Exception:
            continue
        for entry in items:
            if len(records) >= max_docs:
                break
            if not str(entry.get("name", "")).endswith(".json") or not entry.get("download_url"):
                continue
            doc = _http_get_json(entry["download_url"])
            cmds = doc.get("commands", [])
            if not isinstance(cmds, list) or not cmds:
                continue
            cmd_lines = [
                f"{c.get('commandType', 'op')}({json.dumps(c.get('params', {}), separators=(',', ':'))})"
                for c in cmds
            ]
            meta_name = str((doc.get("metadata") or {}).get("protocolName", entry["name"]))
            text = f"Opentrons v{ver} {meta_name}\n" + "\n".join(cmd_lines)
            records.append(
                {
                    "id": f"ot_v{ver}_{entry['name']}",
                    "source": "Opentrons/opentrons shared-data/protocol/fixtures",
                    "text": text,
                    "commands": cmds,
                }
            )
    return records


def fetch_wlp_corpus(max_docs: int = 80) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        if len(records) >= max_docs:
            break
        items = _http_get_json(f"https://api.github.com/repos/chaitanya2334/WLP-Dataset/contents/{split}")
        txt_entries = [x for x in items if str(x.get("name", "")).endswith(".txt") and x.get("download_url")]
        for entry in txt_entries[: max_docs - len(records)]:
            raw = _http_get_bytes(entry["download_url"]).decode("utf-8", errors="replace").strip()
            if raw:
                records.append(
                    {
                        "id": f"wlp_{split}_{entry['name']}",
                        "source": "chaitanya2334/WLP-Dataset (NAACL 2018)",
                        "text": raw,
                    }
                )
    return records


def fetch_bioplanner_corpus(max_docs: int = 80) -> list[dict[str, Any]]:
    items = _http_get_json("https://api.github.com/repos/bioplanner/bioplanner/contents/bioprot")
    json_entries = [x for x in items if str(x.get("name", "")).endswith(".json") and x.get("download_url")]
    records: list[dict[str, Any]] = []
    for entry in json_entries[:max_docs]:
        doc = _http_get_json(entry["download_url"])
        title = str(doc.get("title", ""))
        steps = doc.get("steps", [])
        if isinstance(steps, list):
            step_strs = [json.dumps(s, ensure_ascii=True) if isinstance(s, dict) else str(s) for s in steps]
            body = "\n".join(step_strs)
        else:
            body = str(steps)
        full_text = f"{title}\n{body}".strip()
        if full_text:
            records.append(
                {
                    "id": f"bioplanner_{doc.get('id', entry['name'])}",
                    "source": "bioplanner/bioplanner bioprot (EMNLP 2023)",
                    "text": full_text,
                }
            )
    return records


def fetch_smolinstruct_corpus(max_docs: int = 200) -> list[dict[str, Any]]:
    """Stream `raw/dev/forward_synthesis.jsonl` & `retrosynthesis.jsonl` from SMolInstruct `data.zip` via HTTP Range."""
    zip_url = "https://huggingface.co/datasets/osunlp/SMolInstruct/resolve/main/data.zip"
    req = urllib.request.Request(zip_url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        total_size = int(r.headers["Content-Range"].split("/")[-1])

    tail_len = min(65536, total_size)
    tail = _http_get_bytes(zip_url, byte_range=(total_size - tail_len, total_size - 1))
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("Could not locate ZIP EOCD in SMolInstruct data.zip")
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd + 12)
    cd = _http_get_bytes(zip_url, byte_range=(cd_offset, cd_offset + cd_size - 1))

    targets = {"raw/dev/forward_synthesis.jsonl", "raw/dev/retrosynthesis.jsonl"}
    pos = 0
    records: list[dict[str, Any]] = []
    per_target = max(1, max_docs // 2)

    while pos < len(cd) and cd[pos : pos + 4] == b"PK\x01\x02":
        comp_method = struct.unpack_from("<H", cd, pos + 10)[0]
        comp_sz = struct.unpack_from("<I", cd, pos + 20)[0]
        nlen, elen, clen = struct.unpack_from("<HHH", cd, pos + 28)
        lh_off = struct.unpack_from("<I", cd, pos + 42)[0]
        fname = cd[pos + 46 : pos + 46 + nlen].decode("utf-8", errors="replace")
        pos += 46 + nlen + elen + clen
        if fname not in targets:
            continue

        lh_hdr = _http_get_bytes(zip_url, byte_range=(lh_off, lh_off + 29))
        lh_nlen, lh_elen = struct.unpack_from("<HH", lh_hdr, 26)
        data_start = lh_off + 30 + lh_nlen + lh_elen
        comp_bytes = _http_get_bytes(zip_url, byte_range=(data_start, data_start + comp_sz - 1))
        raw_bytes = zlib.decompress(comp_bytes, -zlib.MAX_WBITS) if comp_method == 8 else comp_bytes
        lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
        task_tag = Path(fname).stem
        for i, ln in enumerate(lines[:per_target]):
            if not ln.strip():
                continue
            obj = json.loads(ln)
            inp = str(obj.get("input", obj.get("raw_input", "")))
            out = str(obj.get("output", obj.get("raw_output", "")))
            text = f"Task: {task_tag}\nReactants/Prompt: {inp}\nProcedure/Product: {out}".strip()
            records.append(
                {
                    "id": f"smol_{task_tag}_{i:04d}",
                    "source": "osunlp/SMolInstruct (raw/dev)",
                    "text": text,
                }
            )
    return records[:max_docs]


def fetch_ord_corpus(max_docs: int = 160) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    page = 80
    while len(records) < max_docs:
        take = min(page, max_docs - len(records))
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=open-reaction-database%2Ford-data&config=default&split=train&offset={offset}&length={take}"
        )
        payload = _http_get_json(url)
        rows = payload.get("rows", [])
        if not rows:
            break
        for r in rows:
            row_obj = r.get("row", {})
            rxn_id = str(row_obj.get("reaction_id", f"ord_{len(records)}"))
            b64_rxn = row_obj.get("reaction", "")
            raw_pb = base64.b64decode(b64_rxn) if isinstance(b64_rxn, str) and b64_rxn else b""
            extracted = _decode_protobuf_utf8_strings(raw_pb)
            text = f"Reaction {rxn_id}: " + " ; ".join(extracted[:40])
            if extracted:
                records.append(
                    {
                        "id": rxn_id,
                        "source": "open-reaction-database/ord-data (HuggingFace train split)",
                        "text": text,
                    }
                )
        offset += take
    return records[:max_docs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch real open protocol corpora and evaluate OBFD packing.")
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "open_corpora"))
    ap.add_argument("--out-json", default=str(REPO_ROOT / "benchmarks" / "results" / "open_corpora_v2.json"))
    ap.add_argument("--out-md", default=str(REPO_ROOT / "benchmarks" / "results" / "open_corpora_v2.md"))
    ap.add_argument("--refresh", action="store_true", help="re-download even if cached JSONL files exist")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    fetchers = (
        ("opentrons_shared_data", lambda: fetch_opentrons_shared_data(max_docs=40)),
        ("wlp", lambda: fetch_wlp_corpus(max_docs=80)),
        ("bioplanner", lambda: fetch_bioplanner_corpus(max_docs=80)),
        ("smolinstruct", lambda: fetch_smolinstruct_corpus(max_docs=200)),
        ("ord", lambda: fetch_ord_corpus(max_docs=160)),
    )

    for cname, fn in fetchers:
        cache_file = data_dir / f"{cname}.jsonl"
        if cache_file.exists() and not args.refresh:
            print(f"  [cache] {cname}: using {cache_file}")
            continue
        print(f"  [fetch] {cname}...", flush=True)
        recs = fn()
        payload_lines = "\n".join(json.dumps(r, ensure_ascii=True) for r in recs) + "\n"
        cache_file.write_text(payload_lines, encoding="utf-8")
        print(f"          saved {len(recs)} records -> {cache_file}")

    summary = evaluate_cached_open_corpora(data_dir, device="cpu")
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = format_open_corpora_markdown([CorpusPackingMetrics(**c) for c in summary["corpora"]])
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
