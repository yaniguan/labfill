"""Bidirectional exporter/importer and error-code bridge between labfill and YLM LabLean."""

from __future__ import annotations

import re
from typing import Any, Sequence

import torch

from labfill.protocol import (
    AMT_SCALE,
    ERR_BAD_CONTAINER,
    ERR_BAD_DURATION,
    ERR_BAD_OP,
    ERR_BAD_PIPETTE,
    ERR_BAD_RPM,
    ERR_BAD_TEMP,
    ERR_CONTAMINATED,
    ERR_FROZEN,
    ERR_NO_TIP,
    ERR_NOT_ENOUGH,
    ERR_NOT_SEALED,
    ERR_OK,
    ERR_OVERFLOW,
    ERR_SAME_CONTAINER,
    ERR_SEALED,
    ERR_TIP_ALREADY_ON,
    ERR_UNBALANCED,
    OP_CENTRIFUGE,
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_MIX,
    OP_NAMES,
    OP_NOOP,
    OP_PICK_TIP,
    OP_SEAL,
    OP_THAW,
    OP_TRANSFER,
    OP_UNSEAL,
    PIP_NONE,
    PIPETTE_NAMES,
    LabStateBatch,
    ProtocolBatch,
    default_device,
)

# Canonical mapping from labfill ERR_* codes to YLM LabLean.Error constructor tags
ERR_TO_LABLEAN_TAG: dict[int, str] = {
    ERR_OK: "ok",
    ERR_BAD_OP: "badOp",
    ERR_BAD_CONTAINER: "unknown",
    ERR_NO_TIP: "noTip",
    ERR_TIP_ALREADY_ON: "tipAlreadyOn",
    ERR_BAD_PIPETTE: "outOfRange",
    ERR_SAME_CONTAINER: "sameContainer",
    ERR_SEALED: "sealed",
    ERR_NOT_SEALED: "notSealed",
    ERR_FROZEN: "notThawed",
    ERR_CONTAMINATED: "contaminated",
    ERR_NOT_ENOUGH: "notEnough",
    ERR_OVERFLOW: "overflow",
    ERR_BAD_TEMP: "badTemp",
    ERR_BAD_RPM: "tooFast",
    ERR_UNBALANCED: "unbalanced",
    ERR_BAD_DURATION: "badDuration",
}

# Reverse mapping covering all LabLean.Error constructors (including WellFormed start-state errors)
LABLEAN_TAG_TO_ERR: dict[str, int] = {
    "ok": ERR_OK,
    "OK": ERR_OK,
    "badOp": ERR_BAD_OP,
    "unknown": ERR_BAD_CONTAINER,
    "notTubeOrPlate": ERR_BAD_CONTAINER,
    "badRotor": ERR_BAD_CONTAINER,
    "twiceInRotor": ERR_SAME_CONTAINER,
    "duplicateIds": ERR_BAD_CONTAINER,
    "idIsPlate": ERR_BAD_CONTAINER,
    "unknownPlate": ERR_BAD_CONTAINER,
    "noTip": ERR_NO_TIP,
    "tipAlreadyOn": ERR_TIP_ALREADY_ON,
    "outOfRange": ERR_BAD_PIPETTE,
    "sameContainer": ERR_SAME_CONTAINER,
    "sealed": ERR_SEALED,
    "alreadySealed": ERR_SEALED,
    "notSealed": ERR_NOT_SEALED,
    "notThawed": ERR_FROZEN,
    "contaminated": ERR_CONTAMINATED,
    "duplicateReagents": ERR_CONTAMINATED,
    "undeclaredReagent": ERR_CONTAMINATED,
    "badContents": ERR_CONTAMINATED,
    "notEnough": ERR_NOT_ENOUGH,
    "overflow": ERR_OVERFLOW,
    "overfull": ERR_OVERFLOW,
    "badTemp": ERR_BAD_TEMP,
    "tooFast": ERR_BAD_RPM,
    "unbalanced": ERR_UNBALANCED,
    "badDuration": ERR_BAD_DURATION,
}

_NAME_TO_OP: dict[str, int] = {name: code for code, name in OP_NAMES.items()}
_NAME_TO_PIP: dict[str, int] = {name: code for code, name in PIPETTE_NAMES.items()}


def err_code_to_lablean(err_code: int) -> str:
    """Map a labfill `ERR_*` integer code to its LabLean error tag."""
    if err_code not in ERR_TO_LABLEAN_TAG:
        raise ValueError(f"Unknown labfill error code: {err_code}")
    return ERR_TO_LABLEAN_TAG[err_code]


def lablean_error_to_err_code(lean_report_or_tag: str) -> int:
    """Parse a LabLean `report` output (e.g. `'REJECTED: LabLean.Error.overflow ...'`) or tag into `ERR_*`."""
    text = lean_report_or_tag.strip()
    if text in LABLEAN_TAG_TO_ERR:
        return LABLEAN_TAG_TO_ERR[text]
    if text.upper() == "OK":
        return ERR_OK
    if text.startswith("REJECTED:"):
        text = text.split("REJECTED:", 1)[1].strip()
    m = re.search(r"(?:LabLean\.Error\.|\.)?([A-Za-z][A-Za-z0-9_]*)", text)
    if m and m.group(1) in LABLEAN_TAG_TO_ERR:
        return LABLEAN_TAG_TO_ERR[m.group(1)]
    raise ValueError(f"Unrecognized LabLean error string: {lean_report_or_tag!r}")


def _bits_of(mask: int) -> list[int]:
    bits: list[int] = []
    b = 0
    val = mask
    while val > 0:
        if val & 1:
            bits.append(b)
        val >>= 1
        b += 1
    return bits


def to_lablean_ast(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
    row: int = 0,
    container_names: Sequence[str] | None = None,
    reagent_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Export one `(LabStateBatch, ProtocolBatch)` row into a LabLean JSON AST."""
    if not (0 <= row < init_state.batch_size):
        raise ValueError(f"row={row} out of range 0..{init_state.batch_size - 1}")
    c = init_state.num_containers
    c_names = list(container_names) if container_names is not None else [f"C{i}" for i in range(c)]
    if len(c_names) != c:
        raise ValueError(f"container_names length {len(c_names)} != num_containers {c}")
    r_names = reagent_names or {}

    vols = init_state.volume[row].to("cpu").tolist()
    caps = init_state.capacity[row].to("cpu").tolist()
    sealed = init_state.sealed[row].to("cpu").tolist()
    frozen = init_state.frozen[row].to("cpu").tolist()
    rmasks = init_state.reagent_mask[row].to("cpu").tolist()
    dil_mask = int(init_state.diluent_mask)

    # Collect active reagent bits and infer initial frozen states per reagent
    used_bits: set[int] = set()
    frozen_bits: set[int] = set()
    for i in range(c):
        bs = _bits_of(max(int(rmasks[i]), 0))
        used_bits.update(bs)
        if frozen[i]:
            frozen_bits.update(bs)
    used_bits.update(_bits_of(max(int(init_state.tip_mask[row].item()), 0)))

    reagents_ast: list[dict[str, Any]] = []
    for bit in sorted(used_bits):
        reagents_ast.append(
            {
                "bit": bit,
                "name": r_names.get(bit, f"r{bit}"),
                "state": "frozen" if bit in frozen_bits else "roomTemp",
                "diluent": bool(dil_mask & (1 << bit)),
            }
        )

    containers_ast: list[dict[str, Any]] = []
    for i in range(c):
        containers_ast.append(
            {
                "index": i,
                "id": c_names[i],
                "vol_tenths_ul": int(vols[i]),
                "cap_tenths_ul": int(caps[i]),
                "sealed": bool(sealed[i]),
                "frozen": bool(frozen[i]),
                "reagent_mask": int(rmasks[i]),
            }
        )

    steps_ast: list[dict[str, Any]] = []
    for t in range(protocols.max_steps):
        if bool(protocols.pad_mask[row, t].item()):
            continue
        op_code = int(protocols.op[row, t].item())
        if op_code == OP_NOOP:
            continue
        s_idx = int(protocols.src[row, t].item())
        d_idx = int(protocols.dst[row, t].item())
        s_id = c_names[s_idx] if 0 <= s_idx < c else f"INVALID_{s_idx}"
        d_id = c_names[d_idx] if 0 <= d_idx < c else f"INVALID_{d_idx}"
        steps_ast.append(
            {
                "op": OP_NAMES.get(op_code, f"bad_op_{op_code}"),
                "op_code": op_code,
                "src": s_id,
                "src_index": s_idx,
                "dst": d_id,
                "dst_index": d_idx,
                "vol_tenths_ul": int(protocols.vol[row, t].item()),
                "pipette": PIPETTE_NAMES.get(int(protocols.pipette[row, t].item()), "NONE"),
                "pipette_code": int(protocols.pipette[row, t].item()),
                "temp_c": int(protocols.temp[row, t].item()),
                "duration_s": int(protocols.duration[row, t].item()),
                "rpm": int(protocols.rpm[row, t].item()),
            }
        )

    return {
        "schema": "lablean_ast_v1",
        "reagents": reagents_ast,
        "containers": containers_ast,
        "diluent_mask": dil_mask,
        "has_tip": bool(init_state.has_tip[row].item()),
        "tip_mask": int(init_state.tip_mask[row].item()),
        "steps": steps_ast,
    }


def batch_to_lablean_ast(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
    container_names: Sequence[str] | None = None,
    reagent_names: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Export all B rows of `(LabStateBatch, ProtocolBatch)` into LabLean JSON AST dicts."""
    return [
        to_lablean_ast(
            init_state,
            protocols,
            row=i,
            container_names=container_names,
            reagent_names=reagent_names,
        )
        for i in range(init_state.batch_size)
    ]


def from_lablean_ast(
    ast_or_list: dict[str, Any] | Sequence[dict[str, Any]],
    device: str | torch.device | None = None,
) -> tuple[LabStateBatch, ProtocolBatch]:
    """Import one or more LabLean JSON AST dicts back into `(LabStateBatch, ProtocolBatch)`."""
    items = [ast_or_list] if isinstance(ast_or_list, dict) else list(ast_or_list)
    if not items:
        raise ValueError("ast_or_list must contain at least one AST object")
    dev = default_device(device)

    c_counts = {len(it["containers"]) for it in items}
    if len(c_counts) > 1:
        raise ValueError(f"All AST rows must have the same container count, got {sorted(c_counts)}")
    dil_masks = {int(it.get("diluent_mask", 0)) for it in items}
    if len(dil_masks) > 1:
        raise ValueError(f"All AST rows must share the same diluent_mask, got {sorted(dil_masks)}")

    vols: list[list[int]] = []
    caps: list[list[int]] = []
    sealed: list[list[bool]] = []
    frozen: list[list[bool]] = []
    rmasks: list[list[int]] = []
    has_tips: list[bool] = []
    tip_masks: list[int] = []
    proto_steps_list: list[list[dict[str, int]]] = []

    for it in items:
        cs = sorted(it["containers"], key=lambda x: int(x["index"]))
        id_to_idx = {str(c_obj["id"]): int(c_obj["index"]) for c_obj in cs}
        vols.append([int(c_obj["vol_tenths_ul"]) for c_obj in cs])
        caps.append([int(c_obj["cap_tenths_ul"]) for c_obj in cs])
        sealed.append([bool(c_obj["sealed"]) for c_obj in cs])
        frozen.append([bool(c_obj["frozen"]) for c_obj in cs])
        rmasks.append([int(c_obj["reagent_mask"]) for c_obj in cs])
        has_tips.append(bool(it.get("has_tip", False)))
        tip_masks.append(int(it.get("tip_mask", 0)))

        row_steps: list[dict[str, int]] = []
        for st in it["steps"]:
            if "op_code" in st:
                op_code = int(st["op_code"])
            else:
                op_code = _NAME_TO_OP[str(st["op"])]
            if "src_index" in st:
                s_idx = int(st["src_index"])
            else:
                s_idx = id_to_idx.get(str(st.get("src", "")), -1)
            if "dst_index" in st:
                d_idx = int(st["dst_index"])
            else:
                d_idx = id_to_idx.get(str(st.get("dst", "")), -1)
            if "pipette_code" in st:
                pip_code = int(st["pipette_code"])
            else:
                pip_code = _NAME_TO_PIP.get(str(st.get("pipette", "NONE")), PIP_NONE)

            row_steps.append(
                {
                    "op": op_code,
                    "src": s_idx,
                    "dst": d_idx,
                    "vol": int(st.get("vol_tenths_ul", 0)),
                    "pipette": pip_code,
                    "temp": int(st.get("temp_c", 0)),
                    "duration": int(st.get("duration_s", 0)),
                    "rpm": int(st.get("rpm", 0)),
                }
            )
        proto_steps_list.append(row_steps)

    state = LabStateBatch(
        volume=torch.tensor(vols, dtype=torch.int64, device=dev),
        capacity=torch.tensor(caps, dtype=torch.int64, device=dev),
        sealed=torch.tensor(sealed, dtype=torch.bool, device=dev),
        frozen=torch.tensor(frozen, dtype=torch.bool, device=dev),
        reagent_mask=torch.tensor(rmasks, dtype=torch.int64, device=dev),
        has_tip=torch.tensor(has_tips, dtype=torch.bool, device=dev),
        tip_mask=torch.tensor(tip_masks, dtype=torch.int64, device=dev),
        diluent_mask=next(iter(dil_masks)),
    )
    proto = ProtocolBatch.from_steps_list(proto_steps_list, device=dev)
    return state, proto


def export_lablean_module(
    init_state: LabStateBatch,
    protocols: ProtocolBatch,
    row: int = 0,
    namespace: str = "LabfillExport",
    container_names: Sequence[str] | None = None,
) -> str:
    """Emit a self-contained Lean 4 source module (`LabState` + `List Step`) for `YLM/LabLean`."""
    ast = to_lablean_ast(init_state, protocols, row=row, container_names=container_names)

    reagent_lines: list[str] = []
    bit_to_rname: dict[int, str] = {}
    for r in ast["reagents"]:
        bit = int(r["bit"])
        rname = str(r["name"])
        bit_to_rname[bit] = rname
        dil_str = ", diluent := true" if r["diluent"] else ""
        st_str = ".frozen" if r["state"] == "frozen" else ".roomTemp"
        reagent_lines.append(f'    {{ name := "{rname}", state := {st_str}{dil_str} }}')

    container_lines: list[str] = []
    plate_entries: list[str] = []
    sealed_ids: list[str] = []
    c_reagent_for_thaw: dict[str, str] = {}

    for c_obj in ast["containers"]:
        cid = str(c_obj["id"])
        v_t = int(c_obj["vol_tenths_ul"])
        cap_t = int(c_obj["cap_tenths_ul"])
        rm = int(c_obj["reagent_mask"])
        bs = _bits_of(max(rm, 0))
        if bs:
            c_reagent_for_thaw[cid] = bit_to_rname[bs[0]]
        if v_t > 0 and bs:
            per_bit = (v_t * AMT_SCALE) // len(bs)
            rem = (v_t * AMT_SCALE) - per_bit * len(bs)
            contents_items = [
                f'("{bit_to_rname[b]}", {per_bit + (rem if k == 0 else 0)})'
                for k, b in enumerate(bs)
            ]
            contents_str = "[" + ", ".join(contents_items) + "]"
        else:
            contents_str = "[]"

        # Model sealable containers as 1-well plates so LabLean `.seal` / `.centrifuge` apply directly
        plate_id = f"PL_{cid}"
        plate_entries.append(f'("{plate_id}", 1000)')
        if c_obj["sealed"]:
            sealed_ids.append(f'"{plate_id}"')
        container_lines.append(
            f'    {{ id := "{cid}", plate := some "{plate_id}", cap := ⟨{cap_t}⟩, '
            f'vol := ⟨{v_t}⟩, contents := {contents_str}, tare := 0 }}'
        )

    step_lines: list[str] = []
    for st in ast["steps"]:
        op_code = int(st["op_code"])
        sid = str(st["src"])
        did = str(st["dst"])
        vt = int(st["vol_tenths_ul"])
        pip_name = str(st["pipette"])
        tc = int(st["temp_c"])
        dur = int(st["duration_s"])
        rpm = int(st["rpm"])

        if op_code == OP_PICK_TIP:
            step_lines.append("  .pickTip")
        elif op_code == OP_DROP_TIP:
            step_lines.append("  .dropTip")
        elif op_code == OP_TRANSFER:
            step_lines.append(f'  .transfer {pip_name} "{sid}" "{did}" ⟨{vt}⟩')
        elif op_code == OP_MIX:
            step_lines.append(f'  .mix {pip_name} "{sid}" ⟨{vt}⟩ 3')
        elif op_code == OP_INCUBATE:
            step_lines.append(f'  .incubate "PL_{sid}" ⟨{tc}⟩ (sec {dur})')
        elif op_code == OP_THAW:
            r_target = c_reagent_for_thaw.get(sid, sid)
            step_lines.append(f'  .thaw "{r_target}"')
        elif op_code == OP_SEAL:
            step_lines.append(f'  .seal "PL_{sid}"')
        elif op_code == OP_UNSEAL:
            step_lines.append(f"  -- unseal PL_{sid} (tracked by labfill extension)")
        elif op_code == OP_CENTRIFUGE:
            step_lines.append(f'  .centrifuge [some "PL_{sid}", some "PL_{did}"] {rpm} (sec {dur})')

    reagents_block = "[\n" + ",\n".join(reagent_lines) + "\n  ]" if reagent_lines else "[]"
    plates_block = "[" + ", ".join(plate_entries) + "]"
    sealed_block = "[" + ", ".join(sealed_ids) + "]"
    cs_block = "[\n" + ",\n".join(container_lines) + "\n  ]" if container_lines else "[]"
    steps_clean = [ln for ln in step_lines if not ln.lstrip().startswith("--")]
    steps_block = "[\n" + ",\n".join(steps_clean) + "\n]" if steps_clean else "[]"

    return "\n".join(
        [
            "import LabLean.Checker",
            f"namespace LabLean.{namespace}",
            "",
            "def init : LabState :=",
            f"  {{ reagents := {reagents_block},",
            f"    plates := {plates_block},",
            f"    sealed := {sealed_block},",
            f"    cs := {cs_block} }}",
            "",
            f"def steps : List Step := {steps_block}",
            "",
            "#eval report init steps",
            "",
            f"end LabLean.{namespace}",
            "",
        ]
    )
