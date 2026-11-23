from __future__ import annotations

import torch

from labfill.checker import check_protocols_batched
from labfill.lablean_io import (
    ERR_TO_LABLEAN_TAG,
    batch_to_lablean_ast,
    err_code_to_lablean,
    export_lablean_module,
    from_lablean_ast,
    lablean_error_to_err_code,
    to_lablean_ast,
)
from labfill.plate import make_serial_dilution_plate_batch
from labfill.protocol import (
    ERR_CONTAMINATED,
    ERR_FROZEN,
    ERR_NO_TIP,
    ERR_OK,
    ERR_OVERFLOW,
    ERR_SEALED,
    ERR_UNBALANCED,
    OP_CENTRIFUGE,
    OP_DROP_TIP,
    OP_MIX,
    OP_PICK_TIP,
    OP_SEAL,
    OP_TRANSFER,
    PIP_P20,
    PIP_P200,
    LabStateBatch,
    ProtocolBatch,
    make_pcr_protocol,
    ul,
)


def test_error_code_bidirectional_mapping() -> None:
    for code, tag in ERR_TO_LABLEAN_TAG.items():
        assert err_code_to_lablean(code) == tag
        assert lablean_error_to_err_code(tag) == code

    assert lablean_error_to_err_code("OK") == ERR_OK
    assert lablean_error_to_err_code('REJECTED: LabLean.Error.overflow "P:A1" 190.0uL 20.0uL 200.0uL') == ERR_OVERFLOW
    assert lablean_error_to_err_code('REJECTED: LabLean.Error.notThawed "enzyme"') == ERR_FROZEN
    assert lablean_error_to_err_code('REJECTED: LabLean.Error.contaminated "CAL:A2"') == ERR_CONTAMINATED
    assert lablean_error_to_err_code('REJECTED: LabLean.Error.overfull "X"') == ERR_OVERFLOW


def test_lablean_ast_roundtrip_and_lean_export_conformance() -> None:
    # 1. PCR round-trip conformance (matches YLM/lean/LabLean/Examples/PCR.lean)
    st_pcr, pr_pcr = make_pcr_protocol()
    ast_pcr = to_lablean_ast(st_pcr, pr_pcr, row=0)
    st_rt, pr_rt = from_lablean_ast(ast_pcr, device=st_pcr.device)

    chk_orig = check_protocols_batched(st_pcr, pr_pcr)
    chk_rt = check_protocols_batched(st_rt, pr_rt)
    assert bool(chk_orig.ok.item()) is True
    assert bool(chk_rt.ok.item()) is True
    assert torch.equal(chk_orig.final_state.volume, chk_rt.final_state.volume)

    lean_src = export_lablean_module(st_pcr, pr_pcr, namespace="PCRExport")
    assert "import LabLean.Checker" in lean_src
    assert "def init : LabState :=" in lean_src
    assert ".pickTip" in lean_src and ".transfer P200" in lean_src
    assert "#eval report init steps" in lean_src

    # 2. SerialDilution round-trip conformance (matches YLM/lean/LabLean/Examples/SerialDilution.lean)
    st_dil, pr_dil = make_serial_dilution_plate_batch(batch_size=4, num_rows_active=1, seed=0)
    asts_dil = batch_to_lablean_ast(st_dil, pr_dil)
    st_dil_rt, pr_dil_rt = from_lablean_ast(asts_dil, device=st_dil.device)
    chk_dil = check_protocols_batched(st_dil_rt, pr_dil_rt)
    assert bool(chk_dil.ok.all().item()) is True

    # 3. Broken examples conformance (matches YLM/lean/LabLean/Examples/Broken.lean)
    # 0=PBS(diluent), 1=FLUO, 2=R1, 3=R2, 4=ENZ(frozen), 5=T1(100uL), 6=T2(500uL), 7=A1(190/200uL), 8=A2(0), 9=A3(0)
    vols = [ul(2000), ul(500), ul(500), ul(500), ul(50), ul(100), ul(500), ul(190), 0, 0]
    caps = [ul(5000), ul(1500), ul(1500), ul(1500), ul(1500), ul(1500), ul(1500), ul(200), ul(200), ul(200)]
    frozen = [False, False, False, False, True, False, False, False, False, False]
    rmask = [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 0, 1 << 0, 1 << 0, 0, 0]
    st_brk = LabStateBatch(
        volume=torch.tensor([vols], dtype=torch.int64),
        capacity=torch.tensor([caps], dtype=torch.int64),
        sealed=torch.zeros((1, len(vols)), dtype=torch.bool),
        frozen=torch.tensor([frozen], dtype=torch.bool),
        reagent_mask=torch.tensor([rmask], dtype=torch.int64),
        diluent_mask=1 << 0,
    )

    broken_suite: list[tuple[list[dict[str, int]], str, int]] = [
        # overflow
        ([{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 2, "dst": 7, "vol": ul(20)}], "overflow", ERR_OVERFLOW),
        # unbalanced
        ([{"op": OP_SEAL, "src": 5}, {"op": OP_SEAL, "src": 6}, {"op": OP_CENTRIFUGE, "src": 5, "dst": 6, "rpm": 5000, "duration": 60}], "unbalanced", ERR_UNBALANCED),
        # reusedTip
        ([{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 2, "dst": 8, "vol": ul(50)}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 3, "dst": 8, "vol": ul(50)}], "contaminated", ERR_CONTAMINATED),
        # notThawed
        ([{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 4, "dst": 8, "vol": ul(5)}], "notThawed", ERR_FROZEN),
        # intoSealed
        ([{"op": OP_SEAL, "src": 8}, {"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 2, "dst": 8, "vol": ul(10)}], "sealed", ERR_SEALED),
        # noTip
        ([{"op": OP_TRANSFER, "pipette": PIP_P20, "src": 2, "dst": 8, "vol": ul(10)}], "noTip", ERR_NO_TIP),
        # backToStock
        ([
            {"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 8, "vol": ul(100)}, {"op": OP_DROP_TIP},
            {"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 1, "dst": 8, "vol": ul(100)}, {"op": OP_MIX, "pipette": PIP_P200, "src": 8, "vol": ul(100)},
            {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 1, "dst": 9, "vol": ul(20)}, {"op": OP_DROP_TIP},
        ], "contaminated", ERR_CONTAMINATED),
        # backUpRow
        ([
            {"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 8, "vol": ul(100)}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 9, "vol": ul(100)}, {"op": OP_DROP_TIP},
            {"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 1, "dst": 8, "vol": ul(100)}, {"op": OP_MIX, "pipette": PIP_P200, "src": 8, "vol": ul(100)},
            {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 8, "dst": 9, "vol": ul(100)}, {"op": OP_MIX, "pipette": PIP_P200, "src": 9, "vol": ul(100)},
            {"op": OP_MIX, "pipette": PIP_P200, "src": 8, "vol": ul(100)}, {"op": OP_DROP_TIP},
        ], "contaminated", ERR_CONTAMINATED),
    ]

    pr_brk = ProtocolBatch.from_steps_list([b[0] for b in broken_suite])
    st_brk_b = st_brk.repeat_batch(len(broken_suite))
    asts_brk = batch_to_lablean_ast(st_brk_b, pr_brk)
    st_brk_rt, pr_brk_rt = from_lablean_ast(asts_brk)
    chk_brk = check_protocols_batched(st_brk_rt, pr_brk_rt)

    for idx, (_steps, exp_lean_tag, exp_code) in enumerate(broken_suite):
        got_code = int(chk_brk.err_code[idx].item())
        assert got_code == exp_code
        assert err_code_to_lablean(got_code) == exp_lean_tag
