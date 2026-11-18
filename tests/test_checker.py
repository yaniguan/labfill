from __future__ import annotations

import torch

from labfill.checker import check_protocols_batched, check_protocols_python
from labfill.protocol import (
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
    LabStateBatch,
    OP_CENTRIFUGE,
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_MIX,
    OP_PICK_TIP,
    OP_SEAL,
    OP_THAW,
    OP_TRANSFER,
    OP_UNSEAL,
    PIP_P20,
    PIP_P200,
    ProtocolBatch,
    make_pcr_protocol,
    make_random_protocols,
    ul,
)


def _base_broken_state() -> LabStateBatch:
    """Mirrors LabLean/Examples/Broken.lean initial containers."""
    # 0=R1(r0), 1=R2(r1), 2=ENZ(r2, frozen), 3=T1(100uL), 4=T2(500uL), 5=A1(190/200uL), 6=A2(0/200uL)
    vols = [ul(500), ul(500), ul(50), ul(100), ul(500), ul(190), 0]
    caps = [ul(1500), ul(1500), ul(1500), ul(1500), ul(1500), ul(200), ul(200)]
    frozen = [False, False, True, False, False, False, False]
    rmask = [1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 3, 1 << 3, 0]
    return LabStateBatch(
        volume=torch.tensor([vols], dtype=torch.int64),
        capacity=torch.tensor([caps], dtype=torch.int64),
        sealed=torch.zeros((1, 7), dtype=torch.bool),
        frozen=torch.tensor([frozen], dtype=torch.bool),
        reagent_mask=torch.tensor([rmask], dtype=torch.int64),
    )


def test_pcr_protocol_valid_and_conserves_volume() -> None:
    st, pr = make_pcr_protocol()
    rb = check_protocols_batched(st, pr)
    rp = check_protocols_python(st, pr)
    assert bool(rb.ok.item()) is True
    assert bool(rp.ok.item()) is True
    assert rb.final_state.volume[0, 8:12].tolist() == [500, 500, 500, 500]
    assert int(st.volume.sum().item()) == int(rb.final_state.volume.sum().item())


def test_lablean_broken_rules_and_tip_contamination() -> None:
    st = _base_broken_state()

    cases = [
        # 1. Overflow: A1 holds 190/200 uL, add 20 uL
        (
            [{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 5, "vol": ul(20)}],
            ERR_OVERFLOW,
        ),
        # 2. Unbalanced centrifuge: T1(100 uL) vs T2(500 uL), both sealed first
        (
            [
                {"op": OP_SEAL, "src": 3},
                {"op": OP_SEAL, "src": 4},
                {"op": OP_CENTRIFUGE, "src": 3, "dst": 4, "rpm": 5000, "duration": 60},
            ],
            ERR_UNBALANCED,
        ),
        # 3. Reused tip across reagents R1 -> A2 then R2 -> A2
        (
            [
                {"op": OP_PICK_TIP},
                {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 6, "vol": ul(50)},
                {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 1, "dst": 6, "vol": ul(50)},
            ],
            ERR_CONTAMINATED,
        ),
        # 4. Frozen enzyme without thaw
        (
            [{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 2, "dst": 6, "vol": ul(5)}],
            ERR_FROZEN,
        ),
        # 5. Pipetting into sealed container
        (
            [
                {"op": OP_SEAL, "src": 6},
                {"op": OP_PICK_TIP},
                {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 6, "vol": ul(10)},
            ],
            ERR_SEALED,
        ),
        # 6. No tip
        (
            [{"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 6, "vol": ul(10)}],
            ERR_NO_TIP,
        ),
        # Additional rules
        ([{"op": OP_PICK_TIP}, {"op": OP_PICK_TIP}], ERR_TIP_ALREADY_ON),
        ([{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 6, "vol": ul(100)}], ERR_BAD_PIPETTE),
        ([{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 0, "vol": ul(10)}], ERR_SAME_CONTAINER),
        ([{"op": OP_THAW, "src": 2}, {"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 2, "dst": 6, "vol": ul(100)}], ERR_NOT_ENOUGH),
        ([{"op": OP_INCUBATE, "src": 6, "temp": 105, "duration": 30}], ERR_BAD_TEMP),
        ([{"op": OP_INCUBATE, "src": 6, "temp": 37, "duration": 0}], ERR_BAD_DURATION),
        ([{"op": OP_UNSEAL, "src": 6}], ERR_NOT_SEALED),
        ([{"op": OP_SEAL, "src": 3}, {"op": OP_SEAL, "src": 4}, {"op": OP_CENTRIFUGE, "src": 3, "dst": 4, "rpm": 20000, "duration": 30}], ERR_BAD_RPM),
        ([{"op": 99}], ERR_BAD_OP),
        ([{"op": OP_THAW, "src": 99}], ERR_BAD_CONTAINER),
    ]

    protos = ProtocolBatch.from_steps_list([c[0] for c in cases], device=st.device)
    st_b = st.repeat_batch(len(cases))
    rb = check_protocols_batched(st_b, protos)
    rp = check_protocols_python(st_b, protos)
    expected = [c[1] for c in cases]
    assert rb.err_code.tolist() == expected
    assert rp.err_code.tolist() == expected

    # One source -> many empty destinations with a single tip IS allowed (matches LabLean!)
    multi_dispense = ProtocolBatch.from_steps_list(
        [
            [
                {"op": OP_PICK_TIP},
                {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 6, "vol": ul(10)},
                {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 6, "vol": ul(10)},
                {"op": OP_MIX, "pipette": PIP_P20, "src": 0, "vol": ul(10)},
                {"op": OP_DROP_TIP},
            ]
        ],
        device=st.device,
    )
    res_ok = check_protocols_batched(st, multi_dispense)
    assert int(res_ok.err_code.item()) == ERR_OK


def test_python_vs_batched_equivalence_on_thousands_of_random_protocols() -> None:
    for seed, err_p in ((1, 0.0), (2, 0.4), (3, 0.85)):
        st, pr = make_random_protocols(
            batch_size=1500, max_steps=26, num_containers=8, error_prob=err_p, seed=seed
        )
        rb = check_protocols_batched(st, pr)
        rp = check_protocols_python(st, pr)
        assert torch.equal(rb.ok, rp.ok)
        assert torch.equal(rb.err_code, rp.err_code)
        assert torch.equal(rb.first_err_step, rp.first_err_step)
        assert torch.equal(rb.final_state.volume, rp.final_state.volume)
