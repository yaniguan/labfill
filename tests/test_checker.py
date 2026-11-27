from __future__ import annotations

import pytest
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
    MAX_CONTAINER_VOL_TENTHS_UL,
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
    PIP_P1000,
    LabStateBatch,
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
        for fname in ("volume", "capacity", "sealed", "frozen", "reagent_mask", "has_tip", "tip_mask"):
            assert torch.equal(getattr(rb.final_state, fname), getattr(rp.final_state, fname)), fname


def test_initial_state_invariants_rejected_even_on_noop() -> None:
    noop_proto = ProtocolBatch.from_steps_list([[]] * 4)
    st = LabStateBatch(
        volume=torch.tensor(
            [
                [100, -10],   # row 0: negative volume -> ERR_NOT_ENOUGH
                [250, 100],   # row 1: volume > capacity -> ERR_OVERFLOW
                [10, 10],     # row 2: negative capacity -> ERR_BAD_CONTAINER
                [10, 10],     # row 3: negative reagent_mask -> ERR_CONTAMINATED
            ],
            dtype=torch.int64,
        ),
        capacity=torch.tensor(
            [
                [200, 200],
                [200, 200],
                [200, -5],
                [200, 200],
            ],
            dtype=torch.int64,
        ),
        sealed=torch.zeros((4, 2), dtype=torch.bool),
        frozen=torch.zeros((4, 2), dtype=torch.bool),
        reagent_mask=torch.tensor(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [-1, 0],
            ],
            dtype=torch.int64,
        ),
    )
    rb = check_protocols_batched(st, noop_proto)
    rp = check_protocols_python(st, noop_proto)
    expected = [ERR_NOT_ENOUGH, ERR_OVERFLOW, ERR_BAD_CONTAINER, ERR_CONTAMINATED]
    assert rb.err_code.tolist() == expected
    assert rp.err_code.tolist() == expected
    assert rb.first_err_step.tolist() == [0, 0, 0, 0]
    assert rp.first_err_step.tolist() == [0, 0, 0, 0]
    assert not bool(rb.ok.any().item())


def test_empty_batch_zero_containers_and_mismatch_guards() -> None:
    st_empty = LabStateBatch(
        volume=torch.zeros((0, 4), dtype=torch.int64),
        capacity=torch.full((0, 4), 100, dtype=torch.int64),
        sealed=torch.zeros((0, 4), dtype=torch.bool),
        frozen=torch.zeros((0, 4), dtype=torch.bool),
    )
    pr_empty = ProtocolBatch.from_steps_list([])
    rb = check_protocols_batched(st_empty, pr_empty)
    rp = check_protocols_python(st_empty, pr_empty)
    assert rb.ok.shape == (0,)
    assert rp.ok.shape == (0,)

    st_zero_c = LabStateBatch(
        volume=torch.zeros((2, 0), dtype=torch.int64),
        capacity=torch.zeros((2, 0), dtype=torch.int64),
        sealed=torch.zeros((2, 0), dtype=torch.bool),
        frozen=torch.zeros((2, 0), dtype=torch.bool),
    )
    pr_two = ProtocolBatch.from_steps_list([[], []])
    with pytest.raises(ValueError, match="at least 1 container"):
        check_protocols_batched(st_zero_c, pr_two)
    with pytest.raises(ValueError, match="at least 1 container"):
        check_protocols_python(st_zero_c, pr_two)

    st_one = _base_broken_state()
    with pytest.raises(ValueError, match="Batch size mismatch"):
        check_protocols_batched(st_one, pr_two)


def test_broken_back_up_row_serial_dilution_concentration_guard() -> None:
    # Mirrors YLM/lean/LabLean/Examples/Broken.lean `backUpRow`:
    # 0=PBS (diluent bit 0), 1=FLUO stock (bit 1), 2=col1, 3=col2, 4=col3
    st = LabStateBatch(
        volume=torch.tensor([[ul(2000), ul(500), 0, 0, 0]], dtype=torch.int64),
        capacity=torch.tensor([[ul(5000), ul(1500), ul(300), ul(300), ul(300)]], dtype=torch.int64),
        sealed=torch.zeros((1, 5), dtype=torch.bool),
        frozen=torch.zeros((1, 5), dtype=torch.bool),
        reagent_mask=torch.tensor([[1 << 0, 1 << 1, 0, 0, 0]], dtype=torch.int64),
        diluent_mask=1 << 0,
    )
    setup_steps = [
        # fillPBS into col2 (3) and col3 (4)
        {"op": OP_PICK_TIP},
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 3, "vol": ul(100)},
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 4, "vol": ul(100)},
        {"op": OP_DROP_TIP},
        # fillStock into col1 (2)
        {"op": OP_PICK_TIP},
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 1, "dst": 2, "vol": ul(200)},
        {"op": OP_DROP_TIP},
    ]
    forward_steps = setup_steps + [
        {"op": OP_PICK_TIP},
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 2, "dst": 3, "vol": ul(100)},
        {"op": OP_MIX, "pipette": PIP_P200, "src": 3, "vol": ul(100)},
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 3, "dst": 4, "vol": ul(100)},
        {"op": OP_MIX, "pipette": PIP_P200, "src": 4, "vol": ul(100)},
        {"op": OP_DROP_TIP},
    ]
    # Re-entering col2 (1/2 stock) with tip that touched col3 (1/4 stock) must fail!
    back_up_row_steps = forward_steps[:-1] + [
        {"op": OP_MIX, "pipette": PIP_P200, "src": 3, "vol": ul(100)},
        {"op": OP_DROP_TIP},
    ]

    protos = ProtocolBatch.from_steps_list([forward_steps, back_up_row_steps])
    st2 = st.repeat_batch(2)
    rb = check_protocols_batched(st2, protos)
    rp = check_protocols_python(st2, protos)
    assert rb.err_code.tolist() == [ERR_OK, ERR_CONTAMINATED]
    assert rp.err_code.tolist() == [ERR_OK, ERR_CONTAMINATED]


def test_incubate_thaw_and_high_temp_seal_preconditions() -> None:
    st = _base_broken_state()  # container 2 is frozen, container 6 is thawed & unsealed
    cases = [
        # Incubating a frozen container fails with ERR_FROZEN
        ([{"op": OP_INCUBATE, "src": 2, "temp": 37, "duration": 30}], ERR_FROZEN),
        # Incubating >= 80 C without sealing fails with ERR_NOT_SEALED
        ([{"op": OP_INCUBATE, "src": 6, "temp": 95, "duration": 30}], ERR_NOT_SEALED),
        # Incubating >= 80 C after sealing succeeds
        ([{"op": OP_SEAL, "src": 6}, {"op": OP_INCUBATE, "src": 6, "temp": 95, "duration": 30}], ERR_OK),
        # Incubating < 80 C unsealed succeeds
        ([{"op": OP_INCUBATE, "src": 6, "temp": 65, "duration": 30}], ERR_OK),
        # Redundant OP_THAW on already-thawed container 0 fails with ERR_FROZEN
        ([{"op": OP_THAW, "src": 0}], ERR_FROZEN),
    ]
    protos = ProtocolBatch.from_steps_list([c[0] for c in cases])
    st_b = st.repeat_batch(len(cases))
    rb = check_protocols_batched(st_b, protos)
    rp = check_protocols_python(st_b, protos)
    expected = [c[1] for c in cases]
    assert rb.err_code.tolist() == expected
    assert rp.err_code.tolist() == expected


def test_large_reservoir_back_transfer_no_int64_overflow() -> None:
    """Issue #37: 195 mL + 150 mL reproducer, 12,000 random 1..300 mL pairs, and 10,000 mL carboys."""
    repro_steps = [
        {"op": OP_PICK_TIP},
        {"op": OP_TRANSFER, "pipette": PIP_P1000, "src": 0, "dst": 1, "vol": ul(100.0)},
        {"op": OP_TRANSFER, "pipette": PIP_P1000, "src": 0, "dst": 1, "vol": ul(100.0)},
        {"op": OP_DROP_TIP},
    ]

    # 1. Exact Issue #37 reproducers: 195 mL + 150 mL, smallest overflow boundary, and 10,000 mL limit
    fixed_pairs = [
        (ul(195_000.0), ul(150_000.0)),
        (2_366_227, 389_266),  # 236,622.7 uL + 38,926.6 uL boundary from Issue #37
        (ul(5_000.0), ul(5_000.0)),
        (MAX_CONTAINER_VOL_TENTHS_UL, MAX_CONTAINER_VOL_TENTHS_UL - ul(500.0)),
    ]
    st_fixed = LabStateBatch(
        volume=torch.tensor([[a, b] for a, b in fixed_pairs], dtype=torch.int64),
        capacity=torch.full((len(fixed_pairs), 2), MAX_CONTAINER_VOL_TENTHS_UL, dtype=torch.int64),
        sealed=torch.zeros((len(fixed_pairs), 2), dtype=torch.bool),
        frozen=torch.zeros((len(fixed_pairs), 2), dtype=torch.bool),
        reagent_mask=torch.tensor([[0b11, 0b01]] * len(fixed_pairs), dtype=torch.int64),
        diluent_mask=1,
    )
    pr_fixed = ProtocolBatch.from_steps_list([repro_steps] * len(fixed_pairs))
    rb_f = check_protocols_batched(st_fixed, pr_fixed)
    rp_f = check_protocols_python(st_fixed, pr_fixed)
    assert rb_f.err_code.tolist() == [ERR_CONTAMINATED] * len(fixed_pairs)
    assert rp_f.err_code.tolist() == [ERR_CONTAMINATED] * len(fixed_pairs)
    assert rb_f.first_err_step.tolist() == [2] * len(fixed_pairs)
    assert rp_f.first_err_step.tolist() == [2] * len(fixed_pairs)
    for fname in ("volume", "capacity", "sealed", "frozen", "reagent_mask", "has_tip", "tip_mask"):
        assert torch.equal(getattr(rb_f.final_state, fname), getattr(rp_f.final_state, fname)), fname

    # 2. Differential test on 12,000 random large-volume pairs up to 300 mL (including clean + back-transfer mixes)
    n_rand = 12_000
    g = torch.Generator(device="cpu").manual_seed(37037)
    vol_a = torch.randint(ul(1_000.0), ul(299_000.0), (n_rand,), generator=g, dtype=torch.int64)
    vol_b = torch.randint(0, ul(299_000.0), (n_rand,), generator=g, dtype=torch.int64)
    # Force ~15% clean empty-well B=0 rows and ~15% near-tolerance diluted rows to exercise both branches
    vol_b = torch.where(torch.arange(n_rand) % 7 == 0, torch.zeros_like(vol_b), vol_b)
    cap_ab = torch.full((n_rand, 3), ul(300_000.0), dtype=torch.int64)
    vol_abc = torch.stack([vol_a, vol_b, torch.zeros_like(vol_a)], dim=1)
    rmask = torch.zeros((n_rand, 3), dtype=torch.int64)
    rmask[:, 0] = 0b11
    rmask[:, 1] = torch.where(vol_b > 0, torch.ones_like(vol_b), torch.zeros_like(vol_b))

    st_rand = LabStateBatch(
        volume=vol_abc,
        capacity=cap_ab,
        sealed=torch.zeros((n_rand, 3), dtype=torch.bool),
        frozen=torch.zeros((n_rand, 3), dtype=torch.bool),
        reagent_mask=rmask,
        diluent_mask=1,
    )
    pr_rand = ProtocolBatch.from_steps_list([repro_steps] * n_rand)
    rb_r = check_protocols_batched(st_rand, pr_rand)
    rp_r = check_protocols_python(st_rand, pr_rand)
    assert torch.equal(rb_r.ok, rp_r.ok)
    assert torch.equal(rb_r.err_code, rp_r.err_code)
    assert torch.equal(rb_r.first_err_step, rp_r.first_err_step)
    for fname in ("volume", "capacity", "sealed", "frozen", "reagent_mask", "has_tip", "tip_mask"):
        assert torch.equal(getattr(rb_r.final_state, fname), getattr(rp_r.final_state, fname)), fname
    # Sanity check: majority have vol_b > 0 (contaminated) and 1/7 have vol_b == 0 (ok)
    assert int((rb_r.err_code == ERR_CONTAMINATED).sum().item()) > 9_000
    assert int((rb_r.err_code == ERR_OK).sum().item()) > 1_500


def test_max_container_volume_guard_raises_value_error() -> None:
    with pytest.raises(ValueError, match="MAX_CONTAINER_VOL_TENTHS_UL"):
        LabStateBatch(
            volume=torch.tensor([[100]], dtype=torch.int64),
            capacity=torch.tensor([[MAX_CONTAINER_VOL_TENTHS_UL + 1]], dtype=torch.int64),
            sealed=torch.zeros((1, 1), dtype=torch.bool),
            frozen=torch.zeros((1, 1), dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="MAX_CONTAINER_VOL_TENTHS_UL"):
        LabStateBatch(
            volume=torch.tensor([[MAX_CONTAINER_VOL_TENTHS_UL + 1]], dtype=torch.int64),
            capacity=torch.tensor([[MAX_CONTAINER_VOL_TENTHS_UL]], dtype=torch.int64),
            sealed=torch.zeros((1, 1), dtype=torch.bool),
            frozen=torch.zeros((1, 1), dtype=torch.bool),
        )
