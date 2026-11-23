from __future__ import annotations

from collections import Counter

import torch

from labfill.checker import check_protocols_batched
from labfill.protocol import (
    ERR_NO_TIP,
    LabStateBatch,
    OP_DROP_TIP,
    OP_MIX,
    OP_PICK_TIP,
    OP_TRANSFER,
    PIP_P20,
    PIP_P1000,
    ProtocolBatch,
    make_random_valid_protocols,
)
from labfill.twins import (
    EXPECTED_PRIMARY_CODES,
    TWIN_ERROR_TYPES,
    batched_repair,
    generate_check_keep,
    make_error_twin,
    measure_twin_recall,
    repair_edit_kind,
)


def _liquid_steps(pr: ProtocolBatch, row: int, with_vol: bool = True) -> Counter:
    """Multiset of (op, src, dst[, vol]) over active transfer and mix steps of one protocol."""
    out: Counter = Counter()
    for t in range(pr.max_steps):
        op = int(pr.op[row, t])
        if bool(pr.pad_mask[row, t]) or op not in (OP_TRANSFER, OP_MIX):
            continue
        dst = int(pr.dst[row, t]) if op == OP_TRANSFER else -1
        key = (op, int(pr.src[row, t]), dst)
        out[key + (int(pr.vol[row, t]),) if with_vol else key] += 1
    return out


def test_seeded_error_twin_recall() -> None:
    rep = measure_twin_recall(batch_size=128, seed=10)
    assert rep.false_positive_rate_on_valid == 0.0
    for et in TWIN_ERROR_TYPES:
        assert rep.per_error_recall[et] == 1.0
    # Every twin with a known primary error must fail with exactly that code
    assert set(rep.per_error_primary_code_match) == set(EXPECTED_PRIMARY_CODES)
    for et in EXPECTED_PRIMARY_CODES:
        assert rep.per_error_primary_code_match[et] == 1.0


def test_swapped_steps_twin_fails_with_no_tip() -> None:
    st, pr = make_random_valid_protocols(64, seed=3)
    st_b, pr_b = make_error_twin(st, pr, "swapped_steps", seed=4)
    res = check_protocols_batched(st_b, pr_b)
    # The transfer now runs before its pick_tip
    assert bool((res.err_code == ERR_NO_TIP).all().item())


def test_generate_check_keep_filters_strictly() -> None:
    res = generate_check_keep(num_candidates=512, mutation_rate=0.3, seed=21)
    assert 0 < res.num_kept < res.num_candidates
    assert res.candidates_per_sec > 0.0
    recheck = check_protocols_batched(res.kept_state, res.kept_protocols)
    assert bool(recheck.ok.all().item()) is True


def test_batched_repair_produces_verified_protocols() -> None:
    st, pr = make_random_valid_protocols(32, seed=5)
    for et in TWIN_ERROR_TYPES:
        st_b, pr_b = make_error_twin(st, pr, et, seed=6)
        assert not bool(check_protocols_batched(st_b, pr_b).ok.any().item())

        rr = batched_repair(st_b, pr_b, num_edits=24)
        assert rr.repair_rate >= 0.95, et
        assert sum(rr.edit_type_counts.values()) == rr.num_repaired
        idx = rr.repaired_mask.nonzero(as_tuple=False).squeeze(1)
        verified = check_protocols_batched(st_b.index_select(idx), rr.repaired_protocols.index_select(idx))
        assert bool(verified.ok.all().item()) is True

        # Counted repairs keep every transfer and mix step: same (op, src, dst) multiset,
        # and the same volumes unless the winning edit was a volume clamp
        for row in idx.tolist():
            kind = repair_edit_kind(int(rr.chosen_edit[row]))
            assert kind != "redirect_dst", et
            before = _liquid_steps(pr_b, row, with_vol=False)
            after = _liquid_steps(rr.repaired_protocols, row, with_vol=False)
            assert before == after, (et, row, kind)
            if kind != "clamp_volume":
                assert _liquid_steps(pr_b, row) == _liquid_steps(rr.repaired_protocols, row), (et, row)
            else:
                removed = _liquid_steps(pr_b, row) - _liquid_steps(rr.repaired_protocols, row)
                added = _liquid_steps(rr.repaired_protocols, row) - _liquid_steps(pr_b, row)
                assert sum(removed.values()) == sum(added.values()) == 1
                (old,), (new,) = removed.elements(), added.elements()
                assert old[:3] == new[:3] and 0 < new[3] < old[3]


def test_overflow_repair_keeps_largest_volume_that_fits() -> None:
    st, pr = make_random_valid_protocols(32, seed=5)
    st_b, pr_b = make_error_twin(st, pr, "overflow", seed=6)
    rr = batched_repair(st_b, pr_b, num_edits=24)
    assert rr.edit_type_counts == {"clamp_volume": 32}

    # The first transfer goes into an empty well; the repair never exceeds its capacity
    first_xfer = (pr_b.op == OP_TRANSFER).to(torch.int64).argmax(dim=1)
    rows = torch.arange(32)
    cap = st_b.capacity[rows, pr_b.dst[rows, first_xfer]]
    new_vol = rr.repaired_protocols.vol[rows, first_xfer]
    assert bool(((new_vol > 0) & (new_vol <= cap) & (new_vol < pr_b.vol[rows, first_xfer])).all())
    assert bool((new_vol == cap).any())

    # Whenever less than the full capacity was kept, the full-capacity version fails later on
    smaller = (new_vol < cap).nonzero(as_tuple=False).squeeze(1)
    assert smaller.numel() > 0
    full = rr.repaired_protocols.index_select(smaller).clone()
    full.vol[torch.arange(smaller.numel()), first_xfer[smaller]] = cap[smaller]
    full.pipette[torch.arange(smaller.numel()), first_xfer[smaller]] = PIP_P1000
    assert not bool(check_protocols_batched(st_b.index_select(smaller), full).ok.any())


def test_deleting_a_transfer_is_not_counted_as_repair() -> None:
    # src == dst: only deleting the transfer or redirecting it would make the checker pass
    c = 8
    st = LabStateBatch(
        volume=torch.tensor([[800] * 3 + [0] * (c - 3)], dtype=torch.int64),
        capacity=torch.full((1, c), 3000, dtype=torch.int64),
        sealed=torch.zeros((1, c), dtype=torch.bool),
        frozen=torch.zeros((1, c), dtype=torch.bool),
        reagent_mask=torch.tensor([[1, 2, 4] + [0] * (c - 3)], dtype=torch.int64),
    )
    pr = ProtocolBatch.from_steps_list(
        [
            [
                {"op": OP_PICK_TIP},
                {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 0, "vol": 100},
                {"op": OP_DROP_TIP},
            ]
        ]
    )
    rr = batched_repair(st, pr, num_edits=24)
    assert rr.num_repaired == 0
    assert rr.repair_rate == 0.0
    assert int(rr.chosen_edit[0]) == -1
    assert rr.edit_type_counts == {}
    assert rr.num_fixed_only_by_intent_change == 1
