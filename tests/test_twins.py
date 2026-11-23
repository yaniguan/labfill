from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch

from labfill.checker import check_protocols_batched
from labfill.plate import make_serial_dilution_plate_batch
from labfill.protocol import (
    ERR_NO_TIP,
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_MIX,
    OP_PICK_TIP,
    OP_TRANSFER,
    PIP_P20,
    PIP_P200,
    PIP_P1000,
    LabStateBatch,
    ProtocolBatch,
    make_pcr_protocol,
    make_random_valid_protocols,
    ul,
)
from labfill.twins import (
    EXPECTED_PRIMARY_CODES,
    TWIN_ERROR_TYPES,
    _insert_step_before,
    batched_repair,
    from_preference_jsonl,
    generate_check_keep,
    make_error_twin,
    measure_twin_recall,
    repair_edit_kind,
    sample_noisy_candidates,
    to_preference_jsonl,
    to_twinstep_records,
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


def test_sample_noisy_candidates_swap_preserves_unpadded_count() -> None:
    st_clean, pr_clean = make_random_valid_protocols(256, max_steps=24, seed=44)
    _, pr_mut = sample_noisy_candidates(256, max_steps=24, mutation_rate=1.0, seed=44)
    # Swapping adjacent steps (kind == 5) and all in-place edits must preserve unpadded slot count
    clean_unpadded = (~st_clean.sealed[:, :1] | True) & True  # dummy bool shape check
    assert clean_unpadded.shape == (256, 1)
    assert torch.equal((~pr_clean.pad_mask).sum(dim=1), (~pr_mut.pad_mask).sum(dim=1))


def test_make_error_twin_absent_op_raises_and_lengths_normalized() -> None:
    st_plate, pr_plate = make_serial_dilution_plate_batch(batch_size=4, num_rows_active=1, seed=0)
    with pytest.raises(ValueError, match="OP_INCUBATE"):
        make_error_twin(st_plate, pr_plate, "bad_temp")
    with pytest.raises(ValueError, match="OP_THAW"):
        make_error_twin(st_plate, pr_plate, "skip_thaw")

    # Transfer at step 0 with no preceding step must raise on swapped_steps
    st_min = LabStateBatch(
        volume=torch.tensor([[ul(100), 0]], dtype=torch.int64),
        capacity=torch.tensor([[ul(200), ul(200)]], dtype=torch.int64),
        sealed=torch.zeros((1, 2), dtype=torch.bool),
        frozen=torch.zeros((1, 2), dtype=torch.bool),
    )
    pr_xfer0 = ProtocolBatch.from_steps_list(
        [[{"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 1, "vol": ul(10)}]]
    )
    with pytest.raises(ValueError, match="step 0"):
        make_error_twin(st_min, pr_xfer0, "swapped_steps")

    # Normalized output max_steps across all 7 twin types on both PCR (full) and random valid
    for st_v, pr_v in (make_pcr_protocol(), make_random_valid_protocols(8, seed=9)):
        lengths = {make_error_twin(st_v, pr_v, et, seed=1)[1].max_steps for et in TWIN_ERROR_TYPES}
        assert lengths == {pr_v.max_steps + 1}


def test_insert_step_before_boundaries_and_parameters() -> None:
    base = ProtocolBatch.from_steps_list(
        [
            [
                {"op": OP_PICK_TIP},
                {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 1, "vol": 50},
            ]
        ] * 3
    )
    # Row 0: insert at pos=0; Row 1: insert at interior pos=1; Row 2: append at pos=max_steps (2)
    pos = torch.tensor([0, 1, 2], dtype=torch.int64)
    new_op = torch.tensor([OP_INCUBATE, OP_MIX, OP_DROP_TIP], dtype=torch.int64)
    new_src = torch.tensor([1, 1, 0], dtype=torch.int64)
    new_vol = torch.tensor([0, 150, 0], dtype=torch.int64)
    new_temp = torch.tensor([37, 0, 0], dtype=torch.int64)
    new_dur = torch.tensor([60, 0, 0], dtype=torch.int64)
    new_pip = torch.tensor([0, PIP_P200, 0], dtype=torch.int64)

    out = _insert_step_before(
        base,
        pos=pos,
        new_op=new_op,
        new_src=new_src,
        new_vol=new_vol,
        new_temp=new_temp,
        new_dur=new_dur,
        new_pip=new_pip,
    )
    assert out.max_steps == 3
    assert out.op[0].tolist() == [OP_INCUBATE, OP_PICK_TIP, OP_TRANSFER]
    assert int(out.temp[0, 0]) == 37 and int(out.duration[0, 0]) == 60
    assert out.op[1].tolist() == [OP_PICK_TIP, OP_MIX, OP_TRANSFER]
    assert int(out.vol[1, 1]) == 150 and int(out.pipette[1, 1]) == PIP_P200
    assert out.op[2].tolist() == [OP_PICK_TIP, OP_TRANSFER, OP_DROP_TIP]
    assert not bool(out.pad_mask.any().item())


def test_twinstep_and_preference_jsonl_roundtrip(tmp_path: Path) -> None:
    st, pr = make_random_valid_protocols(8, seed=12)
    st_b, pr_b = make_error_twin(st, pr, "wrong_pipette", seed=13)
    rr = batched_repair(st_b, pr_b, num_edits=24)
    assert rr.num_repaired == 8

    ts_records = to_twinstep_records(st_b, pr_b, rr.repaired_protocols, error_type="wrong_pipette")
    assert len(ts_records) == 8
    assert all(r["task"] == "error_detection" and r["label"] == "wrong_pipette" for r in ts_records)
    assert all(r["repaired_verified"] is True and r["step_index"] is not None for r in ts_records)

    out_jsonl = tmp_path / "prefs.jsonl"
    raw = to_preference_jsonl(st_b, pr_b, rr.repaired_protocols, path=out_jsonl, error_type="wrong_pipette")
    assert out_jsonl.exists() and len(raw.strip().splitlines()) == 8

    st_rt, bad_rt, good_rt, recs = from_preference_jsonl(out_jsonl)
    assert len(recs) == 8
    assert not bool(check_protocols_batched(st_rt, bad_rt).ok.any().item())
    assert bool(check_protocols_batched(st_rt, good_rt).ok.all().item())
