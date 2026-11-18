from __future__ import annotations

from labfill.checker import check_protocols_batched
from labfill.protocol import make_random_valid_protocols
from labfill.twins import (
    TWIN_ERROR_TYPES,
    batched_repair,
    generate_check_keep,
    make_error_twin,
    measure_twin_recall,
)


def test_seeded_error_twin_recall() -> None:
    rep = measure_twin_recall(batch_size=128, seed=10)
    assert rep.false_positive_rate_on_valid == 0.0
    for et in TWIN_ERROR_TYPES:
        assert rep.per_error_recall[et] == 1.0


def test_generate_check_keep_filters_strictly() -> None:
    res = generate_check_keep(num_candidates=512, mutation_rate=0.3, seed=21)
    assert 0 < res.num_kept < res.num_candidates
    assert res.candidates_per_sec > 0.0
    recheck = check_protocols_batched(res.kept_state, res.kept_protocols)
    assert bool(recheck.ok.all().item()) is True


def test_batched_repair_produces_verified_protocols() -> None:
    st, pr = make_random_valid_protocols(32, seed=5)
    for et in ("wrong_pipette", "missing_tip", "skip_thaw", "sealed_plate", "bad_temp", "swapped_steps"):
        st_b, pr_b = make_error_twin(st, pr, et, seed=6)
        assert not bool(check_protocols_batched(st_b, pr_b).ok.any().item())

        rr = batched_repair(st_b, pr_b, num_edits=24)
        assert rr.repair_rate >= 0.95
        idx = rr.repaired_mask.nonzero(as_tuple=False).squeeze(1)
        verified = check_protocols_batched(st_b.index_select(idx), rr.repaired_protocols.index_select(idx))
        assert bool(verified.ok.all().item()) is True
