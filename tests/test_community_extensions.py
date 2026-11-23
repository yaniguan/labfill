"""Tests for Idea #16: DAPO Zero-Variance Filtering, Curriculum bf16 Packer & LabFillLean."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402
import torch  # noqa: E402

from community_extensions import (  # noqa: E402
    PackedSequenceSlice,
    PromptGroupRollout,
    compute_grpo_advantages,
    filter_and_score_frontier_groups,
    materialize_bf16_packed_batch,
    pack_curriculum_bins_bfd,
    run_benchmark,
    verify_packed_bin,
)


def test_zero_variance_groups_yield_exact_zero_grpo_gradient() -> None:
    all_fail = PromptGroupRollout(group_id=0, rollout_rewards=(0, 0, 0, 0), token_len=256)
    all_pass = PromptGroupRollout(group_id=1, rollout_rewards=(1, 1, 1, 1), token_len=256)
    frontier = PromptGroupRollout(group_id=2, rollout_rewards=(1, 0, 1, 0), token_len=256)

    assert all_fail.has_nonzero_gradient is False
    assert all_pass.has_nonzero_gradient is False
    assert frontier.has_nonzero_gradient is True
    assert frontier.curriculum_weight == pytest.approx(0.25)

    assert compute_grpo_advantages(all_fail.rollout_rewards) == (0.0, 0.0, 0.0, 0.0)
    assert compute_grpo_advantages(all_pass.rollout_rewards) == (0.0, 0.0, 0.0, 0.0)
    adv_front = compute_grpo_advantages(frontier.rollout_rewards)
    assert any(abs(a) > 0.5 for a in adv_front)
    assert sum(adv_front) == pytest.approx(0.0, abs=1e-9)

    kept, dropped = filter_and_score_frontier_groups([all_fail, frontier, all_pass])
    assert dropped == 2
    assert [g.group_id for g in kept] == [2]


def test_verify_packed_bin_mirrors_labfill_lean_rules() -> None:
    valid_slices = (
        PackedSequenceSlice(seq_id=10, offset=0, length=700, curriculum_weight=0.25),
        PackedSequenceSlice(seq_id=11, offset=700, length=800, curriculum_weight=0.1875),
        PackedSequenceSlice(seq_id=12, offset=1500, length=548, curriculum_weight=0.25),
    )
    ok, rule = verify_packed_bin(2048, 2048, valid_slices)
    assert ok is True
    assert rule is None

    # Overlapping distinct sequences rejected
    overlapping = (
        PackedSequenceSlice(seq_id=10, offset=0, length=700),
        PackedSequenceSlice(seq_id=11, offset=650, length=500),
    )
    ok_ov, rule_ov = verify_packed_bin(2048, 1200, overlapping)
    assert ok_ov is False
    assert rule_ov == "packed_slices_non_overlapping"

    # Bin overflow rejected
    overflow = (PackedSequenceSlice(seq_id=10, offset=1500, length=600),)
    ok_cap, rule_cap = verify_packed_bin(2048, 600, overflow)
    assert ok_cap is False
    assert rule_cap == "packed_slice_within_bin_cap"

    # Token mismatch rejected
    ok_tok, rule_tok = verify_packed_bin(2048, 1999, valid_slices)
    assert ok_tok is False
    assert rule_tok == "packed_real_token_conservation"


def test_bfd_packing_and_bf16_tensor_materialization() -> None:
    groups = [
        PromptGroupRollout(0, (0, 0, 0, 0), 900),
        PromptGroupRollout(1, (1, 0, 1, 0), 1100),
        PromptGroupRollout(2, (1, 1, 1, 0), 900),
        PromptGroupRollout(3, (1, 0, 0, 0), 950),
        PromptGroupRollout(4, (1, 1, 1, 1), 600),
    ]
    bins, dropped = pack_curriculum_bins_bfd(groups, bin_cap=2048)
    assert dropped == 2
    # 1100 + 900 <= 2048 fits in bin 0; 950 goes to bin 1 -> 2 bins total
    assert len(bins) == 2
    assert sum(b.used_tokens for b in bins) == 1100 + 900 + 950

    batch = materialize_bf16_packed_batch(bins, bin_cap=2048)
    assert batch["curriculum_weights_bf16"].dtype == torch.bfloat16
    assert batch["seq_ids"].shape == (2, 2048)
    # Position IDs reset to 0 at the start of every packed slice
    for b_idx, pbin in enumerate(bins):
        for sl in pbin.slices:
            assert int(batch["position_ids"][b_idx, sl.offset].item()) == 0
            assert int(batch["position_ids"][b_idx, sl.offset + sl.length - 1].item()) == (
                sl.length - 1
            )

    with pytest.raises(ValueError):
        pack_curriculum_bins_bfd([PromptGroupRollout(9, (1, 0), 3000)], bin_cap=2048)


def test_labfill_lean_module_hygiene() -> None:
    lean_file = REPO_ROOT / "lean" / "LabFillLean.lean"
    assert lean_file.is_file()
    src = lean_file.read_text(encoding="utf-8")
    for required in (
        "structure PackedSlice",
        "def checkPackedBin",
        "theorem checkPackedBin_conserves_tokens",
        "theorem checkPackedBin_pairwise_disjoint",
        "theorem zero_or_all_pass_has_zero_weight",
    ):
        assert required in src
    assert re.search(r"\bsorry\b", src) is None
    assert re.search(r"\baxiom\b", src) is None


def test_run_benchmark_1000_prompt_groups() -> None:
    report = run_benchmark(num_prompt_groups=1000, bin_cap=2048, rollouts_per_group=8, seed=42)
    assert report["num_prompt_groups"] == 1000
    assert report["bin_cap"] == 2048
    assert report["zero_variance_groups_dropped"] > 400

    naive = report["naive_padded_unfiltered"]
    packed = report["dapo_curriculum_bf16_packed"]

    assert packed["all_bins_lean_verified"] is True
    assert packed["nonzero_grad_tokens"] == naive["nonzero_grad_tokens"]
    assert packed["effective_nonzero_grad_tokens_pct"] > 90.0
    assert naive["effective_nonzero_grad_tokens_pct"] < 20.0
    assert report["useful_gradient_throughput_speedup_x"] > 10.0
