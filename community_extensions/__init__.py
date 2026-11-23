"""Community extensions for labfill (Layer L3)."""

from community_extensions.curriculum_difficulty_packer import (
    PackedCurriculumBin,
    PackedSequenceSlice,
    PromptGroupRollout,
    compute_grpo_advantages,
    filter_and_score_frontier_groups,
    materialize_bf16_packed_batch,
    pack_curriculum_bins_bfd,
    run_benchmark,
    verify_packed_bin,
)

__all__ = [
    "PackedCurriculumBin",
    "PackedSequenceSlice",
    "PromptGroupRollout",
    "compute_grpo_advantages",
    "filter_and_score_frontier_groups",
    "materialize_bf16_packed_batch",
    "pack_curriculum_bins_bfd",
    "run_benchmark",
    "verify_packed_bin",
]
