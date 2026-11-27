# `LabFillLean` — Packed BF16 Sequence Non-Overlap, `packBFD` Bounds & Attention Isolation (`L3`)

Self-contained **Lean 4** (`leanprover/lean4:v4.36.0`) verification kernel for `labfill`, synchronized with the central Auto-Lab Trust Regime at [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean) (`AutoLabLean/L3_PostTrain.lean`).

## Formalized Guardrails & Proved Theorems

| Component | Definitions | Machine-Checked Theorems |
| :--- | :--- | :--- |
| **Shared Diagnostics** | `RegimeViolation`, `firstViolation` | `firstViolation_nil`, `firstViolation_cons`, `firstViolation_none_iff` |
| **Packed BF16 Sequence Bins** | `PackedSlice`, `slicesStrictlyDisjoint`, `allPairsDisjoint`, `sumSliceLens`, `checkPackedBin` | `checkPackedBin_conserves_tokens`, `checkPackedBin_pairwise_strict`, `checkPackedBin_unique_seqIds`, `checkPackedBin_pairwise_disjoint`, `checkPackedBin_sum_le_binCap` |
| **Block-Diagonal Attention** | `inSlice`, `seqIdAt`, `posIdAt`, `allowBlockDiagAttn` | `seqIdAt_posIdAt_of_mem`, `attn_zero_cross_sequence_leak`, `attn_pad_isolated_from_real`, `attn_causal_no_future_leak`, `attn_row_nonempty_no_nan` |
| **Bin-Packing & `packBFD`** | `sumBinsTokens`, `sumItemLens`, `packFirstFit`, `packBFD` | `valid_bins_capacity_lower_bound`, `packBFD_all_bins_valid`, `packBFD_pairwise_fullness`, `packBFD_two_approx`, `packBFD_le_two_opt` |
| **DAPO Zero-Variance Filter** | `PromptGroupSpec`, `frontierVarianceWeight`, `isInformativePromptGroup` | `zero_or_all_pass_has_zero_weight`, `informative_group_iff_positive_weight` |

## Build & Verify

```bash
cd lean
lake build
```

Zero external dependencies, `0 sorry`, `0 axiom`, `0 warnings`, and 13 compile-time `#guard` witnesses.
