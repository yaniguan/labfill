# `LabFillLean` — Packed BF16 Sequence Non-Overlap & Zero-Variance Filtering (`L3`)

Self-contained **Lean 4** (`leanprover/lean4:v4.36.0`) verification kernel for `labfill`, synchronized with the central Auto-Lab Trust Regime at [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean) (`AutoLabLean/L3_PostTrain.lean`).

## Formalized Guardrails & Proved Theorems

| Component | Definitions | Machine-Checked Theorems |
| :--- | :--- | :--- |
| **Shared Diagnostics** | `RegimeViolation`, `firstViolation` | `firstViolation_nil`, `firstViolation_cons`, `firstViolation_none_iff` |
| **Packed BF16 Sequence Bins** | `PackedSlice`, `slicesDisjoint`, `sumSliceLens`, `checkPackedBin` | `checkPackedBin_conserves_tokens`, `checkPackedBin_pairwise_disjoint` |
| **DAPO Zero-Variance Filter** | `PromptGroupSpec`, `frontierVarianceWeight`, `isInformativePromptGroup` | `zero_or_all_pass_has_zero_weight`, `informative_group_iff_positive_weight` |

## Build & Verify

```bash
cd lean
lake build
```

Zero external dependencies, `0 sorry`, `0 axiom`, and 8 compile-time `#guard` checks.
