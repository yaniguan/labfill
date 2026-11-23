/-!
# LabFillLean: Packed BF16 Sequence Non-Overlap, Token Conservation & Zero-Variance Filtering
(`labfill` · Layer L3)

Self-contained Lean 4.36.0 formalization synced with the central Auto-Lab Trust Regime
(`yaniguan/auto-lab/tree/main/lean`, `AutoLabLean/L3_PostTrain.lean`), covering:

1. **Packed Sequence Bins (`PackedSlice`, `checkPackedBin`, `checkPackedBin_conserves_tokens`)**:
   Verifies positive slice lengths, `offset + len ≤ binCap`, pairwise non-overlapping intervals
   across distinct sequence IDs (preventing cross-sequence buffer corruption), and exact
   real-token conservation (`sumSliceLens slices = expectedRealTokens`).
2. **DAPO Zero-Variance Frontier Filter (`PromptGroupSpec`, `frontierVarianceWeight`)**:
   Proves that prompt groups with `0%` or `100%` rollout pass rate carry zero within-group
   reward variance (`numPassed * (numRollouts - numPassed) = 0`) and are soundly dropped
   before sequence packing.
-/
namespace LabFillLean

/-- Structured boundary violation reporting rule identifier and diagnostic detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

/-- Return the first failing check in a list of `(holds, violation)` pairs. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, v⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. Packed BF16 Sequence Non-Overlap & Exact Token Conservation -/

structure PackedSlice where
  seqId  : Nat
  offset : Nat
  len    : Nat
  deriving DecidableEq, Repr

def slicesDisjoint (a b : PackedSlice) : Bool :=
  decide (a.seqId = b.seqId ∨ a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset)

def sumSliceLens : List PackedSlice → Nat
  | [] => 0
  | s :: ss => s.len + sumSliceLens ss

def checkPackedBin (binCap expectedRealTokens : Nat) (slices : List PackedSlice) :
    Option RegimeViolation :=
  firstViolation
    [ (slices.all (fun s => decide (0 < s.len ∧ s.offset + s.len ≤ binCap)),
        { repo := "labfill", rule := "packed_slice_within_bin_cap",
          detail := "packed sequence slice has zero length or overflows bin capacity" }),
      (slices.all (fun a => slices.all (fun b => slicesDisjoint a b)),
        { repo := "labfill", rule := "packed_slices_non_overlapping",
          detail := "two distinct sequences overlap in packed token buffer (cross-sequence leak)" }),
      (decide (sumSliceLens slices = expectedRealTokens),
        { repo := "labfill", rule := "packed_real_token_conservation",
          detail := "packed bin token count does not match input protocol token sum" }) ]

/-- Token Conservation & Capacity Theorem:
    if `checkPackedBin` accepts `slices`, total packed tokens equal `expectedRealTokens`
    and every slice fits within `binCap`. -/
theorem checkPackedBin_conserves_tokens (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice)
    (hok : checkPackedBin binCap expectedRealTokens slices = none) :
    sumSliceLens slices = expectedRealTokens ∧
    ∀ s ∈ slices, s.offset + s.len ≤ binCap := by
  unfold checkPackedBin at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             List.all_eq_true, decide_eq_true_eq] at hok
  rcases hok with ⟨h1, _, h3⟩
  exact ⟨h3, fun s hs => (h1 s hs).2⟩

/-- Pairwise Non-Overlap Theorem:
    if `checkPackedBin` accepts `slices`, any two distinct sequence IDs occupy
    disjoint token intervals inside `[0, binCap)`. -/
theorem checkPackedBin_pairwise_disjoint (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice) (a b : PackedSlice)
    (hok : checkPackedBin binCap expectedRealTokens slices = none)
    (ha : a ∈ slices) (hb : b ∈ slices) (hne : a.seqId ≠ b.seqId) :
    a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset := by
  unfold checkPackedBin at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             List.all_eq_true] at hok
  rcases hok with ⟨_, hdisj, _⟩
  have hab := hdisj a ha b hb
  unfold slicesDisjoint at hab
  simp only [decide_eq_true_eq] at hab
  rcases hab with heq | hleft | hright
  · exact False.elim (hne heq)
  · exact Or.inl hleft
  · exact Or.inr hright

/-! ## 2. DAPO Zero-Variance Prompt-Group Filtering -/

structure PromptGroupSpec where
  groupId     : Nat
  numRollouts : Nat
  numPassed   : Nat
  tokenLen    : Nat
  deriving DecidableEq, Repr

/-- Integer Bernoulli variance numerator `numPassed * (numRollouts - numPassed)`
    proportional to `p * (1 - p)` across `numRollouts` binary RLVR rollouts. -/
def frontierVarianceWeight (g : PromptGroupSpec) : Nat :=
  g.numPassed * (g.numRollouts - g.numPassed)

def isInformativePromptGroup (g : PromptGroupSpec) : Bool :=
  decide (0 < g.numPassed ∧ g.numPassed < g.numRollouts)

/-- Zero-Variance Theorem: prompt groups with `0` passes (`pass_rate = 0.0`) or all passes
    (`pass_rate = 1.0`) have strictly `0` variance weight and `isInformativePromptGroup = false`. -/
theorem zero_or_all_pass_has_zero_weight (g : PromptGroupSpec)
    (hdeg : g.numPassed = 0 ∨ g.numRollouts ≤ g.numPassed) :
    frontierVarianceWeight g = 0 ∧ isInformativePromptGroup g = false := by
  unfold frontierVarianceWeight isInformativePromptGroup
  rcases hdeg with hz | hall
  · simp [hz]
  · have hsub : g.numRollouts - g.numPassed = 0 := by omega
    have hnd : decide (0 < g.numPassed ∧ g.numPassed < g.numRollouts) = false :=
      decide_eq_false (by omega)
    simp [hsub, hnd]

/-- Frontier Characterization Theorem: a prompt group is informative (`0 < numPassed < numRollouts`)
    if and only if its integer variance weight is strictly positive. -/
theorem informative_group_iff_positive_weight (g : PromptGroupSpec) :
    isInformativePromptGroup g = true ↔ 0 < frontierVarianceWeight g := by
  unfold isInformativePromptGroup frontierVarianceWeight
  simp only [decide_eq_true_eq]
  constructor
  · intro ⟨hpos, hlt⟩
    have hdiff : 0 < g.numRollouts - g.numPassed := by omega
    exact Nat.mul_pos hpos hdiff
  · intro hmul
    have hp : 0 < g.numPassed := by
      cases Nat.eq_zero_or_pos g.numPassed with
      | inl hz => simp [hz] at hmul
      | inr hp => exact hp
    have hd : 0 < g.numRollouts - g.numPassed := by
      cases Nat.eq_zero_or_pos (g.numRollouts - g.numPassed) with
      | inl hz => simp [hz] at hmul
      | inr hd => exact hd
    omega

/-! ## 3. Compile-Time `#guard` Witnesses -/

#guard checkPackedBin 2048 1500
  [ { seqId := 1, offset := 0, len := 700 },
    { seqId := 2, offset := 700, len := 500 },
    { seqId := 3, offset := 1200, len := 300 } ] == none

#guard (checkPackedBin 2048 1500
  [ { seqId := 1, offset := 0, len := 700 },
    { seqId := 2, offset := 650, len := 500 },
    { seqId := 3, offset := 1150, len := 350 } ]).isSome

#guard (checkPackedBin 2048 2100
  [ { seqId := 1, offset := 0, len := 1200 },
    { seqId := 2, offset := 1200, len := 900 } ]).isSome

#guard (checkPackedBin 2048 1400
  [ { seqId := 1, offset := 0, len := 700 },
    { seqId := 2, offset := 700, len := 500 } ]).isSome

#guard isInformativePromptGroup { groupId := 1, numRollouts := 8, numPassed := 0, tokenLen := 512 } == false
#guard isInformativePromptGroup { groupId := 2, numRollouts := 8, numPassed := 8, tokenLen := 512 } == false
#guard isInformativePromptGroup { groupId := 3, numRollouts := 8, numPassed := 4, tokenLen := 512 } == true
#guard frontierVarianceWeight { groupId := 3, numRollouts := 8, numPassed := 4, tokenLen := 512 } == 16

end LabFillLean
