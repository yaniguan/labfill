/-!
# LabFillLean: Packed BF16 Sequence Non-Overlap, BFD Bin-Packing Bounds, Causal Block-Diagonal
# Attention Isolation & Zero-Variance Filtering (`labfill` · Layer L3)

Self-contained Lean 4.36.0 formalization synchronized with the central Auto-Lab Trust Regime
(`yaniguan/auto-lab/tree/main/lean`, `AutoLabLean/L3_PostTrain.lean`), covering:

1. **Packed Sequence Bins & Unique-`seqId` Disjointness (`PackedSlice`, `checkPackedBin`)**:
   Verifies positive slice lengths, `offset + len ≤ binCap`, index-wise pairwise distinct
   `seqId`s and non-overlapping `[offset, offset + len)` intervals (closing the duplicate-`seqId`
   overlap loophole), exact real-token conservation (`sumSliceLens slices = expectedRealTokens`),
   and single-bin capacity upper bound (`expectedRealTokens ≤ binCap`).
2. **Causal Block-Diagonal Attention Isolation (`seqIdAt`, `posIdAt`, `allowBlockDiagAttn`)**:
   Formalizes token-level `seq_id`, reset `position_ids`, zero cross-sequence attention leakage,
   causal future-masking (`kv ≤ q`), and NaN-free softmax row non-emptiness across active and
   trailing padded positions (`0 ≤ q < binCap`).
3. **Executable First-Fit & Best-Fit-Decreasing (`packFirstFit`, `packBFD`) & Bounds**:
   Formalizes multi-bin FF and BFD packing, proving end-to-end bin validity (`packBFD_all_bins_valid`),
   global token conservation, capacity lower bound (`valid_bins_capacity_lower_bound`),
   pairwise bin fullness (`sumSliceLens B_i + sumSliceLens B_j > binCap`), and the strict
   2-approximation bound (`(bins.length - 1) * binCap < 2 * sumItemLens items`, `bins.length ≤ 2 * OPT`).
4. **DAPO Zero-Variance Frontier Filter (`PromptGroupSpec`, `frontierVarianceWeight`)**:
   Proves that prompt groups with `0%` or `100%` rollout pass rate carry zero within-group
   reward variance (`numPassed * (numRollouts - numPassed) = 0`) and are soundly dropped.
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

/-! ## 1. Packed BF16 Sequence Non-Overlap, Unique `seqId` & Exact Token Conservation -/

structure PackedSlice where
  seqId  : Nat
  offset : Nat
  len    : Nat
  deriving DecidableEq, Repr

/-- Legacy distinct-`seqId` interval predicate retained for compatibility. -/
def slicesDisjoint (a b : PackedSlice) : Bool :=
  decide (a.seqId = b.seqId ∨ a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset)

/-- Strict distinct-entry disjointness: distinct `seqId` AND non-overlapping `[offset, offset + len)`. -/
def slicesStrictlyDisjoint (a b : PackedSlice) : Bool :=
  decide (a.seqId ≠ b.seqId ∧ (a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset))

/-- Executable `List.Pairwise` check enforcing `slicesStrictlyDisjoint` across all `i < j`. -/
def allPairsDisjoint : List PackedSlice → Bool
  | [] => true
  | s :: ss => ss.all (slicesStrictlyDisjoint s) && allPairsDisjoint ss

def sumSliceLens : List PackedSlice → Nat
  | [] => 0
  | s :: ss => s.len + sumSliceLens ss

@[simp] theorem sumSliceLens_nil : sumSliceLens [] = 0 := rfl

@[simp] theorem sumSliceLens_cons (s : PackedSlice) (ss : List PackedSlice) :
    sumSliceLens (s :: ss) = s.len + sumSliceLens ss := rfl

theorem sumSliceLens_append (xs ys : List PackedSlice) :
    sumSliceLens (xs ++ ys) = sumSliceLens xs + sumSliceLens ys := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, Nat.add_assoc]

theorem allPairsDisjoint_iff_pairwise (slices : List PackedSlice) :
    allPairsDisjoint slices = true ↔
    List.Pairwise (fun a b => a.seqId ≠ b.seqId ∧
      (a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset)) slices := by
  induction slices with
  | nil => simp [allPairsDisjoint]
  | cons hd tl ih =>
    simp [allPairsDisjoint, Bool.and_eq_true, List.all_eq_true, slicesStrictlyDisjoint,
          decide_eq_true_eq, List.pairwise_cons, ih]

def checkPackedBin (binCap expectedRealTokens : Nat) (slices : List PackedSlice) :
    Option RegimeViolation :=
  firstViolation
    [ (slices.all (fun s => decide (0 < s.len ∧ s.offset + s.len ≤ binCap)),
        { repo := "labfill", rule := "packed_slice_within_bin_cap",
          detail := "packed sequence slice has zero length or overflows bin capacity" }),
      (allPairsDisjoint slices,
        { repo := "labfill", rule := "packed_slices_non_overlapping",
          detail := "two packed slices share a duplicate seqId or overlap in token buffer" }),
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

/-- Strict Index-Wise Disjointness & Unique-`seqId` Theorem:
    if `checkPackedBin` accepts `slices`, every distinct pair of entries `(i < j)` has
    distinct `seqId`s AND disjoint `[offset, offset + len)` intervals. -/
theorem checkPackedBin_pairwise_strict (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice)
    (hok : checkPackedBin binCap expectedRealTokens slices = none) :
    List.Pairwise (fun a b => a.seqId ≠ b.seqId ∧
      (a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset)) slices := by
  unfold checkPackedBin at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true] at hok
  exact (allPairsDisjoint_iff_pairwise slices).mp hok.2.1

/-- Unique-`seqId` Theorem: no two entries in a valid bin can share the same `seqId`. -/
theorem checkPackedBin_unique_seqIds (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice)
    (hok : checkPackedBin binCap expectedRealTokens slices = none) :
    List.Pairwise (fun a b => a.seqId ≠ b.seqId) slices := by
  exact (checkPackedBin_pairwise_strict binCap expectedRealTokens slices hok).imp (fun h => h.1)

private theorem pairwise_symm_mem {α : Type} (R : α → α → Prop)
    (hsymm : ∀ x y, R x y → R y x) (xs : List α)
    (hp : List.Pairwise R xs) {a b : α}
    (ha : a ∈ xs) (hb : b ∈ xs) (hne : a ≠ b) :
    R a b := by
  induction xs with
  | nil => cases ha
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hp
    simp only [List.mem_cons] at ha hb
    rcases ha with rfl | ha_tl
    · rcases hb with rfl | hb_tl
      · exact False.elim (hne rfl)
      · exact hp.1 b hb_tl
    · rcases hb with rfl | hb_tl
      · exact hsymm b a (hp.1 a ha_tl)
      · exact ih hp.2 ha_tl hb_tl

/-- Pairwise Non-Overlap Theorem:
    if `checkPackedBin` accepts `slices`, any two distinct sequence IDs occupy
    disjoint token intervals inside `[0, binCap)`. -/
theorem checkPackedBin_pairwise_disjoint (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice) (a b : PackedSlice)
    (hok : checkPackedBin binCap expectedRealTokens slices = none)
    (ha : a ∈ slices) (hb : b ∈ slices) (hne : a.seqId ≠ b.seqId) :
    a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset := by
  have hp := checkPackedBin_pairwise_strict binCap expectedRealTokens slices hok
  have hne_slice : a ≠ b := by
    intro heq
    rw [heq] at hne
    exact hne rfl
  have hsymm : ∀ x y : PackedSlice,
      (x.seqId ≠ y.seqId ∧ (x.offset + x.len ≤ y.offset ∨ y.offset + y.len ≤ x.offset)) →
      (y.seqId ≠ x.seqId ∧ (y.offset + y.len ≤ x.offset ∨ x.offset + x.len ≤ y.offset)) := by
    intro x y ⟨h1, h2⟩
    exact ⟨Ne.symm h1, h2.symm⟩
  exact (pairwise_symm_mem _ hsymm slices hp ha hb hne_slice).2

/-! ### Single-Bin Disjoint-Interval Sum Upper Bound (`sumSliceLens slices ≤ binCap`) -/

def insertSliceByOffset (s : PackedSlice) : List PackedSlice → List PackedSlice
  | [] => [s]
  | x :: xs => if s.offset ≤ x.offset then s :: x :: xs else x :: insertSliceByOffset s xs

def sortSlicesByOffset : List PackedSlice → List PackedSlice
  | [] => []
  | s :: ss => insertSliceByOffset s (sortSlicesByOffset ss)

private theorem mem_insertSliceByOffset (s y : PackedSlice) (xs : List PackedSlice) :
    y ∈ insertSliceByOffset s xs ↔ y = s ∨ y ∈ xs := by
  induction xs with
  | nil => simp [insertSliceByOffset]
  | cons hd tl ih =>
    unfold insertSliceByOffset
    split
    · simp
    · simp only [List.mem_cons, ih]
      constructor
      · intro h
        rcases h with h1 | h2 | h3
        · exact Or.inr (Or.inl h1)
        · exact Or.inl h2
        · exact Or.inr (Or.inr h3)
      · intro h
        rcases h with h1 | h2 | h3
        · exact Or.inr (Or.inl h1)
        · exact Or.inl h2
        · exact Or.inr (Or.inr h3)

private theorem mem_sortSlicesByOffset (y : PackedSlice) (xs : List PackedSlice) :
    y ∈ sortSlicesByOffset xs ↔ y ∈ xs := by
  induction xs with
  | nil => simp [sortSlicesByOffset]
  | cons hd tl ih => simp [sortSlicesByOffset, mem_insertSliceByOffset, ih]

private theorem sumSliceLens_insertSliceByOffset (s : PackedSlice) (xs : List PackedSlice) :
    sumSliceLens (insertSliceByOffset s xs) = s.len + sumSliceLens xs := by
  induction xs with
  | nil => simp [insertSliceByOffset]
  | cons hd tl ih =>
    unfold insertSliceByOffset
    split
    · simp
    · simp [ih, Nat.add_left_comm]

private theorem sumSliceLens_sortSlicesByOffset (xs : List PackedSlice) :
    sumSliceLens (sortSlicesByOffset xs) = sumSliceLens xs := by
  induction xs with
  | nil => rfl
  | cons hd tl ih => simp [sortSlicesByOffset, sumSliceLens_insertSliceByOffset, ih]

private theorem pairwise_insertSliceByOffset
    (R : PackedSlice → PackedSlice → Prop)
    (hsymm : ∀ a b, R a b → R b a)
    (s : PackedSlice) (xs : List PackedSlice)
    (hxs : List.Pairwise R xs)
    (hs : ∀ x ∈ xs, R s x) :
    List.Pairwise R (insertSliceByOffset s xs) := by
  induction xs with
  | nil => simp [insertSliceByOffset]
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hxs
    unfold insertSliceByOffset
    split
    · rw [List.pairwise_cons]
      refine ⟨?_, List.pairwise_cons.mpr hxs⟩
      intro y hy
      simp only [List.mem_cons] at hy
      rcases hy with heq | hy_tl
      · rw [heq]
        exact hs hd (by simp)
      · exact hs y (by simp [hy_tl])
    · rw [List.pairwise_cons]
      refine ⟨?_, ih hxs.2 (fun y hy => hs y (by simp [hy]))⟩
      intro y hy
      rw [mem_insertSliceByOffset] at hy
      rcases hy with heq | hy_tl
      · rw [heq]
        exact hsymm s hd (hs hd (by simp))
      · exact hxs.1 y hy_tl

private theorem sorted_insertSliceByOffset (s : PackedSlice) (xs : List PackedSlice)
    (hxs : List.Pairwise (fun a b => a.offset ≤ b.offset) xs) :
    List.Pairwise (fun a b => a.offset ≤ b.offset) (insertSliceByOffset s xs) := by
  induction xs with
  | nil => simp [insertSliceByOffset]
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hxs
    unfold insertSliceByOffset
    split
    next hle =>
      rw [List.pairwise_cons]
      refine ⟨?_, List.pairwise_cons.mpr hxs⟩
      intro y hy
      simp only [List.mem_cons] at hy
      rcases hy with heq | hy_tl
      · rw [heq]
        exact hle
      · exact Nat.le_trans hle (hxs.1 y hy_tl)
    next hgt =>
      rw [List.pairwise_cons]
      refine ⟨?_, ih hxs.2⟩
      intro y hy
      rw [mem_insertSliceByOffset] at hy
      rcases hy with heq | hy_tl
      · rw [heq]
        omega
      · exact hxs.1 y hy_tl

private theorem sorted_sortSlicesByOffset (xs : List PackedSlice) :
    List.Pairwise (fun a b => a.offset ≤ b.offset) (sortSlicesByOffset xs) := by
  induction xs with
  | nil => simp [sortSlicesByOffset]
  | cons hd tl ih => exact sorted_insertSliceByOffset hd (sortSlicesByOffset tl) ih

private theorem pairwise_sortSlicesByOffset (xs : List PackedSlice)
    (hp : List.Pairwise (fun a b => a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset) xs) :
    List.Pairwise (fun a b => a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset)
      (sortSlicesByOffset xs) := by
  induction xs with
  | nil => simp [sortSlicesByOffset]
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hp
    exact pairwise_insertSliceByOffset _ (fun _ _ h => h.symm) hd (sortSlicesByOffset tl)
      (ih hp.2)
      (fun y hy => hp.1 y ((mem_sortSlicesByOffset y tl).mp hy))

private theorem sorted_disjoint_cursor_bound (binCap : Nat) :
    ∀ (ys : List PackedSlice) (cursor : Nat),
      cursor ≤ binCap →
      (∀ y ∈ ys, 0 < y.len ∧ y.offset + y.len ≤ binCap) →
      (∀ y ∈ ys, cursor ≤ y.offset) →
      List.Pairwise (fun a b => a.offset ≤ b.offset) ys →
      List.Pairwise (fun a b => a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset) ys →
      cursor + sumSliceLens ys ≤ binCap
  | [], cursor, hcur, _, _, _, _ => by simp [hcur]
  | y :: ys, cursor, _, hcap, hge, hord, hdisj => by
    rw [List.pairwise_cons] at hord hdisj
    have hy_cap := hcap y (by simp)
    have hy_cur := hge y (by simp)
    have hnext_ge : ∀ z ∈ ys, y.offset + y.len ≤ z.offset := by
      intro z hz
      have hz_cap := hcap z (by simp [hz])
      have h1 := hord.1 z hz
      have h2 := hdisj.1 z hz
      omega
    have ih := sorted_disjoint_cursor_bound binCap ys (y.offset + y.len)
      hy_cap.2
      (fun z hz => hcap z (by simp [hz]))
      hnext_ge
      hord.2
      hdisj.2
    simp only [sumSliceLens_cons]
    omega

/-- Single-Bin Capacity Soundness Theorem:
    any bin accepted by `checkPackedBin` has `expectedRealTokens ≤ binCap` (disjoint slices
    inside `[0, binCap)` cannot sum to more than `binCap` tokens). -/
theorem checkPackedBin_sum_le_binCap (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice)
    (hok : checkPackedBin binCap expectedRealTokens slices = none) :
    expectedRealTokens ≤ binCap := by
  unfold checkPackedBin at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             List.all_eq_true, decide_eq_true_eq] at hok
  rcases hok with ⟨hcap, hdisj_b, hsum⟩
  have hp_strict := (allPairsDisjoint_iff_pairwise slices).mp hdisj_b
  have hp_int : List.Pairwise
      (fun a b => a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset) slices :=
    hp_strict.imp (fun h => h.2)
  have hord := sorted_sortSlicesByOffset slices
  have hdisj_sorted := pairwise_sortSlicesByOffset slices hp_int
  have hcap_sorted : ∀ y ∈ sortSlicesByOffset slices, 0 < y.len ∧ y.offset + y.len ≤ binCap := by
    intro y hy
    exact hcap y ((mem_sortSlicesByOffset y slices).mp hy)
  have hbound := sorted_disjoint_cursor_bound binCap (sortSlicesByOffset slices) 0
    (Nat.zero_le binCap) hcap_sorted (fun _ _ => Nat.zero_le _) hord hdisj_sorted
  have hsum_eq : sumSliceLens (sortSlicesByOffset slices) = sumSliceLens slices :=
    sumSliceLens_sortSlicesByOffset slices
  omega

/-! ## 2. Causal Block-Diagonal Attention Isolation & NaN-Free Softmax Rows -/

/-- Whether token slot `pos` falls inside slice `s`'s half-open interval `[offset, offset + len)`. -/
def inSlice (s : PackedSlice) (pos : Nat) : Bool :=
  decide (s.offset ≤ pos ∧ pos < s.offset + s.len)

/-- Token-level packed `seq_id` (`0` = trailing pad, `s.seqId + 1` on slice `s`). -/
def seqIdAt : List PackedSlice → Nat → Nat
  | [], _ => 0
  | s :: ss, pos => if inSlice s pos then s.seqId + 1 else seqIdAt ss pos

/-- Token-level reset `position_id` (`0..s.len - 1` inside slice `s`, `0` on pad). -/
def posIdAt : List PackedSlice → Nat → Nat
  | [], _ => 0
  | s :: ss, pos => if inSlice s pos then pos - s.offset else posIdAt ss pos

/-- Exact Lean 4 mirror of `labfill.fill_gpu.build_block_diag_mask`:
    active queries (`seqId q > 0`) attend only to keys in the same packed sequence
    (intersected with `kv ≤ q` when `causal = true`), while padded queries (`seqId q = 0`)
    self-attend exclusively (`q = kv`) so softmax denominators are never zero. -/
def allowBlockDiagAttn (causal : Bool) (seqId : Nat → Nat) (q kv : Nat) : Bool :=
  if seqId q = 0 then
    decide (q = kv)
  else
    decide (seqId q = seqId kv ∧ (causal = false ∨ kv ≤ q))

theorem seqIdAt_posIdAt_of_mem (slices : List PackedSlice) (s : PackedSlice) (pos : Nat)
    (hp : List.Pairwise (fun a b => a.seqId ≠ b.seqId ∧
      (a.offset + a.len ≤ b.offset ∨ b.offset + b.len ≤ a.offset)) slices)
    (hs : s ∈ slices)
    (hin : inSlice s pos = true) :
    seqIdAt slices pos = s.seqId + 1 ∧ posIdAt slices pos = pos - s.offset ∧ posIdAt slices pos < s.len := by
  induction slices with
  | nil => cases hs
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hp
    simp only [List.mem_cons] at hs
    unfold seqIdAt posIdAt
    by_cases hhd : inSlice hd pos = true
    · rcases hs with rfl | htl
      · simp only [inSlice, decide_eq_true_eq] at hin
        simp [hhd]
        omega
      · exfalso
        have hdisj := (hp.1 s htl).2
        simp only [inSlice, decide_eq_true_eq] at hhd hin
        omega
    · have hhd_f : inSlice hd pos = false := Bool.eq_false_iff.mpr hhd
      rcases hs with rfl | htl
      · exact False.elim (hhd hin)
      · simp only [hhd_f, Bool.false_eq_true, ↓reduceIte]
        exact ih hp.2 htl

/-- Cross-Sequence Zero-Leakage Theorem:
    for any bin accepted by `checkPackedBin`, a query token `q` in slice `a` and key token `kv`
    in slice `b` with `a.seqId ≠ b.seqId` never attend (`allowBlockDiagAttn = false`). -/
theorem attn_zero_cross_sequence_leak (binCap expectedRealTokens : Nat)
    (slices : List PackedSlice) (a b : PackedSlice) (causal : Bool) (q kv : Nat)
    (hok : checkPackedBin binCap expectedRealTokens slices = none)
    (ha : a ∈ slices) (hb : b ∈ slices) (hne : a.seqId ≠ b.seqId)
    (hq : inSlice a q = true) (hkv : inSlice b kv = true) :
    allowBlockDiagAttn causal (seqIdAt slices) q kv = false := by
  have hp := checkPackedBin_pairwise_strict binCap expectedRealTokens slices hok
  have hqa := (seqIdAt_posIdAt_of_mem slices a q hp ha hq).1
  have hkvb := (seqIdAt_posIdAt_of_mem slices b kv hp hb hkv).1
  unfold allowBlockDiagAttn
  rw [hqa, hkvb]
  split
  · omega
  · exact decide_eq_false (by omega)

/-- Pad Isolation Theorem: active sequence queries (`seqId q > 0`) never attend to pad keys (`seqId kv = 0`). -/
theorem attn_pad_isolated_from_real (causal : Bool) (seqId : Nat → Nat) (q kv : Nat)
    (hq : 0 < seqId q) (hkv : seqId kv = 0) :
    allowBlockDiagAttn causal seqId q kv = false := by
  unfold allowBlockDiagAttn
  split
  · omega
  · rw [hkv]
    exact decide_eq_false (by omega)

/-- Causal No-Future-Leak Theorem: when `causal = true`, any permitted attention pair satisfies `kv ≤ q`. -/
theorem attn_causal_no_future_leak (seqId : Nat → Nat) (q kv : Nat)
    (hattn : allowBlockDiagAttn true seqId q kv = true) :
    kv ≤ q := by
  unfold allowBlockDiagAttn at hattn
  split at hattn
  · simp only [decide_eq_true_eq] at hattn
    omega
  · simp only [decide_eq_true_eq] at hattn
    rcases hattn with ⟨_, hcf | hle⟩
    · cases hcf
    · exact hle

/-- NaN-Free Softmax Row Theorem: every query slot `q < binCap` (both active tokens and trailing pad)
    attends to at least one valid key `kv < binCap` (specifically `kv = q`). -/
theorem attn_row_nonempty_no_nan (causal : Bool) (seqId : Nat → Nat) (binCap q : Nat)
    (hq : q < binCap) :
    ∃ kv, kv < binCap ∧ allowBlockDiagAttn causal seqId q kv = true := by
  refine ⟨q, hq, ?_⟩
  unfold allowBlockDiagAttn
  split
  · simp
  · cases causal <;> simp

/-! ## 3. Bin-Packing Lower Bound, Executable First-Fit & Best-Fit-Decreasing (`packBFD`) -/

def sumBinsTokens : List (List PackedSlice) → Nat
  | [] => 0
  | b :: bs => sumSliceLens b + sumBinsTokens bs

@[simp] theorem sumBinsTokens_nil : sumBinsTokens [] = 0 := rfl

@[simp] theorem sumBinsTokens_cons (b : List PackedSlice) (bs : List (List PackedSlice)) :
    sumBinsTokens (b :: bs) = sumSliceLens b + sumBinsTokens bs := rfl

theorem sumBinsTokens_append (xs ys : List (List PackedSlice)) :
    sumBinsTokens (xs ++ ys) = sumBinsTokens xs + sumBinsTokens ys := by
  induction xs with
  | nil => simp
  | cons x xs ih => simp [ih, Nat.add_assoc]

def sumItemLens : List (Nat × Nat) → Nat
  | [] => 0
  | item :: rest => item.2 + sumItemLens rest

@[simp] theorem sumItemLens_nil : sumItemLens [] = 0 := rfl

@[simp] theorem sumItemLens_cons (item : Nat × Nat) (rest : List (Nat × Nat)) :
    sumItemLens (item :: rest) = item.2 + sumItemLens rest := rfl

/-- Bin-Packing Capacity Lower Bound Theorem:
    any valid multi-bin packing `bins` where each bin satisfies `checkPackedBin` requires
    `sumBinsTokens bins ≤ bins.length * binCap` (establishing `OPT ≥ ceil(totalTokens / binCap)`). -/
theorem valid_bins_capacity_lower_bound (binCap : Nat) (bins : List (List PackedSlice))
    (hvalid : ∀ b ∈ bins, checkPackedBin binCap (sumSliceLens b) b = none) :
    sumBinsTokens bins ≤ bins.length * binCap := by
  induction bins with
  | nil => simp
  | cons b bs ih =>
    have hb : sumSliceLens b ≤ binCap :=
      checkPackedBin_sum_le_binCap binCap (sumSliceLens b) b (hvalid b (by simp))
    have hbs := ih (fun x hx => hvalid x (by simp [hx]))
    simp only [sumBinsTokens_cons, List.length_cons, Nat.add_mul, Nat.one_mul]
    omega

/-- Executable First-Fit single-item bin insertion. -/
def insertFirstFit (binCap sid len : Nat) : List (List PackedSlice) → List (List PackedSlice)
  | [] => [[{ seqId := sid, offset := 0, len := len }]]
  | b :: bs =>
    if sumSliceLens b + len ≤ binCap then
      (b ++ [{ seqId := sid, offset := sumSliceLens b, len := len }]) :: bs
    else
      b :: insertFirstFit binCap sid len bs

/-- Executable First-Fit multi-bin packer over `(seqId, len)` pairs. -/
def packFirstFit (binCap : Nat) (items : List (Nat × Nat)) : List (List PackedSlice) :=
  items.foldl (fun bins item => insertFirstFit binCap item.1 item.2 bins) []

/-- Insertion sort placing `(seqId, len)` items in non-increasing `len` order. -/
def insertItemDesc (x : Nat × Nat) : List (Nat × Nat) → List (Nat × Nat)
  | [] => [x]
  | y :: ys => if y.2 ≤ x.2 then x :: y :: ys else y :: insertItemDesc x ys

def sortItemsDesc : List (Nat × Nat) → List (Nat × Nat)
  | [] => []
  | x :: xs => insertItemDesc x (sortItemsDesc xs)

/-- Structural Best-Fit bin finder returning `(binIdx, usedTokens)` minimizing residual capacity
    `binCap - (used + len)` (equivalently maximizing `used`, breaking ties toward earliest bin). -/
def bestFitBinIdx (binCap len : Nat) : List (List PackedSlice) → Option (Nat × Nat)
  | [] => none
  | b :: bs =>
    let used := sumSliceLens b
    let tailBest := (bestFitBinIdx binCap len bs).map (fun p => (p.1 + 1, p.2))
    if used + len ≤ binCap then
      match tailBest with
      | none => some (0, used)
      | some (tIdx, tUsed) => if used < tUsed then some (tIdx, tUsed) else some (0, used)
    else
      tailBest

/-- Append `{ seqId := sid, offset := sumSliceLens b, len := len }` to bin at `targetIdx`. -/
def updateBinAt : Nat → Nat → Nat → List (List PackedSlice) → List (List PackedSlice)
  | _, _, _, [] => []
  | 0, sid, len, b :: bs =>
    (b ++ [{ seqId := sid, offset := sumSliceLens b, len := len }]) :: bs
  | k + 1, sid, len, b :: bs =>
    b :: updateBinAt k sid len bs

/-- Executable Best-Fit single-item bin insertion. -/
def insertBestFit (binCap sid len : Nat) (bins : List (List PackedSlice)) :
    List (List PackedSlice) :=
  match bestFitBinIdx binCap len bins with
  | none => bins ++ [[{ seqId := sid, offset := 0, len := len }]]
  | some (idx, _) => updateBinAt idx sid len bins

/-- Executable Best-Fit-Decreasing (`packBFD`) multi-bin packer mirroring `pack_curriculum_bins_bfd`. -/
def packBFD (binCap : Nat) (items : List (Nat × Nat)) : List (List PackedSlice) :=
  (sortItemsDesc items).foldl (fun bins item => insertBestFit binCap item.1 item.2 bins) []

/-! ### Helper Invariants for `packBFD` Soundness & Approximation Proofs -/

private theorem mem_insertItemDesc (x y : Nat × Nat) (xs : List (Nat × Nat)) :
    y ∈ insertItemDesc x xs ↔ y = x ∨ y ∈ xs := by
  induction xs with
  | nil => simp [insertItemDesc]
  | cons hd tl ih =>
    unfold insertItemDesc
    split
    · simp
    · simp only [List.mem_cons, ih]
      constructor
      · intro h
        rcases h with h1 | h2 | h3
        · exact Or.inr (Or.inl h1)
        · exact Or.inl h2
        · exact Or.inr (Or.inr h3)
      · intro h
        rcases h with h1 | h2 | h3
        · exact Or.inr (Or.inl h1)
        · exact Or.inl h2
        · exact Or.inr (Or.inr h3)

private theorem mem_sortItemsDesc (y : Nat × Nat) (xs : List (Nat × Nat)) :
    y ∈ sortItemsDesc xs ↔ y ∈ xs := by
  induction xs with
  | nil => simp [sortItemsDesc]
  | cons hd tl ih => simp [sortItemsDesc, mem_insertItemDesc, ih]

private theorem sumItemLens_insertItemDesc (x : Nat × Nat) (xs : List (Nat × Nat)) :
    sumItemLens (insertItemDesc x xs) = x.2 + sumItemLens xs := by
  induction xs with
  | nil => simp [insertItemDesc]
  | cons hd tl ih =>
    unfold insertItemDesc
    split
    · simp
    · simp [ih, Nat.add_left_comm]

theorem sumItemLens_sortItemsDesc (xs : List (Nat × Nat)) :
    sumItemLens (sortItemsDesc xs) = sumItemLens xs := by
  induction xs with
  | nil => rfl
  | cons hd tl ih => simp [sortItemsDesc, sumItemLens_insertItemDesc, ih]

private theorem pairwise_insertItemDesc (x : Nat × Nat) (xs : List (Nat × Nat))
    (hxs : List.Pairwise (fun a b => a.1 ≠ b.1) xs)
    (hx : ∀ y ∈ xs, x.1 ≠ y.1) :
    List.Pairwise (fun a b => a.1 ≠ b.1) (insertItemDesc x xs) := by
  induction xs with
  | nil => simp [insertItemDesc]
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hxs
    unfold insertItemDesc
    split
    · rw [List.pairwise_cons]
      refine ⟨?_, List.pairwise_cons.mpr hxs⟩
      intro y hy
      simp only [List.mem_cons] at hy
      rcases hy with heq | hy_tl
      · rw [heq]
        exact hx hd (by simp)
      · exact hx y (by simp [hy_tl])
    · rw [List.pairwise_cons]
      refine ⟨?_, ih hxs.2 (fun y hy => hx y (by simp [hy]))⟩
      intro y hy
      rw [mem_insertItemDesc] at hy
      rcases hy with heq | hy_tl
      · rw [heq]
        exact Ne.symm (hx hd (by simp))
      · exact hxs.1 y hy_tl

private theorem pairwise_sortItemsDesc (xs : List (Nat × Nat))
    (hxs : List.Pairwise (fun a b => a.1 ≠ b.1) xs) :
    List.Pairwise (fun a b => a.1 ≠ b.1) (sortItemsDesc xs) := by
  induction xs with
  | nil => simp [sortItemsDesc]
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hxs
    exact pairwise_insertItemDesc hd (sortItemsDesc tl) (ih hxs.2)
      (fun y hy => hxs.1 y ((mem_sortItemsDesc y tl).mp hy))

/-- Strong inductive invariant on each open bin constructed by `packFirstFit` / `packBFD`. -/
def WellFormedBin (binCap : Nat) (b : List PackedSlice) : Prop :=
  b ≠ [] ∧
  sumSliceLens b ≤ binCap ∧
  (∀ s ∈ b, 0 < s.len ∧ s.offset + s.len ≤ sumSliceLens b) ∧
  List.Pairwise (fun x y => x.seqId ≠ y.seqId ∧
    (x.offset + x.len ≤ y.offset ∨ y.offset + y.len ≤ x.offset)) b

private theorem wellFormedBin_gives_checkPackedBin (binCap : Nat) (b : List PackedSlice)
    (hwf : WellFormedBin binCap b) :
    checkPackedBin binCap (sumSliceLens b) b = none := by
  rcases hwf with ⟨_, htot, hslices, hpair⟩
  unfold checkPackedBin
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             List.all_eq_true, decide_eq_true_eq, allPairsDisjoint_iff_pairwise]
  refine ⟨?_, hpair⟩
  intro s hs
  have h := hslices s hs
  exact ⟨h.1, Nat.le_trans h.2 htot⟩

private theorem wellFormedBin_singleton (binCap sid len : Nat)
    (hpos : 0 < len) (hle : len ≤ binCap) :
    WellFormedBin binCap [{ seqId := sid, offset := 0, len := len }] := by
  refine ⟨by simp, by simp [hle], ?_, by simp⟩
  intro s hs
  simp at hs
  subst hs
  simp [hpos]

private theorem wellFormedBin_snoc (binCap sid len : Nat) (b : List PackedSlice)
    (hwf : WellFormedBin binCap b)
    (hpos : 0 < len)
    (hfit : sumSliceLens b + len ≤ binCap)
    (hfresh : ∀ s ∈ b, s.seqId ≠ sid) :
    WellFormedBin binCap (b ++ [{ seqId := sid, offset := sumSliceLens b, len := len }]) := by
  rcases hwf with ⟨_, _, hslices, hpair⟩
  have hsum : sumSliceLens (b ++ [{ seqId := sid, offset := sumSliceLens b, len := len }]) =
      sumSliceLens b + len := by
    simp [sumSliceLens_append]
  refine ⟨by simp, by rw [hsum]; exact hfit, ?_, ?_⟩
  · intro s hs
    rw [List.mem_append] at hs
    rcases hs with hsb | hsl
    · have hb_s := hslices s hsb
      refine ⟨hb_s.1, ?_⟩
      rw [hsum]
      omega
    · simp at hsl
      subst hsl
      refine ⟨hpos, ?_⟩
      rw [hsum]
      exact Nat.le_refl _
  · rw [List.pairwise_append]
    refine ⟨hpair, by simp, ?_⟩
    intro x hx y hy
    simp at hy
    subst hy
    exact ⟨hfresh x hx, Or.inl (hslices x hx).2⟩

private theorem bestFitBinIdx_none_iff (binCap len : Nat) (bins : List (List PackedSlice)) :
    bestFitBinIdx binCap len bins = none ↔
    ∀ b ∈ bins, binCap < sumSliceLens b + len := by
  induction bins with
  | nil => simp [bestFitBinIdx]
  | cons b bs ih =>
    unfold bestFitBinIdx
    by_cases hfit : sumSliceLens b + len ≤ binCap
    · simp only [hfit, ↓reduceIte]
      constructor
      · intro h
        cases htb : bestFitBinIdx binCap len bs with
        | none => simp [htb] at h
        | some p =>
          simp only [htb, Option.map_some] at h
          split at h <;> simp at h
      · intro hall
        have hb := hall b (by simp)
        omega
    · simp only [hfit, ↓reduceIte, Option.map_eq_none_iff, ih, List.mem_cons, forall_eq_or_imp]
      constructor
      · intro htail
        exact ⟨by omega, htail⟩
      · intro ⟨_, htail⟩
        exact htail

private theorem bestFitBinIdx_some_spec (binCap len : Nat) (bins : List (List PackedSlice))
    (idx u : Nat) (hsome : bestFitBinIdx binCap len bins = some (idx, u)) :
    idx < bins.length ∧
    (∃ b, bins[idx]? = some b ∧ sumSliceLens b = u ∧ u + len ≤ binCap) := by
  induction bins generalizing idx u with
  | nil => simp [bestFitBinIdx] at hsome
  | cons b bs ih =>
    unfold bestFitBinIdx at hsome
    by_cases hfit : sumSliceLens b + len ≤ binCap
    · simp only [hfit, ↓reduceIte] at hsome
      cases htb : bestFitBinIdx binCap len bs with
      | none =>
        simp [htb] at hsome
        rcases hsome with ⟨rfl, rfl⟩
        exact ⟨by simp, ⟨b, by simp, rfl, hfit⟩⟩
      | some p =>
        rcases p with ⟨tIdx, tUsed⟩
        simp only [htb, Option.map_some] at hsome
        split at hsome
        · simp at hsome
          rcases hsome with ⟨rfl, rfl⟩
          have ⟨hlt, ⟨b', hb', hu', hfit'⟩⟩ := ih tIdx tUsed htb
          refine ⟨by simp; omega, ⟨b', ?_, hu', hfit'⟩⟩
          simp [hb']
        · simp at hsome
          rcases hsome with ⟨rfl, rfl⟩
          exact ⟨by simp, ⟨b, by simp, rfl, hfit⟩⟩
    · simp only [hfit, ↓reduceIte, Option.map_eq_some_iff] at hsome
      rcases hsome with ⟨⟨tIdx, tUsed⟩, htb, heq⟩
      have hidx : tIdx + 1 = idx := congrArg Prod.fst heq
      have hu : tUsed = u := congrArg Prod.snd heq
      subst hidx hu
      have ⟨hlt, ⟨b', hb', hu', hfit'⟩⟩ := ih tIdx tUsed htb
      refine ⟨by simp; omega, ⟨b', ?_, hu', hfit'⟩⟩
      simp [hb']

private theorem updateBinAt_preserves (binCap sid len idx : Nat)
    (bins : List (List PackedSlice))
    (hpos : 0 < len)
    (hb_spec : ∃ b, bins[idx]? = some b ∧ sumSliceLens b + len ≤ binCap)
    (hwf : ∀ b ∈ bins, WellFormedBin binCap b)
    (hfresh : ∀ b ∈ bins, ∀ s ∈ b, s.seqId ≠ sid)
    (hfull : List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) bins) :
    let bins' := updateBinAt idx sid len bins
    bins'.length = bins.length ∧
    sumBinsTokens bins' = sumBinsTokens bins + len ∧
    (∀ b' ∈ bins', WellFormedBin binCap b') ∧
    (∀ b' ∈ bins', ∀ s ∈ b', s.seqId = sid ∨ ∃ b ∈ bins, s ∈ b) ∧
    (∀ i, i < bins.length → ∃ b b', bins[i]? = some b ∧ bins'[i]? = some b' ∧ sumSliceLens b ≤ sumSliceLens b') ∧
    List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) bins' := by
  induction bins generalizing idx with
  | nil =>
    rcases hb_spec with ⟨_, hget, _⟩
    simp at hget
  | cons hd tl ih =>
    rw [List.pairwise_cons] at hfull
    cases idx with
    | zero =>
      rcases hb_spec with ⟨b0, hget0, hfit0⟩
      simp at hget0
      subst hget0
      have hwf_hd' : WellFormedBin binCap (hd ++ [{ seqId := sid, offset := sumSliceLens hd, len := len }]) :=
        wellFormedBin_snoc binCap sid len hd (hwf hd (by simp)) hpos hfit0 (hfresh hd (by simp))
      simp only [updateBinAt]
      refine ⟨by simp, ?_, ?_, ?_, ?_, ?_⟩
      · simp [sumSliceLens_append, Nat.add_right_comm]
      · intro b' hb'
        simp only [List.mem_cons] at hb'
        rcases hb' with heq | htl
        · rw [heq]
          exact hwf_hd'
        · exact hwf b' (by simp [htl])
      · intro b' hb' s hs
        simp only [List.mem_cons] at hb'
        rcases hb' with heq | htl
        · rw [heq, List.mem_append] at hs
          rcases hs with hshd | hsl
          · exact Or.inr ⟨hd, by simp, hshd⟩
          · simp at hsl
            rw [hsl]
            exact Or.inl rfl
        · exact Or.inr ⟨b', by simp [htl], hs⟩
      · intro i hi
        cases i with
        | zero =>
          refine ⟨hd, hd ++ [{ seqId := sid, offset := sumSliceLens hd, len := len }], by simp, by simp, ?_⟩
          simp [sumSliceLens_append]
        | succ j =>
          simp only [List.length_cons] at hi
          have hj : j < tl.length := by omega
          exact ⟨tl[j], tl[j], by simp [hj], by simp [hj], Nat.le_refl _⟩
      · rw [List.pairwise_cons]
        refine ⟨?_, hfull.2⟩
        intro b2 hb2
        have h12 := hfull.1 b2 hb2
        simp only [sumSliceLens_append, sumSliceLens_cons, sumSliceLens_nil, Nat.add_zero]
        omega
    | succ k =>
      rcases hb_spec with ⟨bk, hgetk, hfitk⟩
      simp at hgetk
      have step := ih k ⟨bk, hgetk, hfitk⟩
        (fun b hb => hwf b (by simp [hb]))
        (fun b hb => hfresh b (by simp [hb]))
        hfull.2
      rcases step with ⟨hlen, hsum, hwf_tl, hmem_tl, hmono_tl, hfull_tl⟩
      simp only [updateBinAt]
      refine ⟨by simp [hlen], ?_, ?_, ?_, ?_, ?_⟩
      · simp [hsum, Nat.add_assoc]
      · intro b' hb'
        simp only [List.mem_cons] at hb'
        rcases hb' with heq | htl
        · rw [heq]
          exact hwf hd (by simp)
        · exact hwf_tl b' htl
      · intro b' hb' s hs
        simp only [List.mem_cons] at hb'
        rcases hb' with heq | htl
        · rw [heq] at hs
          exact Or.inr ⟨hd, by simp, hs⟩
        · rcases hmem_tl b' htl s hs with hsid | ⟨b0, hb0, hs0⟩
          · exact Or.inl hsid
          · exact Or.inr ⟨b0, by simp [hb0], hs0⟩
      · intro i hi
        cases i with
        | zero => exact ⟨hd, hd, by simp, by simp, Nat.le_refl _⟩
        | succ j =>
          simp only [List.length_cons] at hi
          have hj : j < tl.length := by omega
          rcases hmono_tl j hj with ⟨b0, b1, hb0, hb1, hle⟩
          exact ⟨b0, b1, by simp [hb0], by simp [hb1], hle⟩
      · rw [List.pairwise_cons]
        refine ⟨?_, hfull_tl⟩
        intro b2' hb2'
        have ⟨j, hj_lt, hj_eq⟩ := List.mem_iff_getElem.mp hb2'
        have hj_tl : j < tl.length := by omega
        rcases hmono_tl j hj_tl with ⟨b_old, b_new, hbold, hbnew, hle_load⟩
        have hb_new_eq : b_new = b2' := by
          rw [List.getElem?_eq_getElem hj_lt] at hbnew
          injection hbnew with h_eq
          exact h_eq.symm.trans hj_eq
        subst hb_new_eq
        have hmem_old : b_old ∈ tl := List.mem_of_getElem? hbold
        have h12 := hfull.1 b_old hmem_old
        omega

private theorem insertBestFit_step (binCap sid len : Nat) (bins : List (List PackedSlice))
    (hpos : 0 < len) (hle : len ≤ binCap)
    (hwf : ∀ b ∈ bins, WellFormedBin binCap b)
    (hfresh : ∀ b ∈ bins, ∀ s ∈ b, s.seqId ≠ sid)
    (hfull : List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) bins) :
    let bins' := insertBestFit binCap sid len bins
    bins' ≠ [] ∧
    sumBinsTokens bins' = sumBinsTokens bins + len ∧
    (∀ b' ∈ bins', WellFormedBin binCap b') ∧
    (∀ b' ∈ bins', ∀ s ∈ b', s.seqId = sid ∨ ∃ b ∈ bins, s ∈ b) ∧
    List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) bins' := by
  unfold insertBestFit
  cases hbf : bestFitBinIdx binCap len bins with
  | none =>
    have hnone := (bestFitBinIdx_none_iff binCap len bins).mp hbf
    have hwf_new : WellFormedBin binCap [{ seqId := sid, offset := 0, len := len }] :=
      wellFormedBin_singleton binCap sid len hpos hle
    refine ⟨by simp, ?_, ?_, ?_, ?_⟩
    · simp [sumBinsTokens_append]
    · intro b' hb'
      rw [List.mem_append] at hb'
      rcases hb' with h1 | h2
      · exact hwf b' h1
      · simp at h2
        rw [h2]
        exact hwf_new
    · intro b' hb' s hs
      rw [List.mem_append] at hb'
      rcases hb' with h1 | h2
      · exact Or.inr ⟨b', h1, hs⟩
      · simp at h2
        rw [h2] at hs
        simp at hs
        rw [hs]
        exact Or.inl rfl
    · rw [List.pairwise_append]
      refine ⟨hfull, by simp, ?_⟩
      intro b1 hb1 b2 hb2
      simp at hb2
      rw [hb2]
      have hgt := hnone b1 hb1
      simp only [sumSliceLens_cons, sumSliceLens_nil, Nat.add_zero]
      omega
  | some p =>
    rcases p with ⟨idx, u⟩
    have ⟨hlt, ⟨b0, hb0, hu0, hfit0⟩⟩ := bestFitBinIdx_some_spec binCap len bins idx u hbf
    subst hu0
    have upd := updateBinAt_preserves binCap sid len idx bins hpos ⟨b0, hb0, hfit0⟩ hwf hfresh hfull
    rcases upd with ⟨hlen, hsum, hwf', hmem', _, hfull'⟩
    have hne : updateBinAt idx sid len bins ≠ [] := by
      intro hemp
      rw [hemp] at hlen
      simp at hlen
      omega
    exact ⟨hne, hsum, hwf', hmem', hfull'⟩

private theorem foldl_insertBestFit_invariant (binCap : Nat) :
    ∀ (items : List (Nat × Nat)) (bins : List (List PackedSlice)),
      (∀ it ∈ items, 0 < it.2 ∧ it.2 ≤ binCap) →
      List.Pairwise (fun a b => a.1 ≠ b.1) items →
      (∀ b ∈ bins, WellFormedBin binCap b) →
      (∀ it ∈ items, ∀ b ∈ bins, ∀ s ∈ b, s.seqId ≠ it.1) →
      List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) bins →
      let out := items.foldl (fun acc item => insertBestFit binCap item.1 item.2 acc) bins
      (items ≠ [] ∨ bins ≠ [] → out ≠ []) ∧
      sumBinsTokens out = sumBinsTokens bins + sumItemLens items ∧
      (∀ b ∈ out, WellFormedBin binCap b) ∧
      List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) out
  | [], bins, _, _, hwf, _, hfull => by
    refine ⟨fun h => ?_, by simp, hwf, hfull⟩
    rcases h with hnil | hbins
    · exact False.elim (hnil rfl)
    · exact hbins
  | it :: rest, bins, hbounds, huniq, hwf, hdisj_ids, hfull => by
    rw [List.pairwise_cons] at huniq
    have hit := hbounds it (by simp)
    have step := insertBestFit_step binCap it.1 it.2 bins hit.1 hit.2 hwf
      (hdisj_ids it (by simp)) hfull
    rcases step with ⟨hne1, hsum1, hwf1, hmem1, hfull1⟩
    have hdisj_next : ∀ it2 ∈ rest, ∀ b' ∈ insertBestFit binCap it.1 it.2 bins, ∀ s ∈ b', s.seqId ≠ it2.1 := by
      intro it2 hit2 b' hb' s hs
      rcases hmem1 b' hb' s hs with hsid | ⟨b0, hb0, hs0⟩
      · rw [hsid]
        exact huniq.1 it2 hit2
      · exact hdisj_ids it2 (by simp [hit2]) b0 hb0 s hs0
    have ih := foldl_insertBestFit_invariant binCap rest (insertBestFit binCap it.1 it.2 bins)
      (fun x hx => hbounds x (by simp [hx]))
      huniq.2 hwf1 hdisj_next hfull1
    rcases ih with ⟨hne_out, hsum_out, hwf_out, hfull_out⟩
    simp only [List.foldl_cons]
    refine ⟨fun _ => hne_out (Or.inr hne1), ?_, hwf_out, hfull_out⟩
    rw [hsum_out, hsum1]
    simp [Nat.add_assoc]

/-- End-to-End `packBFD` Validity & Token Conservation Theorem:
    for all inputs with distinct `seqId`s and `0 < len ≤ binCap`, every bin emitted by
    `packBFD` is non-empty, passes `checkPackedBin`, and conserves `sumItemLens items`. -/
theorem packBFD_all_bins_valid (binCap : Nat) (items : List (Nat × Nat))
    (hbounds : ∀ it ∈ items, 0 < it.2 ∧ it.2 ≤ binCap)
    (huniq : List.Pairwise (fun a b => a.1 ≠ b.1) items) :
    let bins := packBFD binCap items
    (∀ b ∈ bins, b ≠ [] ∧ checkPackedBin binCap (sumSliceLens b) b = none) ∧
    sumBinsTokens bins = sumItemLens items := by
  have hbounds_s : ∀ it ∈ sortItemsDesc items, 0 < it.2 ∧ it.2 ≤ binCap :=
    fun it hit => hbounds it ((mem_sortItemsDesc it items).mp hit)
  have huniq_s : List.Pairwise (fun a b => a.1 ≠ b.1) (sortItemsDesc items) :=
    pairwise_sortItemsDesc items huniq
  have inv := foldl_insertBestFit_invariant binCap (sortItemsDesc items) []
    hbounds_s huniq_s (by simp) (by simp) (by simp)
  rcases inv with ⟨_, hsum, hwf, _⟩
  unfold packBFD
  refine ⟨?_, ?_⟩
  · intro b hb
    have hw := hwf b hb
    exact ⟨hw.1, wellFormedBin_gives_checkPackedBin binCap b hw⟩
  · rw [hsum, sumBinsTokens_nil, Nat.zero_add, sumItemLens_sortItemsDesc]

/-- `packBFD` Pairwise-Fullness Theorem:
    any two distinct bins `(i < j)` emitted by `packBFD` satisfy
    `binCap < sumSliceLens B_i + sumSliceLens B_j` (so at most one bin is `≤ binCap / 2` full). -/
theorem packBFD_pairwise_fullness (binCap : Nat) (items : List (Nat × Nat))
    (hbounds : ∀ it ∈ items, 0 < it.2 ∧ it.2 ≤ binCap)
    (huniq : List.Pairwise (fun a b => a.1 ≠ b.1) items) :
    List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2)
      (packBFD binCap items) := by
  have hbounds_s : ∀ it ∈ sortItemsDesc items, 0 < it.2 ∧ it.2 ≤ binCap :=
    fun it hit => hbounds it ((mem_sortItemsDesc it items).mp hit)
  have huniq_s : List.Pairwise (fun a b => a.1 ≠ b.1) (sortItemsDesc items) :=
    pairwise_sortItemsDesc items huniq
  have inv := foldl_insertBestFit_invariant binCap (sortItemsDesc items) []
    hbounds_s huniq_s (by simp) (by simp) (by simp)
  exact inv.2.2.2

private theorem pairwise_full_two_approx_aux (binCap : Nat) :
    ∀ (bins : List (List PackedSlice)),
      bins ≠ [] →
      (∀ b ∈ bins, 0 < sumSliceLens b) →
      List.Pairwise (fun b1 b2 => binCap < sumSliceLens b1 + sumSliceLens b2) bins →
      (bins.length - 1) * binCap < 2 * sumBinsTokens bins
  | [], hne, _, _ => False.elim (hne rfl)
  | [b1], _, hpos, _ => by
    have hb1 := hpos b1 (by simp)
    simp
    omega
  | [b1, b2], _, _, hfull => by
    rw [List.pairwise_cons] at hfull
    have h12 := hfull.1 b2 (by simp)
    simp
    omega
  | b1 :: b2 :: b3 :: rest, _, hpos, hfull => by
    rw [List.pairwise_cons] at hfull
    have h12 := hfull.1 b2 (by simp)
    have hfull_tail := hfull.2.tail
    have ih := pairwise_full_two_approx_aux binCap (b3 :: rest) (by simp)
      (fun b hb => hpos b (by simp [hb])) hfull_tail
    have hlen_eq : ((b1 :: b2 :: b3 :: rest).length - 1) * binCap =
        ((b3 :: rest).length - 1) * binCap + 2 * binCap := by
      simp [Nat.add_mul]
    rw [hlen_eq]
    simp only [sumBinsTokens_cons] at ih ⊢
    omega

/-- Strict 2-Approximation Token Bound (`packBFD_two_approx`):
    for any non-empty `items` with `0 < len ≤ binCap` and unique `seqId`s,
    `(bins.length - 1) * binCap < 2 * sumItemLens items`. -/
theorem packBFD_two_approx (binCap : Nat) (items : List (Nat × Nat))
    (hne : items ≠ [])
    (hbounds : ∀ it ∈ items, 0 < it.2 ∧ it.2 ≤ binCap)
    (huniq : List.Pairwise (fun a b => a.1 ≠ b.1) items) :
    ((packBFD binCap items).length - 1) * binCap < 2 * sumItemLens items := by
  have hne_s : sortItemsDesc items ≠ [] := by
    intro hemp
    cases items with
    | nil => exact hne rfl
    | cons x xs =>
      have hx : x ∈ sortItemsDesc (x :: xs) := (mem_sortItemsDesc x (x :: xs)).mpr (by simp)
      rw [hemp] at hx
      cases hx
  have hbounds_s : ∀ it ∈ sortItemsDesc items, 0 < it.2 ∧ it.2 ≤ binCap :=
    fun it hit => hbounds it ((mem_sortItemsDesc it items).mp hit)
  have huniq_s : List.Pairwise (fun a b => a.1 ≠ b.1) (sortItemsDesc items) :=
    pairwise_sortItemsDesc items huniq
  have inv := foldl_insertBestFit_invariant binCap (sortItemsDesc items) []
    hbounds_s huniq_s (by simp) (by simp) (by simp)
  rcases inv with ⟨hne_bins, hsum, hwf, hfull⟩
  have hpos_bins : ∀ b ∈ packBFD binCap items, 0 < sumSliceLens b := by
    intro b hb
    rcases hwf b hb with ⟨hb_ne, _, hslices, _⟩
    cases b with
    | nil => exact False.elim (hb_ne rfl)
    | cons s ss =>
      have hs := (hslices s (by simp)).1
      simp
      omega
  have hbound := pairwise_full_two_approx_aux binCap (packBFD binCap items)
    (hne_bins (Or.inl hne_s)) hpos_bins hfull
  unfold packBFD at hbound
  rw [hsum, sumBinsTokens_nil, Nat.zero_add, sumItemLens_sortItemsDesc] at hbound
  exact hbound

/-- End-to-End Optimality Ratio Theorem (`packBFD_le_two_opt`):
    compared against ANY valid bin-packing `optBins` conserving the same `sumItemLens items`,
    `packBFD` uses at most `2 * optBins.length` bins. -/
theorem packBFD_le_two_opt (binCap : Nat) (items : List (Nat × Nat))
    (optBins : List (List PackedSlice))
    (hne : items ≠ [])
    (hbounds : ∀ it ∈ items, 0 < it.2 ∧ it.2 ≤ binCap)
    (huniq : List.Pairwise (fun a b => a.1 ≠ b.1) items)
    (hopt_valid : ∀ b ∈ optBins, checkPackedBin binCap (sumSliceLens b) b = none)
    (hopt_sum : sumBinsTokens optBins = sumItemLens items) :
    (packBFD binCap items).length ≤ 2 * optBins.length := by
  have hcap_pos : 0 < binCap := by
    cases items with
    | nil => exact False.elim (hne rfl)
    | cons it rest =>
      have h := hbounds it (by simp)
      omega
  have hupper := packBFD_two_approx binCap items hne hbounds huniq
  have hlower := valid_bins_capacity_lower_bound binCap optBins hopt_valid
  rw [hopt_sum] at hlower
  have hchain : ((packBFD binCap items).length - 1) * binCap < (2 * optBins.length) * binCap := by
    calc ((packBFD binCap items).length - 1) * binCap
        < 2 * sumItemLens items := hupper
      _ ≤ 2 * (optBins.length * binCap) := Nat.mul_le_mul_left 2 hlower
      _ = (2 * optBins.length) * binCap := by rw [Nat.mul_assoc]
  have hlt : (packBFD binCap items).length - 1 < 2 * optBins.length := by
    cases Nat.lt_or_ge ((packBFD binCap items).length - 1) (2 * optBins.length) with
    | inl h => exact h
    | inr hge =>
      have hmul_ge := Nat.mul_le_mul_right binCap hge
      have := Nat.pos_iff_ne_zero.mp hcap_pos
      omega
  omega

/-! ## 4. DAPO Zero-Variance Prompt-Group Filtering -/

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

/-! ## 5. Compile-Time `#guard` Witnesses -/

-- Valid non-overlapping bin with distinct seqIds
#guard checkPackedBin 2048 1500
  [ { seqId := 1, offset := 0, len := 700 },
    { seqId := 2, offset := 700, len := 500 },
    { seqId := 3, offset := 1200, len := 300 } ] == none

-- Overlapping distinct sequences rejected
#guard (checkPackedBin 2048 1500
  [ { seqId := 1, offset := 0, len := 700 },
    { seqId := 2, offset := 650, len := 500 },
    { seqId := 3, offset := 1150, len := 350 } ]).isSome

-- Duplicate-seqId overlapping loophole rejected (#33 counterexample!)
#guard (checkPackedBin 2048 1000
  [ { seqId := 1, offset := 0, len := 600 },
    { seqId := 1, offset := 200, len := 400 } ]).isSome

-- Duplicate-seqId even on disjoint intervals rejected
#guard (checkPackedBin 2048 1000
  [ { seqId := 1, offset := 0, len := 600 },
    { seqId := 1, offset := 600, len := 400 } ]).isSome

-- Bin capacity overflow rejected
#guard (checkPackedBin 2048 2100
  [ { seqId := 1, offset := 0, len := 1200 },
    { seqId := 2, offset := 1200, len := 900 } ]).isSome

-- Token conservation mismatch rejected
#guard (checkPackedBin 2048 1400
  [ { seqId := 1, offset := 0, len := 700 },
    { seqId := 2, offset := 700, len := 500 } ]).isSome

-- Causal block-diagonal attention isolation & NaN-free pad self-attention witnesses
#guard (let s := [ { seqId := 1, offset := 0, len := 4 : PackedSlice },
                   { seqId := 2, offset := 4, len := 3 : PackedSlice } ]
        allowBlockDiagAttn true (seqIdAt s) 2 1 == true &&
        allowBlockDiagAttn true (seqIdAt s) 1 2 == false &&
        allowBlockDiagAttn false (seqIdAt s) 2 5 == false &&
        allowBlockDiagAttn true (seqIdAt s) 9 9 == true &&
        allowBlockDiagAttn true (seqIdAt s) 9 2 == false)

-- Best-Fit-Decreasing strictly beats arrival-order First-Fit (3 optimal bins vs 4 bins at binCap=10)
#guard (let items := [(0, 4), (1, 4), (2, 4), (3, 6), (4, 6), (5, 6)]
        let ffBins := packFirstFit 10 items
        let bfdBins := packBFD 10 items
        ffBins.length == 4 &&
        bfdBins.length == 3 &&
        sumBinsTokens bfdBins == 30 &&
        bfdBins.all (fun b => (checkPackedBin 10 (sumSliceLens b) b).isNone))

-- Tight (11/9)*OPT + 1 boundary witness (OPT = 3 bins of 14; BFD uses 4 <= (11*3)/9 + 1 = 4 bins)
#guard (let items := [(0, 4), (1, 5), (2, 6), (3, 4), (4, 5), (5, 6), (6, 4), (7, 5), (8, 3)]
        let bfdBins := packBFD 14 items
        bfdBins.length == 4 &&
        bfdBins.length ≤ (11 * 3) / 9 + 1 &&
        sumBinsTokens bfdBins == 42 &&
        bfdBins.all (fun b => (checkPackedBin 14 (sumSliceLens b) b).isNone))

-- DAPO zero-variance frontier filter witnesses
#guard isInformativePromptGroup { groupId := 1, numRollouts := 8, numPassed := 0, tokenLen := 512 } == false
#guard isInformativePromptGroup { groupId := 2, numRollouts := 8, numPassed := 8, tokenLen := 512 } == false
#guard isInformativePromptGroup { groupId := 3, numRollouts := 8, numPassed := 4, tokenLen := 512 } == true
#guard frontierVarianceWeight { groupId := 3, numRollouts := 8, numPassed := 4, tokenLen := 512 } == 16

end LabFillLean
