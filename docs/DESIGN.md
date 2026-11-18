# Design

## One idea

Most lab checks are small: one protocol, a few dozen steps, a few containers.
One small check cannot use a GPU. Ten thousand of them, written as tensor ops, can.
So every recipe here takes a **batch** of lab objects and works on all of them at once.

## Core data: a protocol as tensors

A batch of `B` protocols, each with up to `T` steps:

- `op` (B, T): op code (pick_tip, drop_tip, transfer, mix, incubate, thaw, seal, unseal, centrifuge, noop)
- `src`, `dst` (B, T): container ids
- `vol` (B, T): volume in 0.1 uL integer units (same unit as LabLean)
- `temp`, `duration`, `rpm`, `pipette` (B, T)
- padding mask (B, T)

Container state (B, C): volume, capacity, sealed, frozen, reagent bit sets.
The checker scans over `T` and updates all `B` states in one step. It returns,
per protocol: ok, first error step, error code, final state.

## Recipes

1. **Batched validation.** Pipette ranges (P2/P20/P200/P1000), overflow, enough source
   liquid, tip held, tip contamination, thaw before use, no pipetting into a sealed plate,
   incubate 4..98 C, centrifuge balance and speed. A plain-Python checker gives the same
   answers; tests compare both on thousands of random protocols.
2. **Seeded-error twins and verifier-gated generation.** Take a valid protocol, inject one
   known error (overflow, wrong pipette, missing tip, skipped thaw, sealed plate, bad
   temperature, swapped steps). Use twins to test checkers and to make negative training
   examples. Generate N candidates, check them in one batch, keep the valid ones. Repair:
   try K single-step edits of a broken protocol in one batch and keep the ones that pass.
3. **Planning and scheduling.** Beam search over step orders that respect a step DAG, with
   the batched checker as the gate. Instrument scheduling with many random-restart local
   searches run in parallel, compared with greedy list scheduling. This is a baseline;
   real solvers are better for large problems.
4. **Fill one GPU.** A realistic training loop for protocol tasks: a small transformer on
   synthetic protocol token sequences with very uneven lengths (8..512 tokens). A profiler
   harness reports real tokens/s, useful-token fraction, data-wait time, analytic FLOPs
   (useful vs executed) and MFU when a peak is given. Toggles: padding (baseline), length
   buckets, packing with a block-diagonal mask and reset position ids, varlen attention
   with nested tensors, bf16 autocast, torch.compile, async data loading, block-sparse
   local+global attention, and CUDA Graphs (GPU only).
5. **Plate rehearsal.** Dry-run many 96-well plate protocols and check final volumes and
   volume conservation.

## How it fits with other work in this account

- [YLM](https://github.com/yaniguan/YLM) has LabLean, a Lean 4 protocol language with a
  proven checker. labfill follows a subset of its rules as a fast pre-filter; LabLean
  stays the source of truth. Planned: export a labfill protocol to LabLean and back.
- YLM also has a single-GPU saturation plan and protocol-QA eval (open PRs). Recipe 4 is a
  standalone, measured version of the same question for protocol-task training.
- [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain) holds post-training
  code (SFT, DPO, GRPO) and lab checkers. Recipe 2 is the "generate, check, keep" step that
  feeds verified data into such training.

## Honest limits

- Synthetic protocols only so far. No real protocol dataset is used yet.
- The twins are built against the same rules the checker follows, so 100% recall on twins
  checks self-consistency, not real-world error coverage.
- CPU numbers were measured on a shared machine; GPU numbers are not yet measured.

## Roadmap

1. Run all benchmarks on one GPU; fill the GPU columns.
2. JAX backend (`vmap` + `jit`) for the checker and plate rehearsal.
3. LabLean round trip: export protocols to Lean, check that both checkers agree.
4. Generate/check/keep with a real LLM sampler (batched inference) in the loop.
5. Hand large scheduling problems to a real LP/MILP/CP solver; keep ours as a baseline.
6. Real protocol datasets, with a license check before use (some are non-commercial).
