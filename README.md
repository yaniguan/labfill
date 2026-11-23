# labfill

Fill the plate, fill the GPU.

GPU math libraries are fast because they batch many small problems into one tensor op.
labfill tries the same idea for lab work: protocols, steps, error checks, planning,
scheduling, and training small models on protocol tasks.

It is a set of short **recipes**: runnable code, a test, and a measured benchmark for each.
Everything runs on CPU. If a GPU is present, the same code uses it (`cuda` if available, else `cpu`).

Status: early. All numbers below were measured on CPU. GPU numbers are **not yet measured**.

Part of **[Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab)** ([Project Studio](https://yaniguan.github.io/auto-lab/)) — **Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput`**.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full code-level architecture, benchmark tables (`main` vs open PR branches), and commit history are documented in **[`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md)**:

- **Object**: Batches small wet-lab protocol checks, error-twin generation/repair, DAG beam planning, 96-well rehearsals, and variable-length Transformer training into unit-stride PyTorch tensor ops (`cuda` if available, else `cpu`).
- **Data**: Deterministic MIT-licensed synthetic protocol suites (`0.1 uL` integer volumes matching LabLean): PCR master-mix (`C=13, T=46`), random/corrupted batches, 7-class error twins, 16-step precedence DAG, 96-job heterogeneous schedules, lognormal `8..512` token traces, and 128-plate 96-well serial dilutions.
- **Model**: Sync-free `(T, B)` state-machine checker (16 error codes), `M x K` (`K=24`) single-edit repair expander, bitmask DAG beam search + `R`-restart simulated annealing, 4-layer `d=256` `ProtocolTransformer` (dense packed, jagged `nested_tensor` SDPA, tiled `W=64+G=4` block-sparse), and `float32` plate solute advection.
- **Training**: 10-point single-GPU saturation ablation (`a_padded_baseline` through `i_cuda_graphs`) with PaLM Appendix B useful-vs-executed FLOP accounting (`5.42x` CPU real-token speedup from sequence packing + `bf16` autocast on open PR `origin/cpu-benchmark-results`; GPU runs **not yet measured**).
- **History**: Initial release `ae91eb3` on `main`, plus open PR branches `origin/cpu-benchmark-results` (`d1a7510`, 1-thread & 8-thread Xeon CPU benchmarks and `docs/DESIGN.md`) and `origin/review-fixes-nov18` (`55eb158`, block-sparse key deduplication, intent-preserving repair accounting, full Python checker final state).
- **Unique Properties**: Exact `0.1 uL` volume and `A1..H12` well-index alignment with [`YLM`](https://github.com/yaniguan/YLM) LabLean, zero cross-sequence attention leakage under packing, and direct verifier-filtering interfaces for [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), [`twinstep`](https://github.com/yaniguan/twinstep), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), [`lab-verify`](https://github.com/yaniguan/lab-verify), [`labshift`](https://github.com/yaniguan/labshift), and [`stepbot`](https://github.com/yaniguan/stepbot).

## Repository Layout

```text
labfill/
├── labfill/
│   ├── __init__.py          # Public API exports
│   ├── protocol.py          # LabStateBatch, ProtocolBatch, PCR & random generators (0.1 uL units)
│   ├── checker.py           # Batched PyTorch checker & plain-Python reference checker
│   ├── twins.py             # 7 seeded-error twins, generate/check/keep, batched K-edit repair
│   ├── planner.py           # Checker-gated DAG beam search & parallel SA instrument scheduler
│   ├── fill_gpu.py          # ProtocolTransformer, packing/varlen/block-sparse/CUDA-Graph ablation
│   └── plate.py             # Batched 96-well plate rehearsal & solute concentration tracking
├── recipes/                 # Runnable end-to-end recipes (01..05)
├── benchmarks/run_all.py    # Reproducible JSON + Markdown benchmark harness
├── docs/
│   ├── ARCHITECTURE_AND_HISTORY.md
│   └── related_work.md
└── tests/                   # Differential checker, twin/repair, DAG/SA, packing, and plate tests
```

## Recipes

| # | Recipe | When to use |
|---|--------|-------------|
| 1 | Batched protocol validation | Check thousands of protocols at once for volume, pipette, tip, thaw, seal, temperature and centrifuge rules |
| 2 | Seeded-error twins, generate/check/keep, batched repair | Make known-bad copies of good protocols, filter generated candidates, try many single-edit fixes in one batch |
| 3 | Batched planning and scheduling | Beam search over step orders under a step DAG; many-restart local search for instrument schedules |
| 4 | Fill one GPU | Profile and fix low utilization when training on short, variable-length protocol sequences |
| 5 | Batched 96-well plate rehearsal | Dry-run many plate protocols and read final well volumes |

In recipe 2, the repair rate counts only edits that keep every transfer and mix step (volumes are reduced only as far as needed to fit); fixes that delete or redirect a transfer are reported separately and not counted.

## Quickstart

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e ".[dev]"
    pytest
    python recipes/01_batched_validation.py
    python recipes/04_fill_one_gpu.py
    python benchmarks/run_all.py        # writes benchmarks/results/

Results are being measured; tables will be added here.

## What this is not

- Not a replacement for a proven checker. The LabLean checker in
  [YLM](https://github.com/yaniguan/YLM) (Lean 4) stays the source of truth.
  The batched checker here is a fast pre-filter that follows a subset of the same rules.
- Not a physics or robot simulator.
- Not a MILP or CP solver. For large scheduling problems use a real solver.

## Related tools

See [docs/related_work.md](docs/related_work.md). Tool names are used only to point to related work; no affiliation.

## License

MIT
