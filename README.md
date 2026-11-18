# labfill

Fill the plate, fill the GPU.

GPU math libraries are fast because they batch many small problems into one tensor op.
labfill tries the same idea for lab work: protocols, steps, error checks, planning,
scheduling, and training small models on protocol tasks.

It is a set of short **recipes**: runnable code, a test, and a measured benchmark for each.
Everything runs on CPU. If a GPU is present, the same code uses it (`cuda` if available, else `cpu`).

Status: early. All numbers below were measured on CPU. GPU numbers are **not yet measured**.

## Recipes

| # | Recipe | When to use |
|---|--------|-------------|
| 1 | Batched protocol validation | Check thousands of protocols at once for volume, pipette, tip, thaw, seal, temperature and centrifuge rules |
| 2 | Seeded-error twins, generate/check/keep, batched repair | Make known-bad copies of good protocols, filter generated candidates, try many single-edit fixes in one batch |
| 3 | Batched planning and scheduling | Beam search over step orders under a step DAG; many-restart local search for instrument schedules |
| 4 | Fill one GPU | Profile and fix low utilization when training on short, variable-length protocol sequences |
| 5 | Batched 96-well plate rehearsal | Dry-run many plate protocols and read final well volumes |

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
