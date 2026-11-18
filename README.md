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
    python benchmarks/run_all.py --fresh   # re-measures; writes benchmarks/results/

## Measured results (CPU only)

Measured on 2026-11-18: Intel Xeon, 8 cores (AVX-512, AMX-BF16), PyTorch 2.15.0 CPU, Python 3.11.
The machine was shared and busy (load average about 13-17 on 8 cores at the start of each recipe),
so these runs use **one PyTorch thread** (`--threads 1`).
An earlier run with 8 PyTorch threads is kept in `benchmarks/results/cpu_2026-11-18.md`. It started
at a 1-minute load average of 38.7-45.3, versus 12.9-17.0 for the 1-thread run. Under that load even
the plain-Python checker, which does not use PyTorch threads, was 4.8x slower at B=1024
(50,825.0 vs 244,011.1 proto/s) and 3.6x slower at B=16384 (50,326.3 vs 182,366.9 proto/s).
So the two runs differ mainly in machine load and say nothing about thread scaling; the effect of
the thread count is not measured. Per-recipe thread counts and load averages are in
`machine_per_recipe` in each JSON file.
Full data: `benchmarks/results/cpu_2026-11-18_t1.json`. GPU columns: **not yet measured**.

Reproduce: `python benchmarks/run_all.py --fresh --threads 1 --tag t1`

`benchmarks/results/parts/` (8 threads) and `benchmarks/results/parts_t1/` (1 thread) are the
cached per-recipe parts behind the published reports. Without `--fresh` the runner reuses them and
only re-assembles the report; with `--fresh` it re-measures and overwrites them. The runner refuses
to assemble a report from parts measured with different thread counts.

### Recipe 1: Batched Protocol Validation (T=24 steps, C=8 containers)

| Batch B | Plain Python (proto/s) | Batched Torch CPU (proto/s) | Torch CPU / Python | Batched Torch GPU (proto/s) |
|---:|---:|---:|---:|---|
| 1 | 192,049.2 | 153.3 | 0.00x | not yet measured |
| 64 | 235,991.7 | 8,848.1 | 0.04x | not yet measured |
| 1024 | 244,011.1 | 104,513.6 | 0.43x | not yet measured |
| 16384 | 182,366.9 | 225,851.2 | 1.24x | not yet measured |

### Recipe 2: Seeded-Error Twins, Generate/Check/Keep, and Batched Repair

- **False-positive rate on valid protocols**: `0.0%`
- **Generate-N / Check / Keep (`N=4096`)**: kept `77.4%` at `163,078.0` candidates/s on CPU (GPU: `not yet measured`)

| Error Type | Detection Recall | Repair Rate (K=24) | Repair CPU (proto/s) | Repair GPU (proto/s) |
|---|---:|---:|---:|---|
| `overflow` | 100.0% | 68.8% | 4,840.7 | not yet measured |
| `wrong_pipette` | 100.0% | 100.0% | 4,548.4 | not yet measured |
| `missing_tip` | 100.0% | 100.0% | 4,339.0 | not yet measured |
| `skip_thaw` | 100.0% | 100.0% | 4,496.0 | not yet measured |
| `sealed_plate` | 100.0% | 100.0% | 4,376.5 | not yet measured |
| `bad_temp` | 100.0% | 100.0% | 4,752.8 | not yet measured |
| `swapped_steps` | 100.0% | 100.0% | 4,770.7 | not yet measured |

### Recipe 3: Batched Step-DAG Planning & Instrument Makespan Scheduling

- **DAG Beam Search (`S=16`, `beam=32`)**: random topological pass rate `1.6%` vs checker-gated beam found=`True` (`735` candidates checked in `43.07` ms CPU; GPU: `not yet measured`)

| Scheduler (96 jobs, 8 instruments) | Makespan (s) | CPU Time (ms) | GPU Time (ms) |
|---|---:|---:|---|
| `greedy_arrival` | 2859.0 | 0.09 | not yet measured |
| `greedy_lpt` | 2696.0 | 0.09 | not yet measured |
| `batched_sa_r256` | 2688.0 | 43.05 | not yet measured |

### Recipe 4: Fill One GPU Ablation (4-layer d=256 Transformer, lognormal 8..512 tokens)

| Config | Real tok/s (CPU) | Useful-token % | Step Time (ms) | Data-wait % | Speedup vs (a) | MFU | GPU tok/s | Notes |
|---|---:|---:|---:|---:|---:|---|---|---|
| `a_padded_baseline` | 1,509.9 | 30.6% | 652.9 | 0.0% | 1.00x | N/A | not yet measured | ok |
| `b_length_bucketed` | 3,150.4 | 58.9% | 160.6 | 0.0% | 2.09x | N/A | not yet measured | ok |
| `c_sequence_packed` | 3,931.5 | 92.1% | 237.1 | 0.0% | 2.60x | N/A | not yet measured | ok |
| `d_varlen_nested_sdpa` | 2,060.8 | 100.0% | 446.8 | 0.0% | 1.36x | N/A | not yet measured | PyTorch jagged nested_tensor + SDPA (is_causal=False) |
| `e_packed_bf16_autocast` | 8,180.5 | 92.1% | 113.7 | 0.0% | 5.42x | N/A | not yet measured | ok |
| `f_packed_torch_compile` | — | — | — | — | — | N/A | not yet measured | `failed (InductorError)`: InvalidCxxCompiler: No working C++ compiler found in torch._inductor.config.cpp.cxx: (None, 'g++') |
| `g1_sync_slow_tokenizer` | 4,957.9 | 78.5% | 190.2 | 25.7% | 3.28x | N/A | not yet measured | num_workers=0 with CPU tokenizer work |
| `g2_async_dataloader` | 5,574.6 | 78.5% | 158.1 | 10.0% | 3.69x | N/A | not yet measured | num_workers=2 (spawn), prefetch_factor=2, pin_memory=(cuda) |
| `h_block_sparse_bf16` | 7,428.2 | 92.1% | 119.1 | 0.0% | 4.92x | N/A | not yet measured | Tiled W=64+G=4 (NotImplementedError: FlexAttention does not support backward on CPU.) |
| `i_cuda_graphs` | — | — | — | — | — | N/A | not yet measured | `GPU only, not yet measured`: GPU only, not yet measured |

### Recipe 5: Batched 96-Well Plate Rehearsal

| Batch Plates | Steps / Plate | Volume Conserved | CPU (plates/s) | CPU Time (ms) | GPU (plates/s) |
|---:|---:|---|---:|---:|---|
| 128 | 104 | `True` | 3,823.2 | 33.48 | not yet measured |

### What helped most in the "fill one GPU" ablation (CPU, one thread)

- **Stop computing on padding.** Padded batches spend only 30.6% of token slots on real tokens.
  Length buckets raise this to 58.9% (2.09x real tokens/s) and packing to 92.1% (2.60x).
  The useful-token fraction does not depend on the hardware, so it carries over to a GPU. The
  tokens/s speedups were measured on CPU only; on a GPU they are not yet measured.
- **bf16 autocast on top of packing** gave the largest single jump: 2.08x over fp32 packing
  (8,180.5 vs 3,931.5 real tok/s). Packing and bf16 together give 5.42x over the padded fp32
  baseline. This CPU has AMX-BF16 units; the gain on a GPU will be different and is not yet measured.
- **Async data loading** cut data-wait time from 25.7% to 10.0% of the step with a slow tokenizer
  (3.28x to 3.69x vs baseline). On a GPU, where compute is faster, data wait is a larger share,
  so this fix likely matters more there (not yet measured).
- **Did not help here:** nested-tensor varlen attention (100% useful tokens but only 1.36x; the
  CPU nested path is slow), and block-sparse local+global attention (4.92x, below dense packed
  bf16 at 5.42x, because most sequences are short). FlexAttention backward is not supported on CPU.
- **Not run:** torch.compile (no C++ compiler on this machine) and CUDA Graphs (GPU only).
- MFU is N/A on CPU because no peak FLOP/s was given. Pass a peak to get MFU on a GPU.

### Other honest notes

- Recipe 1: on one CPU thread, the batched checker only beats a plain Python loop at large
  batches (1.24x at 16,384 protocols). The batched form is built for a GPU; GPU numbers are
  not yet measured.
- Recipe 2: the twins are built against the same rules the checker follows, so 100% recall
  is a self-consistency check, not real-world error coverage. False positives on valid
  protocols: 0.0%.
- Recipe 3: 256 parallel annealing restarts beat the best greedy schedule by 0.3%
  (2,688 s vs 2,696 s makespan). Use a real solver for real scheduling problems.


## What this is not

- Not a replacement for a proven checker. The LabLean checker in
  [YLM](https://github.com/yaniguan/YLM) (Lean 4) stays the source of truth.
  The batched checker here is a fast pre-filter that follows a subset of the same rules.
- Not a physics or robot simulator.
- Not a MILP or CP solver. For large scheduling problems use a real solver.

## Design and roadmap

See [docs/DESIGN.md](docs/DESIGN.md). Next steps: run every recipe on one GPU and fill the
GPU columns; a JAX backend; a LabLean round trip (export to Lean, check both checkers agree);
generate/check/keep with a real LLM sampler; real solvers for large schedules; real protocol
datasets after a license check.

## Related tools

See [docs/related_work.md](docs/related_work.md). Tool names are used only to point to related work; no affiliation.

## License

MIT
