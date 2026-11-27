# labfill

Fill the plate, fill the GPU.

GPU math libraries are fast because they batch many small problems into one tensor op.
`labfill` applies the same idea to wet-lab automation: protocols, step preconditions, error twins,
single-edit repair, DAG planning, scheduling, and variable-length Transformer training.

It is a set of short **recipes**: runnable code, tests, and reproducible benchmarks for each.
Everything runs on CPU. If a GPU is present, the same code uses it (`cuda` if available, else `cpu`).

Status: CPU benchmarks measured (`benchmarks/results/cpu_2026-11-26_main_v2.json`). Physical single-A100/H100 GPU numbers are **`PENDING_GPU`** (ready-to-launch harness in `benchmarks/run_gpu_flash_flex.py`; Issue #35 stays open until run on physical GPU hardware).

Part of **[Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab)** ([Project Studio](https://yaniguan.github.io/auto-lab/)) — **Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput`**.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full code-level architecture, rule-set delta allowlists, benchmark tables, and commit history are documented in **[`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md)**:

- **Object**: Batches wet-lab protocol checks, error-twin generation/repair, DAG beam planning, 96-well rehearsals, and variable-length Transformer training into unit-stride PyTorch tensor ops (`cuda` if available, else `cpu`).
- **Data**: Deterministic MIT-licensed synthetic protocol suites (`0.1 uL` integer volumes matching LabLean, supporting container volumes/capacities up to `MAX_CONTAINER_VOL_TENTHS_UL = 100,000,000`, i.e., `10,000 mL` / `10.0 L`) plus an open-corpora evaluation pipeline (`labfill/open_corpora.py`, `scripts/fetch_open_corpora.py`) over **5 real public datasets**: `Opentrons/shared-data`, `WLP`, `BioPlanner`, `SMolInstruct`, and `ORD` (cached in gitignored `data/`).
- **Model**: Sync-free `(T, B)` state-machine checker (16 error codes, 120-bit limb-splitting back-transfer concentration check with zero `int64` overflow), `M x K` (`K=26`) single-edit repair expander, bitmask DAG beam search + `R`-restart simulated annealing, Ding et al. (2024) **`O(N log N)` Segment-Tree Best-Fit-Decreasing (`pack_sequences_obfd`)**, 4-layer `d=256` `ProtocolTransformer` (dense packed, jagged `nested_tensor` / `flash_attn_varlen_func`, tiled `W=64+G=4` & compiled `FlexAttention`), and 96-well solute advection.
- **Training**: Single-GPU saturation ablation (`a_padded_baseline` through `i_cuda_graphs`) reported as **median and `[min, max]` spread over 5 seeds** with PaLM Appendix B useful-vs-executed FLOP accounting (`PENDING_GPU` for physical A100/H100 runs via `benchmarks/run_gpu_flash_flex.py`).
- **LabLean Live Conformance**: Bidirectional `.lean` / JSON AST exporter (`labfill/lablean_io.py`) continuously compiled and checked in CI against [`YLM`](https://github.com/yaniguan/YLM) LabLean (`55e817b`, Lean 4.36.0) across **310 protocols**: **0 compile errors**, **105/105** valid protocols accepted by both, **105/105** error twins rejected with the exact same error code, **98/100** random corrupted protocols rejected by both (`85` identical error code), and **2/310** accounted for by the 4 documented rule-set deltas (`unseal`, per-container vs per-reagent `thaw`, high-temp seal / positive duration & RPM, and `5 uL` 2-slot volume vs `100 mg` `n`-slot rotor balance).
- **Unique Properties**: Exact `0.1 uL` volume and `A1..H12` well-index alignment with [`YLM`](https://github.com/yaniguan/YLM) LabLean, exact compact `step_index` & `twinstep_step_index` error localization, zero cross-sequence attention leakage under packing, and direct verifier-filtering interfaces for [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), [`twinstep`](https://github.com/yaniguan/twinstep), [`lab-attrib`](https://github.com/yaniguan/lab-attrib), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), [`lab-verify`](https://github.com/yaniguan/lab-verify), [`labshift`](https://github.com/yaniguan/labshift), and [`stepbot`](https://github.com/yaniguan/stepbot).

## 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "labfill"`, `LAYER = "L3"`) implements `process_packet(packet: dict) -> dict` for the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) closed-loop wire protocol:

- **Upstream (`twinstep` & `lab-attrib`)**: Accepts `error_twins.token_lengths` (plus optional `prompt_lengths`, `pass_rates` / `rollout_rewards`, and `drop_zero_variance`) from [`twinstep`](https://github.com/yaniguan/twinstep) alongside post-execution causal blame and certified provenance from [`lab-attrib`](https://github.com/yaniguan/lab-attrib). Packets already marked `halted=True` pass through unchanged.
- **Layer `L3` Packing & Formal Guard (`LabFillLean`)**: Packs variable-length protocol/twin sequences into fixed-capacity BF16 bins (`bin_cap`, default `512`, auto-aligning implicit capacity up to `2048` / `8192` on longer protocols and supporting `overflow_policy="halt" | "quarantine_oversized" | "promote_bin_cap" | "chunk"`) using `labfill.fill_gpu.pack_sequence_pairs`, O(`bin_cap`) `build_cu_seqlens` boundary verification, and `materialize_bf16_packed_batch` (with an exact pure-Python fallback if PyTorch is absent). Verifies the `LabFillLean` invariants (`0 < len`, `offset + len <= bin_cap`, pairwise distinct `seqId`s and non-overlapping intervals, and `sum(packed_slices) == sum(token_lengths)`), halting on any overlap or conservation mismatch with `halt_reason = "L3_LABFILL_PACK_BLOCK"` (`bin_count = 0`, `useful_token_ratio = 0.0`).
- **Downstream (`labbench-posttrain`)**: Writes `packed_batch` (`bin_cap`, `bin_count`, `total_real_tokens`, `useful_token_ratio`, `zero_cross_seq_overlap`, `bins`, `cu_seqlens`, `bin_cu_seqlens`, `max_seqlen`, `loss_mask_spans`, `total_completion_tokens`), exposes `packed_batch_to_posttrain_tensors(...)`, and appends `trace` and `bottom_reasoning_traces` for [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) `SplitGuard` and SFT/MPO/DPO/GRPO/GDPO post-training.

## Repository Layout

```text
labfill/
├── autolab_wire.py                  # 15-repo Auto-Lab wire node (autolab-wire/v1, Layer L3)
├── labfill/
│   ├── __init__.py                  # Public API exports
│   ├── protocol.py                  # LabStateBatch (<= 10,000 mL), ProtocolBatch, PCR & random generators
│   ├── checker.py                   # 120-bit overflow-free batched PyTorch checker & Python reference
│   ├── lablean_io.py                # Strict YLM LabLean (.lean / JSON AST) exporter & error bridge
│   ├── twins.py                     # 7 seeded-error twins, K-edit repair & compact step_index JSONL bridge
│   ├── planner.py                   # Checker-gated DAG beam search & parallel SA instrument scheduler
│   ├── fill_gpu.py                  # ProtocolTransformer, O(N log N) OBFD packing, varlen/FlexAttn/CUDA-Graph
│   ├── open_corpora.py              # Real public protocol tokenization, OBFD packing & checker evaluation
│   └── plate.py                     # Batched 96-well plate rehearsal & solute concentration tracking
├── scripts/
│   └── fetch_open_corpora.py        # Fetch Opentrons, WLP, BioPlanner, SMolInstruct & ORD into data/
├── community_extensions/            # DAPO zero-variance filtering & OBFD curriculum BF16 bin packer
├── lean/LabFillLean.lean            # Lean 4 proofs for packed-bin non-overlap & token conservation
├── recipes/                         # Runnable end-to-end recipes (01..05)
├── benchmarks/
│   ├── run_all.py                   # Reproducible 5-seed JSON + Markdown benchmark harness
│   └── run_gpu_flash_flex.py        # Ready-to-launch A100/H100 FlashAttn/FlexAttn/OBFD runner (PENDING_GPU)
├── docs/
│   ├── ARCHITECTURE_AND_HISTORY.md
│   ├── DESIGN.md
│   └── related_work.md
└── tests/                           # 12k-pair reservoir diff, live LabLean 310-proto parity, twins, OBFD & drift tests
```

## Recipes

| # | Recipe | When to use |
|---|--------|-------------|
| 1 | Batched protocol validation | Check thousands of protocols at once (volumes up to `10,000 mL` / `MAX_CONTAINER_VOL_TENTHS_UL = 100,000,000`) |
| 2 | Seeded-error twins, generate/check/keep, batched repair | Make known-bad copies of good protocols, filter generated candidates, try `K=26` single-edit fixes in one batch |
| 3 | Batched planning and scheduling | Beam search over step orders under a step DAG; many-restart local search for instrument schedules |
| 4 | Fill one GPU | Profile and fix low utilization on variable-length protocol sequences (`O(N log N)` OBFD packing, BF16, async loader) |
| 5 | Batched 96-well plate rehearsal | Dry-run many plate protocols and read final well volumes |

## Container Volume Bounds (`Issue #37`)

`LabStateBatch` stores volumes and capacities in exact `0.1 uL` `int64` units up to **`MAX_CONTAINER_VOL_TENTHS_UL = 100,000_000` (`10,000 mL` / `10.0 L`)**, covering standard microplates (`0.2–2 mL`), conical tubes (`15–50 mL`), deck reservoirs (`195–300 mL`), and multi-liter carboys (`10 L`). Passing any volume or capacity above `100,000,000` raises a `ValueError` in `LabStateBatch.__post_init__`. Back-transfer concentration comparisons (`1000 * src_amt * tip_min_vol > (1000 + CONC_TOL_PERMILLE) * tip_min_amt * src_vol`) use exact 120-bit unsigned limb splitting (`_mul_u120_limbs` in `labfill/checker.py`), eliminating `int64` overflow with zero floating-point rounding.

## Quickstart

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -e ".[dev]"
    pytest -p no:cacheprovider -v
    python recipes/01_batched_validation.py
    python recipes/04_fill_one_gpu.py
    python scripts/fetch_open_corpora.py                         # fetches 5 public corpora into gitignored data/
    python benchmarks/run_all.py --fresh --threads 1 --tag main_v2
    python benchmarks/run_gpu_flash_flex.py --tag a100           # run on A100/H100 (currently PENDING_GPU)

## Measured Results (CPU Measured · GPU `PENDING_GPU`)

Measured on `2026-11-26`: 16-core AuthenticAMD (`AMD64 Family 26 Model 96 Stepping 0`), Python `3.11.15`, PyTorch `2.14.0+cpu`, NumPy `2.4.6`, **1 PyTorch thread** (`--threads 1`).
Full data: `benchmarks/results/cpu_2026-11-26_main_v2.json` (historical Intel Xeon AMX-BF16 runs remain in `benchmarks/results/cpu_2026-11-18_t1.json` and `cpu_2026-11-18.json`).
All GPU columns below are **`PENDING_GPU`** (`not yet measured` in JSON until `benchmarks/run_gpu_flash_flex.py` is executed on physical A100/H100 hardware).

Reproduce: `python benchmarks/run_all.py --fresh --threads 1 --tag main_v2`

### Recipe 1: Batched Protocol Validation (T=24 steps, C=8 containers)

| Batch B | Plain Python (proto/s) | Batched Torch CPU (proto/s) | Torch CPU / Python | Batched Torch GPU (proto/s) |
|---:|---:|---:|---:|---|
| 1 | 29,673.6 | 16.9 | 0.00x | not yet measured (`PENDING_GPU`) |
| 64 | 25,601.0 | 849.4 | 0.03x | not yet measured (`PENDING_GPU`) |
| 1024 | 34,924.8 | 11,402.2 | 0.33x | not yet measured (`PENDING_GPU`) |
| 16384 | 23,109.4 | 26,928.9 | 1.17x | not yet measured (`PENDING_GPU`) |

### Recipe 2: Seeded-Error Twins, Generate/Check/Keep, and Batched Repair

- **False-positive rate on valid protocols**: `0.0%`
- **Generate-N / Check / Keep (`N=4096`)**: kept `77.1%` at `17,076.4` candidates/s on CPU (GPU: `PENDING_GPU`)

`Passes Checker (1 Edit)` counts only intent-preserving single edits (keeping every transfer, mix, incubate, thaw, and centrifuge step); `Exact Recovery` measures exact equality with the pre-corruption protocol, and `Mean Changed Fields` reports mean step-field edit distance (`0.00` = identical protocol).

| Error Type | Detection Recall | Passes Checker (1 Edit) | Exact Recovery | Mean Changed Fields | Winning Edits | Repair CPU (proto/s) | Repair GPU (proto/s) |
|---|---:|---:|---:|---:|---|---:|---|
| `overflow` | 100.0% | 100.0% | 0.0% | 1.44 | clamp_volume: 128 | 366.9 | not yet measured (`PENDING_GPU`) |
| `wrong_pipette` | 100.0% | 100.0% | 100.0% | 0.00 | set_pipette: 128 | 402.3 | not yet measured (`PENDING_GPU`) |
| `missing_tip` | 100.0% | 100.0% | 100.0% | 0.00 | insert_step: 128 | 355.1 | not yet measured (`PENDING_GPU`) |
| `skip_thaw` | 100.0% | 100.0% | 32.8% | 0.67 | insert_step: 128 | 343.0 | not yet measured (`PENDING_GPU`) |
| `sealed_plate` | 100.0% | 100.0% | 100.0% | 0.00 | delete_prior_seal: 128 | 345.3 | not yet measured (`PENDING_GPU`) |
| `bad_temp` | 100.0% | 100.0% | 0.8% | 1.12 | fix_incubate: 128 | 550.9 | not yet measured (`PENDING_GPU`) |
| `swapped_steps` | 100.0% | 100.0% | 100.0% | 0.00 | swap_adjacent: 128 | 344.1 | not yet measured (`PENDING_GPU`) |

### Recipe 3: Batched Step-DAG Planning & Instrument Makespan Scheduling

- **DAG Beam Search (`S=16`, `beam=32`)**: random topological pass rate `1.6%` vs checker-gated beam found=`True` (`735` candidates checked in `577.48` ms CPU; GPU: `PENDING_GPU`)

| Scheduler (96 jobs, 8 instruments) | Makespan (s) | CPU Time (ms) | GPU Time (ms) |
|---|---:|---:|---|
| `greedy_arrival` | 2859.0 | 0.44 | not yet measured (`PENDING_GPU`) |
| `greedy_lpt` | 2696.0 | 0.28 | not yet measured (`PENDING_GPU`) |
| `batched_sa_r256` | 2688.0 | 184.83 | not yet measured (`PENDING_GPU`) |

### Recipe 4: Fill One GPU Ablation (4-layer d=256 Transformer, lognormal 8..512 tokens, 5 seeds)

Speedups report **median and `[min, max]` across 5 seeds (`0..4`)**. Claims below compare two configurations only when their 5-seed `[min, max]` intervals do not overlap.

| Config | Real tok/s (CPU) | Useful-token % | Step Time (ms) | Data-wait % | Speedup vs (a) [min, max] | MFU | GPU tok/s | Notes |
|---|---:|---:|---:|---:|---|---|---|---|
| `a_padded_baseline` | 438.5 | 25.0% | 1948.5 | 0.0% | 1.00x [1.00x, 1.00x] | N/A | PENDING_GPU | ok |
| `b_length_bucketed` | 1,031.7 | 58.3% | 411.4 | 0.0% | 2.09x [1.76x, 2.85x] | N/A | PENDING_GPU | ok |
| `c_sequence_packed` | 1,337.5 | 92.4% | 697.2 | 0.0% | 3.05x [2.37x, 3.51x] | N/A | PENDING_GPU | ok |
| `d_varlen_nested_sdpa` | 768.3 | 100.0% | 1033.4 | 0.0% | 1.71x [1.54x, 1.98x] | N/A | PENDING_GPU | PyTorch jagged nested_tensor + SDPA (is_causal=False) |
| `e_packed_bf16_autocast` | 1,631.6 | 92.4% | 581.7 | 0.0% | 3.36x [2.49x, 4.19x] | N/A | PENDING_GPU | ok |
| `e1_obfd_packed_bf16` | 1,450.8 | 92.4% | 674.9 | 0.0% | 3.30x [2.50x, 3.72x] | N/A | PENDING_GPU | Ding et al. (2024) O(N log N) Segment-Tree OBFD + BF16 |
| `f_packed_torch_compile` | — | — | — | — | — | N/A | PENDING_GPU | `failed (InductorError)`: InvalidCxxCompiler: Compiler: cl is not found. |
| `g1_sync_slow_tokenizer` | 790.0 | 78.0% | 1096.6 | 21.8% | 2.03x [1.64x, 2.40x] | N/A | PENDING_GPU | num_workers=0 with CPU tokenizer work |
| `g2_async_dataloader` | 1,217.2 | 78.0% | 718.9 | 5.2% | 2.72x [2.00x, 3.46x] | N/A | PENDING_GPU | num_workers=2 (spawn), prefetch_factor=2, pin_memory=(cuda) |
| `h_block_sparse_bf16` | 2,019.3 | 92.4% | 466.6 | 0.0% | 4.24x [3.32x, 6.27x] | N/A | PENDING_GPU | Tiled SDPA W=64+G=4 (FlexAttention backward unsupported on CPU) |
| `i_cuda_graphs` | — | — | — | — | — | N/A | PENDING_GPU | GPU only (`PENDING_GPU`) |

### Recipe 5: Batched 96-Well Plate Rehearsal

| Batch Plates | Steps / Plate | Volume Conserved | CPU (plates/s) | CPU Time (ms) | GPU (plates/s) |
|---:|---:|---|---:|---:|---|
| 128 | 104 | `True` | 307.1 | 416.75 | not yet measured (`PENDING_GPU`) |

### Open Protocol Corpora: Real Length Skew, Truncation-Free OBFD Packing & Checker Throughput

Measured via `python scripts/fetch_open_corpora.py` (`benchmarks/results/open_corpora_v2.json`; raw downloads cached in gitignored `data/open_corpora/`):

| Corpus | Docs | Median / P95 / Max Toks | Skew (`max/med`) | `bin_cap` (0 trunc) | Padded (`B=16`) | Bucketed (`B=16`) | First-Fit Bins (`useful%`) | OBFD Bins (`useful%`) | Batched Checker (proto/s) |
|---|---:|---|---:|---:|---:|---:|---|---|---:|
| `opentrons_shared_data` | 19 | 1058 / 4216 / 4216 | 3.98x | 8192 | 39.0% | 37.6% | 4 (84.0%) | 4 (84.0%) | 5,233.7 |
| `wlp` | 80 | 316 / 1032 / 1301 | 4.11x | 2048 | 37.4% | 71.5% | 15 (95.3%) | 15 (95.3%) | 7,113.0 |
| `bioplanner` | 80 | 450 / 1002 / 1257 | 2.80x | 2048 | 53.6% | 77.5% | 19 (97.2%) | 19 (97.2%) | 8,236.6 |
| `smolinstruct` | 200 | 90 / 185 / 361 | 3.99x | 512 | 47.3% | 89.4% | 40 (96.4%) | 39 (98.8%) | 80,516.4 |
| `ord` | 160 | 746 / 944 / 1085 | 1.45x | 2048 | 78.0% | 96.1% | 68 (86.8%) | 65 (90.8%) | 19,173.4 |

### Findings Supported Strictly by Non-Overlapping 5-Seed Intervals

- **Eliminating padding strictly improves real-token throughput over dynamic padding.** Across all 5 seeds, `b_length_bucketed` (`[1.76x, 2.85x]`, `58.3%` useful tokens), `c_sequence_packed` (`[2.37x, 3.51x]`, `92.4%`), `e_packed_bf16_autocast` (`[2.49x, 4.19x]`), `e1_obfd_packed_bf16` (`[2.50x, 3.72x]`), and `h_block_sparse_bf16` (`[3.32x, 6.27x]`) have minimum speedups strictly above `a_padded_baseline` (`1.00x`, `25.0%` useful tokens).
- **Fixed-row sequence packing (`c, e, e1, h`) strictly beats CPU jagged `nested_tensor` SDPA (`d`).** Despite `100.0%` useful tokens, CPU `nested_tensor` SDPA stays in `[1.54x, 1.98x]`, strictly below the `2.37x` 5-seed minimum of every packed configuration.
- **Packed vs. BF16 vs. OBFD vs. block-sparse overlap on `96`-sequence CPU microbatches.** While `h_block_sparse_bf16` (`[3.32x, 6.27x]`) strictly exceeds `b_length_bucketed` (`[1.76x, 2.85x]`), the 5-seed `[min, max]` ranges of `c_sequence_packed`, `e_packed_bf16_autocast`, `e1_obfd_packed_bf16`, and `h_block_sparse_bf16` overlap on this CPU, so no throughput rank claim is made among them. On larger real corpora (`SMolInstruct`, `ORD`), `pack_sequences_obfd` deterministically saves packed bins (`40 -> 39` and `68 -> 65`) with zero truncations.
- **Async `DataLoader` cuts tokenizer data-wait share (`21.8% -> 5.2%`)**, though overall CPU step-speedup intervals (`[1.64x, 2.40x]` vs `[2.00x, 3.46x]`) overlap when worker processes share CPU cores with compute.
- **Repair validity vs. exact protocol recovery (`Recipe 2`).** Single-edit repair (`K=26`) achieves `100.0%` checker pass rate across all 7 twin types and `100.0%` exact recovery on `wrong_pipette`, `missing_tip`, `sealed_plate` (via `delete_prior_seal`), and `swapped_steps`, `32.8%` on `skip_thaw` (where the twin generator also overwrites `first_xfer.src` on ~2/3 of rows), `0.8%` on `bad_temp` (original random incubation temperature is unknown), and `0.0%` on `overflow` (clamped to capacity rather than guessing pre-corruption volume).

## What this is not

- Not a replacement for a proven checker. The LabLean checker in
  [YLM](https://github.com/yaniguan/YLM) (Lean 4) stays the formal source of truth.
  On the 310-protocol live CI conformance suite (`tests/test_lablean_live_parity.py`), `labfill` and `YLM/LabLean` compile 310/310 exports with 0 errors and agree on 308/310 (`99.4%`) accept/reject verdicts (`105/105` valid protocols accepted, `105/105` error twins exact error-code match, `2/310` accounted for by the 4 documented rule-set deltas in [`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md)).
- Not a physics or robot simulator.
- Not a MILP or CP solver. For large scheduling problems use a dedicated solver such as [`labshift`](https://github.com/yaniguan/labshift).

## Design and roadmap

See [docs/DESIGN.md](docs/DESIGN.md). Next steps: execute `benchmarks/run_gpu_flash_flex.py` on a physical single-A100/H100 node to populate the `PENDING_GPU` columns (Issue #35); generate/check/keep with a live LLM sampler; multi-instrument CP-SAT integration.

## Related tools

See [docs/related_work.md](docs/related_work.md). Tool names are used only to point to related work; no affiliation.

## License

MIT
