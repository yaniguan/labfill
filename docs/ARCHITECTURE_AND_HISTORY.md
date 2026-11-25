# `labfill` — Architecture, Benchmarks, and Engineering History

> **Auto-Lab Ecosystem Role:** Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Individual wet-lab protocol checks, 96-well plate rehearsals, single-edit repairs, and step-DAG searches operate on small discrete structures—typically 10–110 steps over 6–99 containers. Executed one protocol at a time in Python, they cannot saturate SIMD vector units or GPUs; meanwhile, training neural models on short, skewed-length protocol sequences (`8..512` tokens) wastes most accelerator FLOPs on padding tokens and synchronous CPU preprocessing stalls.

`labfill` (*"Fill the plate, fill the GPU"*) reformulates five core laboratory automation and protocol-learning workloads as **batched PyTorch tensor recipes** that run identically on CPU and CUDA (`cuda` when available, else `cpu`):

1. **Recipe 1 — Batched Protocol Validation (`labfill.checker`)**: Unit-stride `B`-wide state-machine verification across `B` protocols and `T` steps for pipette volume bounds, tip lifecycle, cross-reagent tip contamination, freeze/thaw rules, container sealing, incubation temperature/duration, source depletion, destination overflow, and centrifuge balance/speed.
2. **Recipe 2 — Seeded-Error Twins, Verifier-Gated Filtering & Batched Repair (`labfill.twins`)**: Deterministic single-error twin injection across 7 physical failure modes, `Generate-N / Check / Keep` candidate filtering, and single-pass `M x K` (`K = 24`) single-edit protocol repair.
3. **Recipe 3 — Checker-Gated DAG Beam Planning & Parallel SA Scheduling (`labfill.planner`)**: Batched beam search over partial step-precedence DAGs gated at every depth by `check_protocols_batched`, plus `R`-restart parallel simulated annealing (`schedule_batched_sa`) for heterogeneous multi-instrument makespan minimization.
4. **Recipe 4 — Single-GPU Training Saturation (`labfill.fill_gpu`)**: Controlled 10-configuration throughput and Model FLOPs Utilization (MFU) ablation on a 4-layer `d_model = 256` `ProtocolTransformer` over lognormal (`8..512` token) protocol traces, isolating dynamic padding, length bucketing, sequence packing with reset `position_ids` and block-diagonal masks, jagged `nested_tensor` SDPA, `bfloat16` autocast, `torch.compile`, asynchronous `DataLoader` workers, tiled local+global block-sparse attention, and static-shape CUDA Graphs.
5. **Recipe 5 — Batched 96-Well Plate Rehearsal (`labfill.plate`)**: Parallel dry-run execution of `B` 96-well serial-dilution protocols (`A1..H12` row-major indexing matching LabLean) tracking exact `0.1 uL` integer volumes and `float32` advected solute concentrations.

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture:
- **Layer (`L3`)**: Supplies high-throughput tensorized reward/verifier pre-filtering, negative-twin generation, single-edit repair, and sequence-packed training recipes for post-training RLVR (`labbench-posttrain`).
- **Relationship to `L2` Formal Verification (`YLM` LabLean & `lab-verify`)**: `labfill` is a fast statistical/tensor pre-filter that mirrors `LabLean`'s core unit and pipette conventions (`0.1 uL` integer volumes, `P2`/`P20`/`P200`/`P1000` ranges, `A1..H12` well indexing), whereas the Lean 4 checker in [`YLM`](https://github.com/yaniguan/YLM) (`lean/LabLean`) remains the formally proven default-deny gate prior to physical robot execution.

### I/O Contract & Scope Boundaries

| Module | Primary Inputs | Primary Outputs |
|---|---|---|
| `labfill.protocol` | Step dicts or generator seeds (`B, T, C`) | `LabStateBatch` `(B, C)` + `ProtocolBatch` `(B, T)` (`int64`/`bool` tensors) |
| `labfill.checker` | `LabStateBatch`, `ProtocolBatch` | `CheckResult` (`ok`, `err_code`, `first_err_step`, `final_state`) |
| `labfill.lablean_io` | `LabStateBatch`, `ProtocolBatch` (or JSON AST) | Lean 4 `LabState` + `List Step` modules, JSON AST, `ERR_* <-> LabLean.Error` tags |
| `labfill.twins` | Valid/broken `(LabStateBatch, ProtocolBatch)`, `error_type`, `K` | Corrupted twins, `TwinRecallReport`, `GenerateKeepResult`, `RepairResult`, DPO/`twinstep` JSONL |
| `labfill.planner` | `StepDAGProblem` (`S <= 62`) or durations `(J,)` / `(J, M)` | `DAGPlanResult` (valid topological order) / `ScheduleResult` (assignment, makespan) |
| `labfill.fill_gpu` | `ProtocolSequenceItem` list (`tokens`, `labels`, `raw_text`) | `PreparedBatch`, `AblationRow` metrics, `torch.profiler` ops, `FlexAttention` probe |
| `labfill.plate` | 96-well + reservoir `LabStateBatch` (`C >= 96`), `ProtocolBatch` | `PlateRehearsalResult` (`well_volumes_tenths_ul`, `well_volumes_ul`, `solute_conc`) |

**Explicit Scope Limits:**
- Not a replacement for a theorem-proved verifier (`YLM` LabLean).
- Not a 3D rigid-body, fluid-dynamics, or deck-collision simulator.
- Not an exact combinatorial solver; constrained multi-operation scheduling with release/deadline/wait windows belongs in dedicated solvers such as [`labshift`](https://github.com/yaniguan/labshift) (OR-Tools CP-SAT) or GPU MILP/routing engines.

---

## 2. Data

`labfill` ships **synthetic, deterministic, commercially clean protocol generators** (MIT licensed; zero external dataset downloads required). Physical units match `LabLean`:
- **Volume**: `int64` tenths of a microliter (`0.1 uL`; `ul(2.5) == 25`).
- **Temperature**: `int64` whole degrees Celsius (`MIN_INCUBATE_C = 4`, `MAX_INCUBATE_C = 98`).
- **Duration**: `int64` whole seconds (`> 0`).
- **Centrifuge Speed & Balance**: `int64` whole RPM (`1..15000`), balance tolerance `ROTOR_TOL_TENTHS_UL = 50` (`5.0 uL`).

### Bundled Synthetic Protocol & Sequence Suites

| Generator | Location | Dimensions & Structure | Seed & Determinism |
|---|---|---|---|
| `make_pcr_protocol` | `labfill/protocol.py` | `B=1`, `C=13` containers (`WATER`, frozen `BUF`/`DNTP`/`FWD`/`REV`/`TMPL`, thawed `TAQ`, `MM`, wells `A1..A4`, `BAL_WELL`), `T=46` steps mirroring `LabLean/Examples/PCR.lean` (5x master mix for 4x `50.0 uL` reactions, single-tip multi-dispense into empty wells, per-well primers/template + mix, seal, balanced spin, `98 C` incubation). | Deterministic (no randomness). |
| `make_random_valid_protocols` | `labfill/protocol.py` | Default `T=24`, `C=8` (`3` stock tubes at `800.0 uL` / `cap 1500.0 uL` with stocks `1, 2` frozen; reaction wells `3..C-2` and balance well `C-1` at `cap 300.0 uL`); thaws `1, 2`, runs `pick_tip -> transfer -> drop_tip` blocks, incubates well `3` (`25..95 C`), seals/unseals, and appends balanced `seal + seal + centrifuge` on ~70% of balanced lanes. | CPU `torch.Generator` seeded with `seed`. |
| `make_random_protocols` | `labfill/protocol.py` | Corrupts a `error_prob` fraction of `make_random_valid_protocols` at up to 2 random step slots per selected row using out-of-range opcodes (`0..10`), container IDs (`-1..C`), volumes (`-10..11999`), pipettes (`0..5`), temps (`-20..129`), durations (`-5..119`), and RPMs (`-100..19999`). | `seed` and `seed + 99991`. |
| `make_error_twin` | `labfill/twins.py` | Injects 1 of 7 typed errors into valid batches: `overflow`, `wrong_pipette`, `missing_tip`, `skip_thaw`, `sealed_plate` (`T+1` insertion), `bad_temp` (`120 C` or `-10 C`), `swapped_steps`. | `seed + 1337`. |
| `make_assay_dag_problem` | `labfill/planner.py` | `S=16` unordered steps, `C=6` containers, `14` directed precedence edges: 2 stock thaws, 3 unordered reagent-to-master-mix blocks (`A, B, C`), 2-well dispense block `D`, and `65 C` incubation. | Deterministic problem; `seed` for random topo baseline. |
| `make_lab_schedule_durations` | `labfill/planner.py` | Default `J=96` jobs across `M=8` heterogeneous instruments: base duration `30..419 s` scaled by linear speed factors `0.85..1.25` plus `0..34 s` per-(job, instrument) setup jitter. | CPU `torch.Generator` seeded with `seed`. |
| `generate_protocol_sequences` | `labfill/fill_gpu.py` | Lognormal token lengths (`mean=3.65`, `sigma=0.95`, clipped to `[8, 512]`, `VOCAB_SIZE=256`, `PAD=0, BOS=1, EOS=2, SEP=3`); 5-token step tuples with 8 error classes (`0` clean, `1..7` corrupted 4-token span in 80% of items where `i % 5 != 0`). | `numpy.random.default_rng(seed)`. |
| `make_serial_dilution_plate_batch` | `labfill/plate.py` | Default `B=128` plates, `C=99` containers (`0..95` wells `A1..H12` cap `300 uL`; `96` diluent `5000 uL`; `97` stock `2000 uL`; `98` waste cap `10000 uL`), `4` active rows x `8` cols (`T=104` steps), random 2-fold dilution volume `dv in {50, 75, 100} uL`. | CPU `torch.Generator` seeded with `seed`. |

---

## 3. Model

### 3.1 Tensorized Protocol State Machine (`labfill/checker.py`)
`LabStateBatch` stores per-protocol state across `B` lanes and `C` containers (`volume`, `capacity`, `reagent_mask` as `(B, C) int64`; `sealed`, `frozen` as `(B, C) bool`; `has_tip` as `(B,) bool`; `tip_mask` as `(B,) int64` bitmask supporting up to 62 distinct reagents). `ProtocolBatch` encodes `T` padded steps (`op`, `src`, `dst`, `vol`, `temp`, `duration`, `pipette`, `rpm` as `(B, T) int64`; `pad_mask` as `(B, T) bool`).

`check_protocols_batched` transposes all `(B, T)` step tensors once into contiguous `(T, B)` arrays so each step `t` executes coalesced unit-stride vector operations over `B` protocols without Python branching or host-device synchronization:
1. **Active Lane Mask**: `active = (err_code == ERR_OK) & (~pad_tb[t]) & (op_t != OP_NOOP)`. Once a lane records an error, `err_code != ERR_OK` freezes its state at the pre-failure snapshot.
2. **Clamped Gather & Rule Priority**: Container indices are safely clamped to `[0, C - 1]` (`src_c2`, `dst_c2`) after bounds checking (`ERR_BAD_CONTAINER`), enabling branch-free `gather` and `scatter_add_`. Each active lane evaluates 16 error conditions in strict priority order (`ERR_BAD_OP=1` -> `ERR_BAD_CONTAINER=2` -> `ERR_TIP_ALREADY_ON=4` / `ERR_NO_TIP=3` -> `ERR_BAD_PIPETTE=5` -> `ERR_SAME_CONTAINER=6` -> `ERR_SEALED=7` -> `ERR_FROZEN=9` -> `ERR_CONTAMINATED=10` -> `ERR_NOT_ENOUGH=11` -> `ERR_OVERFLOW=12` -> `ERR_BAD_TEMP=13` / `ERR_BAD_DURATION=16` -> `ERR_NOT_SEALED=8` -> `ERR_BAD_RPM=14` -> `ERR_UNBALANCED=15`).
3. **State Transitions**: Passing lanes (`ok_now`) update `has_tip`, `tip_mask` (`tip_mask | src_rmask | dst_rmask` on transfer; `tip_mask | src_rmask` on mix), paired `volume` `scatter_add_` (`-vol_t` at `src`, `+vol_t` at `dst`), `reagent_mask` union (`dst_rmask | src_rmask`), `frozen`, and `sealed`.
4. **Reference Equivalence**: `check_protocols_python` implements the exact scalar state machine over plain Python lists for differential testing (`tests/test_checker.py` verifies exact match over 4,500 random/corrupted protocols).

### 3.2 Batched Single-Edit Repair (`labfill/twins.py`)
`batched_repair` runs `check_protocols_batched` once to locate `first_err_step` for `M` broken protocols, expands them into `M * K` (`K = 24`) length-`(T + 1)` candidates via `_propose_k_edits` (pipette swaps `0..3`, volume+pipette clamps `4..7`, incubation fixes `8..9`, step deletion `10`, adjacent swaps `11..12`, step insertions `13..20`, prior-seal deletion `21`, destination redirects `22..23`), evaluates all `M * K` candidates in a single batched checker pass, and selects the lowest-indexed passing edit per protocol. *(On open PR `origin/review-fixes-nov18`, edits `4..7` compute exact capacity/source-fitting clamps from `state_at_err` and intent-changing edits `10` on `INTENT_OPS` and `22+` are excluded from `repair_rate`.)*

### 3.3 DAG Beam Search & Batched Simulated Annealing (`labfill/planner.py`)
- **`batched_beam_plan_dag`**: Encodes DAG prerequisites as 62-bit integer masks (`pred_bits[v] |= 1 << u`). At each depth `0..S-1`, all enabled `(beam, next_step)` pairs are gathered into partial `ProtocolBatch` prefixes, screened in one `check_protocols_batched` call, scored with an additive transition cost (`+1.0` pipette switch, `+0.25` source container jump, `-0.1` early thaw bonus), and pruned via `torch.topk` to `beam_width`.
- **`schedule_batched_sa`**: Evolves `R` assignments `(R, J)` in parallel over `num_steps`, alternating O(1) 6-way tournament bottleneck-job moves (60% min-load instrument, 40% uniform random) with random 2-job instrument swaps under Metropolis acceptance `exp(-delta / T)`.

### 3.4 `ProtocolTransformer` & Attention Modes (`labfill/fill_gpu.py`)
`ProtocolTransformer` is a 4-layer pre-norm Transformer (`d_model = 256`, `n_heads = 4`, `d_ff = 1024`, `vocab_size = 256`, `max_seq_len = 512`, `num_classes = 8`) supporting three attention backends without cross-sequence contamination:
- **Dense / Sequence-Packed (`mode="dense"`)**: `pack_sequence_pairs` greedily bin-packs variable-length sequences into fixed `pack_len = 512` rows, resets `position_ids` to `0..n_i - 1` at every sequence boundary, and assigns `seq_id in {1..K}` (`0` for trailing pad). `build_block_diag_mask` enforces `(seq_id_q == seq_id_k) & (seq_id_q > 0)` while routing padded query slots (`seq_id == 0`) to the identity diagonal so softmax never encounters all-masked rows (`NaN`-free).
- **Jagged Varlen (`mode="varlen"`)**: `collate_varlen_pairs` flattens tokens into 1D `(T_real,)` tensors with cumulative `cu_seqlens`, building `torch.nested.nested_tensor_from_jagged` Q/K/V views per block for zero-padding SDPA.
- **Tiled Block-Sparse (`mode="block_sparse"`)**: `tiled_local_global_attention` partitions length-`S` rows into `S / W` blocks (`W = 64`), attending each query block to `[prev_block, cur_block]` (`2W = 128` local keys) plus `G = 4` row-prefix global keys (`kv_len = 132`), masked by `seq_id`. *(Open PR `origin/review-fixes-nov18` commit `990ae3d` masks duplicate global keys when `G` overlaps `[prev_blk, cur_blk]` in blocks 0 and 1.)*

### 3.5 Batched 96-Well Rehearsal (`labfill/plate.py`)
`rehearse_96well_protocols` validates `B` 96-well plate protocols via `check_protocols_batched` and runs a synchronized `float32` solute-mass advection pass across valid step prefixes (`lane_valid = (first_err_step < 0) | (t < first_err_step)`), reshaping final volumes and relative concentrations into `(B, 8, 12)` plate grids.

---

## 4. Training

### Recipe 4: Single-GPU Saturation & Analytic FLOP Accounting
`run_fill_gpu_ablation` (`labfill/fill_gpu.py`) trains `ProtocolTransformer` with `AdamW(lr=3e-4)` and token-level cross-entropy (`ignore_index = -100`) across 10 configurations (`a_padded_baseline` through `i_cuda_graphs`), measuring wall-clock `real_tokens_per_sec`, `useful_token_fraction`, median `step_time_ms`, `data_wait_fraction` (`SlowTokenizerDataset` performing deterministic `blake2b` CPU work with `num_workers=0` vs `num_workers=2` `spawn` + `prefetch_factor=2`), and PaLM Appendix B analytic FLOPs (`count_batch_training_flops`, `3x` forward FLOPs):
- **Useful FLOPs**: Counts linear projections (`L * (8 d^2 + 4 d f) + 2 d C_out`) only on `real_tokens` and attention (`4 L d * sum_i s_i^2`) only within true unpadded sequence lengths `s_i`.
- **Executed FLOPs**: Counts actual dense `(B, S, S)` or tiled `(B, S, 2W + G)` operations including pad slots and masked cross-document tiles.

### Role in Verifier-Gated Post-Training (RLVR / DPO / SFT)
Recipe 2 (`make_error_twin`, `generate_check_keep`, `batched_repair`) implements the high-throughput **verifier-gated data factory** pattern: sampling `N` noisy candidate protocols, rejecting physically invalid rollouts in one tensor pass, and pairing broken rollouts with minimal verified repairs for SFT, DPO preference pairs, or GRPO reward shaping in [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain).

### Measured Benchmark Status (CPU Measured vs. GPU Pending)
- **GPU Status**: **Not yet measured (`PENDING`)** across all five recipes (`main` and all open PR branches). `MFU` reports `"N/A"` until a hardware `peak_flops` value is supplied on a CUDA device, and `i_cuda_graphs` is gated behind `dev.type == "cuda"`.
- **CPU Measurements (Open PR `origin/cpu-benchmark-results`, measured 2026-11-18)**: Recorded on an 8-core Intel Xeon (`avx2, avx512f, avx512_bf16, amx_bf16, amx_tile`), Python `3.11.16`, PyTorch `2.15.0+cpu`, NumPy `2.4.6`. Because the shared host had elevated background load (`12.9–17.0` 1-min load avg for `--threads 1` in `cpu_2026-11-18_t1.json`; `38.7–45.3` for `--threads 8` in `cpu_2026-11-18.json`), single-thread (`--threads 1`) numbers are the primary reference:
  - **Recipe 1 (`T=24, C=8`)**: Batched Torch CPU reaches `225,851.2 proto/s` at `B=16,384` (`1.24x` plain Python's `182,366.9 proto/s`), while Python wins at small batches (`192,049.2` vs `153.3 proto/s` at `B=1`; `244,011.1` vs `104,513.6 proto/s` at `B=1,024`).
  - **Recipe 2**: `0.0%` false positives on valid protocols; `100.0%` recall on all 7 self-consistent twin classes; `generate_check_keep` (`N=4,096`) keeps `77.4%` at `163,078.0 candidates/s`; `batched_repair` (`K=24`) repairs `100.0%` of 6 twin types and `68.8%` of `overflow` at `4,339.0–4,840.7 proto/s` on `main`'s fixed clamp grid.
  - **Recipe 3**: Checker-gated DAG beam search (`S=16, beam=32`) finds a valid topological plan (`735` candidates checked in `43.07 ms`) where naive random topological orders pass only `1.6%`; `batched_sa_r256` reaches `2,688.0 s` makespan (`43.05 ms`) vs `2,696.0 s` (`greedy_lpt`, `0.09 ms`) and `2,859.0 s` (`greedy_arrival`, `0.09 ms`).
  - **Recipe 4 (`4-layer d=256`)**: Useful-token fraction climbs from `30.6%` (`a_padded_baseline`, `1,509.9 real tok/s`) to `58.9%` (`b_length_bucketed`, `3,150.4 tok/s`, `2.09x`) and `92.1%` (`c_sequence_packed`, `3,931.5 tok/s`, `2.60x`). Adding `bfloat16` autocast (`e_packed_bf16_autocast`) yields `8,180.5 real tok/s` (`5.42x` vs baseline on Xeon AMX-BF16). Async `DataLoader` (`g2`) cuts data-wait share from `25.7%` to `10.0%` (`4,957.9 -> 5,574.6 tok/s`). `f_packed_torch_compile` failed cleanly due to missing `g++` (`InductorError`), and `FlexAttention` backward is unsupported on CPU (`h_block_sparse_bf16` ran the tiled fallback at `7,428.2 tok/s`, `4.92x`).
  - **Recipe 5 (`B=128, T=104`)**: `3,823.2 plates/s` (`33.48 ms` per 128-plate batch).
- **Local Test Audit (2026-11-22)**: `16 passed, 1 warning` under `pytest` (Python `3.11.15`, CPU).

---

## 5. History

### Repository & Open PR Branch Evolution

| Commit | Branch / Ref | Date | Summary of Technical Changes |
|---|---|---|---|
| `ae91eb3` | `main` (`origin/main`) | 2026-11-18 | Initial `labfill` release: core tensor protocol encoding (`protocol.py`), batched & plain-Python checkers (`checker.py`), twins/filtering/repair (`twins.py`), DAG beam planner & parallel SA scheduler (`planner.py`), `ProtocolTransformer` saturation ablation (`fill_gpu.py`), 96-well rehearsal (`plate.py`), 5 runnable recipes, `benchmarks/run_all.py`, `docs/related_work.md`, 6 test modules, and GitHub Actions CPU CI. |
| `783283b` | `origin/cpu-benchmark-results` | 2026-11-18 | Saved measured 8-thread CPU JSON parts for Recipes 1, 2, 3, and 5 under `benchmarks/results/parts/`. |
| `5440a0c` | `origin/cpu-benchmark-results` | 2026-11-18 | Added `docs/DESIGN.md` summarizing tensor layout, ecosystem links (`YLM`, `labbench-posttrain`), limits, and roadmap. |
| `d2b0ac0` | `origin/cpu-benchmark-results` | 2026-11-18 | Added Recipe 4 8-thread CPU measurements and assembled full `cpu_2026-11-18.{json,md}` report. |
| `84369b0` | `origin/cpu-benchmark-results` | 2026-11-18 | Added `--threads` and `--tag` flags to `benchmarks/run_all.py` to isolate single-thread runs on shared hosts. |
| `fdc57a8` | `origin/cpu-benchmark-results` | 2026-11-18 | Saved 1-thread CPU benchmark results (`benchmarks/results/cpu_2026-11-18_t1.{json,md}` and `parts_t1/`). |
| `0d5a4d4` | `origin/cpu-benchmark-results` | 2026-11-18 | Published measured 1-thread CPU tables and ablation findings in `README.md`. |
| `d1a7510` | `origin/cpu-benchmark-results` (HEAD) | 2026-11-18 | Hardened `benchmarks/run_all.py` (`assemble_payload` refuses mixed `torch_num_threads` parts, prints per-recipe load averages, clarifies `--fresh` and AMX-BF16 notes) and added `tests/test_run_all.py`. |
| `990ae3d` | `origin/review-fixes-nov18` | 2026-11-18 | Fixed `tiled_local_global_attention` double-counting global keys in blocks 0 and 1; set `AdamW(..., capturable=True)` with error capture in `run_fill_gpu_ablation` config `i_cuda_graphs`; added dense-vs-sparse equivalence and CUDA Graph tests. |
| `14d065a` | `origin/review-fixes-nov18` | 2026-11-18 | Refactored `batched_repair` so intent-changing edits (deleting `INTENT_OPS` or redirecting `dst`) are excluded from `repair_rate` and tracked in `num_fixed_only_by_intent_change`; upgraded edits `4..7` to adaptive capacity/source-fitting volume clamps; stopped double-reporting `swapped_steps` recall as primary-code match. |
| `55eb158` | `origin/review-fixes-nov18` (HEAD) | 2026-11-18 | Updated `check_protocols_python` to return full `LabStateBatch` fields (not just `volume`); added analytical final-volume assertions in `tests/test_plate.py`; bumped `pyproject.toml` minimum to `torch>=2.5`; clarified non-uniform random topological sampling and tautological volume conservation comments. |

*(Note: Both `origin/cpu-benchmark-results` and `origin/review-fixes-nov18` are open PR branches ahead of `ae91eb3`; `main` currently reflects `ae91eb3` plus this architecture/history documentation.)*

---

## 6. Unique Properties

### Key Invariants & Design Guarantees
1. **LabLean Unit & Well-Index Fidelity**: Volumes use exact `int64` `0.1 uL` steps (`ul`), pipette ranges match `LabLean` (`P2..P1000`), and 96-well coordinates use row-major `0..95` / 1-based `label_1based(65) == "F5"` (`A1..H12`).
2. **Sync-Free, Branch-Free Batched Checking**: Pre-transposing `(B, T) -> (T, B)` and clamping gather/scatter indices allows `check_protocols_batched` to validate arbitrary mixtures of valid and broken protocols without `.item()` calls or dynamic control-flow splits across `B`.
3. **Zero Cross-Sequence Contamination in Packed Training**: `pack_sequence_pairs` resets `position_ids` per packed sequence and isolates attention via `seq_id` block-diagonal masks (`build_block_diag_mask`), verified by perturbation invariance tests (`tests/test_fill_gpu.py`).
4. **Honest Useful-vs-Executed FLOP Separation**: `count_batch_training_flops` credits only non-pad tokens and intra-document `sum(s_i^2)` attention toward useful throughput and MFU, preventing padding FLOP inflation.

### Known Modeling Divergences & Limits (vs. `YLM` LabLean)
- **Aggregated Non-Diluent Solute vs. Per-Reagent Concentration Vectors**: `labfill` combines `int64` reagent presence bitmasks (`(tip_mask & ~src_rmask) != 0`) with integer non-diluent solute tracking (`AMT_SCALE = 10000`, `CONC_TOL_PERMILLE = 1`, `diluent_mask`), catching serial-dilution back-transfers (`Broken.backUpRow`), whereas `LabLean` tracks independent `(reagent_name, amount)` pairs per reagent.
- **Two-Slot Volume Centrifuge vs. `N`-Slot Mass Rotor**: `check_protocols_batched` validates initial-state invariants (`0 <= volume <= capacity`) before step 0, and models centrifugation as a 2-container volume-difference check (`<= 5.0 uL`, both containers sealed) rather than `LabLean`'s `n`-slot rotor mass balance with container tare and reagent densities.
- **Self-Consistent Synthetic Twins**: Twin recall (`100%`) reflects rule self-consistency on synthetic mutations rather than empirical recall on noisy human/LLM wet-lab transcripts.

### Ecosystem Interfaces
- **Upstream / Peer (`L2`–`L4`)**: Shares step semantics and unit conventions with [`YLM`](https://github.com/yaniguan/YLM) (`lean/LabLean`), ingests error-twin sequence lengths from [`twinstep`](https://github.com/yaniguan/twinstep) alongside post-execution causal blame and certified ledger provenance from [`lab-attrib`](https://github.com/yaniguan/lab-attrib), and feeds verifier-filtered rollouts and single-GPU packing recipes into [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain).
- **Downstream (`L4`–`L6` & Body Gates)**: Fast pre-screening complements [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) deck dry-runs and [`lab-verify`](https://github.com/yaniguan/lab-verify) default-deny gates, while multi-job multi-instrument scheduling escalates to [`labshift`](https://github.com/yaniguan/labshift) and verified liquid-handling protocols compile to Opentrons Python API v2 via [`stepbot`](https://github.com/yaniguan/stepbot).

---

## 7. 15-Repository Auto-Lab Wire Contract (`autolab-wire/v1`)

`autolab_wire.py` (`REPO_ID = "labfill"`, `LAYER = "L3"`) exposes `process_packet(packet: dict) -> dict` for deterministic cross-repository packet execution across the 15-repository [Auto-Lab](https://github.com/yaniguan/auto-lab) stack:

- **Upstream Inputs (`twinstep` & `lab-attrib`)**: Reads `error_twins.token_lengths` (default `[96, 144, 180, 72]`), optional `prompt_lengths` (`0 <= prompt_len <= token_len`), and optional `pass_rates` / `rollout_rewards` (`p * (1 - p)` frontier weights) produced by `twinstep` while preserving post-execution attribution state (`attribution`, `certified_ledger_entries`) emitted by `lab-attrib`. If `packet["halted"]` is already `True`, `process_packet` returns a deep copy unchanged.
- **BF16 Sequence Packing & `LabFillLean` Verification**: Invokes `labfill.fill_gpu.pack_sequence_pairs`, O(`bin_cap`) `build_cu_seqlens` run-length boundary verification (avoiding O(`bin_cap^2`) boolean masks), and `community_extensions.curriculum_difficulty_packer.materialize_bf16_packed_batch` (falling back to `pack_lengths_pure_python` only if PyTorch is absent) to pack sequence lengths into non-overlapping rows of capacity `bin_cap` (default `512`, auto-aligning up to `2048` / `8192` when `bin_cap` is unspecified). Enforces `LabFillLean.checkPackedBin` (`0 < len`, `offset + len <= bin_cap`, unique `seqId`s and pairwise disjoint intervals, and `sum(packed_slices) == sum(token_lengths)`), halting with `halt_reason = "L3_LABFILL_PACK_BLOCK"` (`bin_count = 0`, `useful_token_ratio = 0.0`) on any overlap (`force_overlapping_slices`) or conservation violation.
- **Downstream Outputs (`labbench-posttrain`)**: Populates `out["packed_batch"]` (`bin_cap`, `bin_count`, `total_real_tokens`, `useful_token_ratio`, `zero_cross_seq_overlap`, `bins`, `cu_seqlens`, `bin_cu_seqlens`, `max_seqlen`, `loss_mask_spans`, `total_completion_tokens`), provides `packed_batch_to_posttrain_tensors(...)`, and appends structured audit entries to `out["trace"]` and `out["bottom_reasoning_traces"]` (`invariant = "checkPackedBin_conserves_tokens"`) for downstream `SplitGuard` and SFT/MPO/DPO/GRPO/GDPO post-training in `labbench-posttrain`.
