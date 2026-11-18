# Related tools and papers

GPU tools, compiler primitives, and autonomous-lab papers for protocol validation, error detection/repair, planning, lab scheduling, liquid-handling simulation, verifier-gated LLM generation, and single-GPU training on short variable-length sequences. Every URL below was opened and checked on 2026-11-18 (HTTP 200, page title matches).

## 1. cuOpt (Open-Source Routing, LP, MILP)
- **What:** Apache-2.0 GPU solver for linear programming (LP), vehicle routing, and beta mixed-integer linear programming (MILP).
- **Why:** Accelerates multi-instrument lab scheduling, plate routing, and reagent allocation on GPU.
- **URL:** https://github.com/NVIDIA/cuopt

## 2. cuGraph / RAPIDS
- **What:** RAPIDS open-source GPU library for parallel graph traversal, shortest-path, and dependency analytics.
- **Why:** Protocol workflows are directed acyclic graphs (DAGs) needing fast cycle checks and critical-path analysis.
- **URL:** https://github.com/rapidsai/cugraph

## 3. Warp
- **What:** Apache-2.0 Python framework JIT-compiling Python functions into differentiable CPU/GPU simulation kernels.
- **Why:** Simulates well volumes, thermal steps, or deck collisions across thousands of protocols simultaneously.
- **URL:** https://github.com/NVIDIA/warp

## 4. Isaac Lab
- **What:** Open-source GPU robot-learning framework running parallel physics and sensor simulation across thousands of environments.
- **Why:** Trains and validates robotic arms, grippers, and deck-handling policies before lab deployment.
- **URL:** https://github.com/isaac-sim/IsaacLab

## 5. JAX `vmap`/`jit` and PyTorch `torch.func.vmap`
### JAX `vmap` and `jit`
- **What:** JAX transforms auto-vectorizing single-example functions (`vmap`) and compiling fused XLA GPU kernels (`jit`).
- **Why:** Turns a single-protocol checker or plate simulator into a fused multi-thousand-protocol GPU batch evaluator.
- **URL:** https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html https://docs.jax.dev/en/latest/_autosummary/jax.jit.html

### PyTorch `torch.func.vmap`
- **What:** PyTorch vectorizing map transform adding a batch axis to per-sample tensor operations without Python loops.
- **Why:** Vectorizes container updates, pipette range checks, and single-edit repair sweeps in PyTorch.
- **URL:** https://docs.pytorch.org/docs/2.15/generated/torch.func.vmap.html

## 6. vLLM and SGLang
### vLLM and PagedAttention (arXiv:2309.06180)
- **What:** High-throughput LLM engine using PagedAttention to store dynamic KV caches in non-contiguous blocks.
- **Why:** Samples hundreds of candidate protocol drafts per prompt in one GPU batch for verifier filtering.
- **URL:** https://github.com/vllm-project/vllm https://arxiv.org/abs/2309.06180

### SGLang and RadixAttention (arXiv:2312.07104)
- **What:** Structured LLM runtime reusing shared KV prefixes via RadixAttention with fast finite-state constrained decoding.
- **Why:** Multi-branch protocol planning and repair share long reagent prefixes and require strict structured outputs.
- **URL:** https://github.com/sgl-project/sglang https://arxiv.org/abs/2312.07104

## 7. GPU SAT, MaxSAT, and LP / Constraint Solving
### ParaFROST
- **What:** Parallel SAT solver running clause simplification and variable elimination on CUDA GPUs alongside CDCL search.
- **Why:** Simplifies Boolean encodings of hard rules like tip contamination, seal states, and step precedence.
- **URL:** https://github.com/muhos/ParaFROST

### TurboSAT (arXiv:2511.07737)
- **What:** Hybrid GPU-CPU SAT solver using parallel differentiable matrix-multiplication search on GPU to guide CPU CDCL.
- **Why:** Explores candidate step orderings and container assignments on GPU before exact discrete verification.
- **URL:** https://arxiv.org/abs/2511.07737

### GPU MaxSAT: `torchmSAT` (arXiv:2402.03640)
- **What:** Training-free differentiable MaxSAT approximation solver maximizing clause satisfaction via GPU backpropagation.
- **Why:** Guides minimum-edit step repair and soft-constraint scheduling when draft protocols break multiple rules.
- **URL:** https://arxiv.org/abs/2402.03640

### PDLP / cuPDLP (`cuPDLP.jl`, `cuPDLP-C`)
- **What:** GPU solvers implementing restarted primal-dual hybrid gradient (PDHG) first-order methods for large linear programs.
- **Why:** Solves LP relaxations of reagent allocation, instrument load balancing, and timing windows via sparse matvecs.
- **URL:** https://arxiv.org/abs/2311.12180 https://github.com/COPT-Public/cuPDLP-C

## 8. GPU MCTS (`mctx`)
- **What:** DeepMind JAX-native Monte Carlo tree search library supporting batched AlphaZero, MuZero, and Gumbel MuZero.
- **Why:** Expands and scores thousands of lookahead protocol planning branches or multi-step repairs in parallel.
- **URL:** https://github.com/google-deepmind/mctx

## 9. Batched Simulators (`Brax`, `MJX`, `EnvPool`, `Madrona`)
### Brax
- **What:** JAX rigid-body physics and RL library running thousands of parallel environments on one accelerator.
- **Why:** Demonstrates zero-copy GPU rollout loops keeping simulation and policy updates inside device memory.
- **URL:** https://github.com/google/brax

### MJX (MuJoCo XLA)
- **What:** MuJoCo JAX/XLA backend executing contact-rich robotic physics across thousands of batched GPU environments.
- **Why:** Validates grasp stability and collision-free trajectories for lab arms handling tubes, plates, and lids.
- **URL:** https://mujoco.readthedocs.io/en/stable/mjx.html

### EnvPool
- **What:** C++ thread-pool execution engine for vectorized RL environments with low-overhead batch stepping.
- **Why:** Prevents legacy CPU protocol simulators from bottlenecking GPU training when tensor rewrites are impractical.
- **URL:** https://github.com/sail-sg/envpool

### Madrona
- **What:** Entity-component-system (ECS) batch simulator that runs environment logic on the GPU.
- **Why:** Maps discrete lab items (wells, tips, plates, incubators) into GPU tables for parallel plate rehearsal.
- **URL:** https://github.com/shacklettbp/madrona

## 10. Lab Scheduling and Self-Driving Lab Papers
### Automated Lab Scheduling (arXiv:2607.01188)
- **What:** NIST framework (*Optimal Resource Utilization for Autonomous Laboratory Orchestrators*) pairing constraint-programming schedules with status dependencies.
- **Why:** Formalizes multi-instrument capacity and timing constraints targeted by GPU heuristics and MILP solvers.
- **URL:** https://arxiv.org/abs/2607.01188

### Coscientist (Nature 2023; arXiv:2304.05332)
- **What:** LLM chemical agent planning syntheses, reading docs, writing liquid-handler code, and running experiments.
- **Why:** Shows LLM liquid-handling scripts need fast pre-execution checks for volumes, tips, and temperatures.
- **URL:** https://www.nature.com/articles/s41586-023-06792-0 https://arxiv.org/abs/2304.05332

### BioPlanner (arXiv:2310.10632)
- **What:** Benchmark converting biology protocols into pseudocode functions to evaluate LLM planning and generation.
- **Why:** Typed pseudocode actions map cleanly onto integer opcodes and tensor states for GPU validation.
- **URL:** https://arxiv.org/abs/2310.10632

### BioProBench (arXiv:2505.07889)
- **What:** Protocol corpus and benchmark evaluating QA, step ordering, error correction, generation, and reasoning.
- **Why:** Its tasks (step ordering, error correction, generation) are natural targets for verifier-gated training.
- **URL:** https://arxiv.org/abs/2505.07889 https://github.com/YuyangSunshine/bioprobench

### ChemOS
- **What:** Modular self-driving lab software linking active-learning planners, databases, and automated chemical hardware.
- **Why:** Provides the closed-loop queue where fast GPU verifiers and schedulers can guard execution.
- **URL:** https://github.com/the-matter-lab/ChemOS https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229862

### HELAO
- **What:** Asynchronous FastAPI framework coordinating distributed multi-instrument workflows, active learning, and data provenance.
- **Why:** Concurrent multi-instrument runs create live rescheduling and timing-window validation workloads across shared stations.
- **URL:** https://github.com/High-Throughput-Experimentation/helao-async

## 11. Single-GPU Utilization for Short Variable-Length Sequences
### FlashAttention-2 `varlen` (arXiv:2307.08691)
- **What:** IO-aware fused exact attention offering unpadded `flash_attn_varlen_*` kernels indexed by cumulative sequence lengths.
- **Why:** Concatenates short, skewed-length protocol sequences flat without wasting GPU memory or FLOPs on padding.
- **URL:** https://arxiv.org/abs/2307.08691 https://github.com/Dao-AILab/flash-attention

### Sequence Packing without Cross-Contamination (arXiv:2107.02027)
- **What:** Packs multiple variable-length sequences per row while resetting position IDs and masking cross-sequence attention.
- **Why:** Eliminates heavy padding waste when training small transformers on short, variable-length protocol sequences.
- **URL:** https://arxiv.org/abs/2107.02027

### `torch.compile`
- **What:** PyTorch Dynamo/Inductor compiler tracing Python models and fusing elementwise ops and reductions into fast kernels.
- **Why:** Cuts Python dispatch and kernel-launch overhead that bottleneck small protocol models and tensor checkers.
- **URL:** https://docs.pytorch.org/docs/2.15/generated/torch.compile.html

### CUDA Graphs
- **What:** CUDA capture-and-replay API (`torch.cuda.CUDAGraph`) launching a static-shape GPU kernel sequence in one CPU call.
- **Why:** Removes CPU launch stalls when training or checking fixed-shape buffers of packed protocol tokens.
- **URL:** https://docs.pytorch.org/docs/2.15/generated/torch.cuda.CUDAGraph.html

### PyTorch FlexAttention (arXiv:2412.05496)
- **What:** Compiler-backed PyTorch API (`torch.nn.attention.flex_attention`) lowering Python `score_mod` and `block_mask` rules into fused kernels.
- **Why:** Runs packed-protocol document masks and step-dependency attention biases without writing custom kernels.
- **URL:** https://docs.pytorch.org/docs/2.15/nn.attention.flex_attention.html https://arxiv.org/abs/2412.05496

### Block-Sparse Attention: Sparse Transformers (arXiv:1904.10509)
- **What:** Factorized block-sparse attention patterns and kernels skipping masked tiles instead of computing dense matrices.
- **Why:** Skips empty off-diagonal tiles in packed protocol batches and local-step plus global-container masks.
- **URL:** https://arxiv.org/abs/1904.10509 https://github.com/openai/sparse_attention

### `torch.profiler`
- **What:** PyTorch built-in profiler recording CPU/GPU kernel timelines, operator FLOPs, memory usage, and DataLoader idle gaps.
- **Why:** Separates compute time from tokenizer stalls and padding waste when diagnosing low single-GPU utilization.
- **URL:** https://docs.pytorch.org/docs/2.15/profiler.html

### MFU Definition: PaLM Appendix B (arXiv:2204.02311)
- **What:** Model FLOPs Utilization, defined in PaLM Appendix B as observed throughput divided by theoretical peak model-FLOP throughput.
- **Why:** Counting FLOPs only on real (non-pad) protocol tokens measures true single-GPU gains from packing.
- **URL:** https://arxiv.org/abs/2204.02311

## Names
Tool names are used only to point to related open tools. This project has no affiliation with any of them.

## Gaps
These points reflect **our reading of the surveyed sources, not established fact**:
1. **No tensor-batched wet-lab protocol checker:** GPU simulators (`Warp`, `Isaac Lab`, `MJX`, `Madrona`) model 3D rigid bodies, while protocol benchmarks (`BioPlanner`, `BioProBench`) check protocols sequentially. None offer a batched tensor state machine checking wet-lab rules (pipette ranges, tip contamination, overflow, thaw-first, sealed plates, centrifuge balance) across thousands of protocols simultaneously.
2. **No turnkey GPU verifier-gated generation or repair loop:** `vLLM`, `SGLang`, and `mctx` generate or search branches in parallel, and `BioProBench` documents frequent ordering and numeric errors, yet none pair batched decoding with a vectorized GPU protocol pre-filter or single-edit repair sweep.
3. **Disconnected lab orchestrators and GPU solvers:** `ChemOS`, `HELAO`, and CP-SAT schedulers use CPU queues/solvers, while `cuOpt`, `cuPDLP`, `ParaFROST`, and `TurboSAT` lack encodings for wet-lab incubation windows and deck constraints.
4. **No single-GPU saturation recipe for short protocol sequences:** `FlashAttention-2` `varlen`, packing, `FlexAttention`, `torch.compile`, and `CUDA Graphs` target general LLMs, leaving a gap for small reproducible benchmarks measuring real-token MFU on short, skewed-length protocol datasets.
