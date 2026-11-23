"""Recipe 3: Batched planning over step DAGs and batched parallel SA for instrument scheduling.

Note on scope: dedicated solvers (MILP, OR-Tools CP-SAT, GPU LP/routing solvers such as cuOpt)
are the right tool for large or heavily constrained scheduling instances. These PyTorch routines
provide zero-dependency tensor baselines when protocols already live on CPU/GPU tensors.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from labfill.checker import check_protocols_batched
from labfill.protocol import (
    OP_DROP_TIP,
    OP_INCUBATE,
    OP_PICK_TIP,
    OP_THAW,
    OP_TRANSFER,
    PIP_NONE,
    PIP_P20,
    PIP_P200,
    LabStateBatch,
    ProtocolBatch,
    default_device,
    ul,
)


@dataclass
class StepDAGProblem:
    """Unordered protocol steps with partial precedence edges (u, v) meaning u before v."""

    init_state: LabStateBatch  # (1, C)
    steps: ProtocolBatch       # (1, S)
    edges: list[tuple[int, int]]

    def __post_init__(self) -> None:
        validate_dag_problem(self)


def validate_dag_problem(problem: StepDAGProblem) -> None:
    """Validate single-instance batch shape, device alignment, edge bounds, and DAG acyclicity."""
    if problem.init_state.batch_size != 1 or problem.steps.batch_size != 1:
        raise ValueError(
            f"StepDAGProblem requires batch_size == 1, got init_state={problem.init_state.batch_size}, "
            f"steps={problem.steps.batch_size}"
        )
    if problem.init_state.device != problem.steps.device:
        raise ValueError(
            f"Device mismatch: init_state on {problem.init_state.device} != steps on {problem.steps.device}"
        )
    s = problem.steps.max_steps
    if s < 1:
        raise ValueError(f"StepDAGProblem must have at least 1 step, got S={s}")

    adj: list[list[int]] = [[] for _ in range(s)]
    indeg = [0] * s
    seen_edges: set[tuple[int, int]] = set()
    for u, v in problem.edges:
        if not (0 <= u < s and 0 <= v < s):
            raise ValueError(f"DAG edge ({u}, {v}) out of valid step range 0..{s - 1}")
        if u == v:
            raise ValueError(f"Self-loop edge ({u}, {v}) makes StepDAGProblem cyclic")
        if (u, v) not in seen_edges:
            seen_edges.add((u, v))
            adj[u].append(v)
            indeg[v] += 1

    q: deque[int] = deque(i for i in range(s) if indeg[i] == 0)
    visited = 0
    while q:
        node = q.popleft()
        visited += 1
        for nxt in adj[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    if visited != s:
        raise ValueError("StepDAGProblem edges contain a cycle")


@dataclass
class DAGPlanResult:
    found: bool
    best_order: list[int]
    best_cost: float
    best_protocol: ProtocolBatch | None
    num_valid_beams: int
    total_candidates_checked: int
    elapsed_sec: float


def make_assay_dag_problem(
    device: str | torch.device | None = None,
) -> StepDAGProblem:
    """Build a 16-step master-mix + dispense DAG with partial precedence constraints.

    Three reagent-addition blocks (A, B, C) each require pick_tip -> transfer -> drop_tip,
    and frozen stocks 1 & 2 must be thawed before their transfers, but the DAG does NOT
    force an order between blocks A, B, and C. Naive topological sorts interleave tip ops
    and fail physical validation, whereas checker-gated beam search prunes bad interleavings.
    """
    dev = default_device(device)
    # Containers: 0=water(thawed), 1=buf(frozen), 2=primer(frozen), 3=MM, 4=wellA1, 5=wellA2
    vols = [ul(500), ul(100), ul(100), 0, 0, 0]
    caps = [ul(1500), ul(1500), ul(1500), ul(1500), ul(200), ul(200)]
    frozen = [False, True, True, False, False, False]
    rmask = [1 << 0, 1 << 1, 1 << 2, 0, 0, 0]

    state = LabStateBatch(
        volume=torch.tensor([vols], dtype=torch.int64, device=dev),
        capacity=torch.tensor([caps], dtype=torch.int64, device=dev),
        sealed=torch.zeros((1, 6), dtype=torch.bool, device=dev),
        frozen=torch.tensor([frozen], dtype=torch.bool, device=dev),
        reagent_mask=torch.tensor([rmask], dtype=torch.int64, device=dev),
        diluent_mask=1 << 0,
    )

    raw_steps = [
        {"op": OP_THAW, "src": 1},                                                   # 0
        {"op": OP_THAW, "src": 2},                                                   # 1
        # Block A: water -> MM (P200)
        {"op": OP_PICK_TIP},                                                         # 2
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 0, "dst": 3, "vol": ul(90)}, # 3
        {"op": OP_DROP_TIP},                                                         # 4
        # Block B: buf -> MM (P20)
        {"op": OP_PICK_TIP},                                                         # 5
        {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 1, "dst": 3, "vol": ul(15)},  # 6
        {"op": OP_DROP_TIP},                                                         # 7
        # Block C: primer -> MM (P20)
        {"op": OP_PICK_TIP},                                                         # 8
        {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 2, "dst": 3, "vol": ul(15)},  # 9
        {"op": OP_DROP_TIP},                                                         # 10
        # Block D: dispense MM -> wellA1, wellA2 (P200)
        {"op": OP_PICK_TIP},                                                         # 11
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 3, "dst": 4, "vol": ul(50)}, # 12
        {"op": OP_TRANSFER, "pipette": PIP_P200, "src": 3, "dst": 5, "vol": ul(50)}, # 13
        {"op": OP_DROP_TIP},                                                         # 14
        # Final incubation
        {"op": OP_INCUBATE, "src": 4, "temp": 65, "duration": 60},                   # 15
    ]
    steps = ProtocolBatch.from_steps_list([raw_steps], device=dev)

    edges = [
        # Thaw before corresponding aspirate
        (0, 6),
        (1, 9),
        # Within-block tip order
        (2, 3), (3, 4),
        (5, 6), (6, 7),
        (8, 9), (9, 10),
        # Master mix complete before dispense
        (4, 11), (7, 11), (10, 11),
        (11, 12), (12, 13), (13, 14),
        # Dispense complete before incubate
        (14, 15),
    ]
    return StepDAGProblem(init_state=state, steps=steps, edges=edges)


def _gather_orders(steps_1s: ProtocolBatch, orders: Tensor) -> ProtocolBatch:
    """Build a (B, L) ProtocolBatch by indexing a (1, S) step catalog with `orders` (B, L)."""
    b, seq_len = orders.shape
    exp = steps_1s.repeat_batch(b)
    return ProtocolBatch(
        op=exp.op.gather(1, orders),
        src=exp.src.gather(1, orders),
        dst=exp.dst.gather(1, orders),
        vol=exp.vol.gather(1, orders),
        temp=exp.temp.gather(1, orders),
        duration=exp.duration.gather(1, orders),
        pipette=exp.pipette.gather(1, orders),
        rpm=exp.rpm.gather(1, orders),
        pad_mask=torch.zeros((b, seq_len), dtype=torch.bool, device=orders.device),
    )


def verify_dag_order(order: list[int], num_steps: int, edges: list[tuple[int, int]]) -> bool:
    """Return True iff `order` is a permutation of 0..num_steps-1 satisfying all DAG edges."""
    if len(order) != num_steps or set(order) != set(range(num_steps)):
        return False
    pos = {node: i for i, node in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in edges)


def sample_random_topological_orders(
    problem: StepDAGProblem,
    num_samples: int = 256,
    seed: int = 0,
) -> tuple[Tensor, float]:
    """Sample random topological sorts (ignoring physical rules) and check pass rate.

    Orders come from random-priority greedy selection, which is not uniform over all orderings.
    """
    validate_dag_problem(problem)
    s = problem.steps.max_steps
    if s > 62:
        raise ValueError("Bitmask DAG sampler supports up to 62 steps")
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    dev = problem.steps.device
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    pred_bits = torch.zeros((s,), dtype=torch.int64, device=dev)
    for u, v in problem.edges:
        pred_bits[v] |= 1 << u

    node_bit = (1 << torch.arange(s, dtype=torch.int64, device=dev)).unsqueeze(0)  # (1, S)
    done_mask = torch.zeros((num_samples,), dtype=torch.int64, device=dev)
    orders = torch.zeros((num_samples, s), dtype=torch.int64, device=dev)

    for depth in range(s):
        already = (done_mask.unsqueeze(1) & node_bit) != 0
        prereq_ok = (done_mask.unsqueeze(1) & pred_bits.unsqueeze(0)) == pred_bits.unsqueeze(0)
        avail = (~already) & prereq_ok  # (B, S)
        noise = torch.rand((num_samples, s), generator=g).to(dev)
        scores = torch.where(avail, noise, torch.full_like(noise, -1.0))
        pick = scores.argmax(dim=1)
        orders[:, depth] = pick
        done_mask = done_mask | (1 << pick)

    cand_proto = _gather_orders(problem.steps, orders)
    cand_state = problem.init_state.repeat_batch(num_samples)
    res = check_protocols_batched(cand_state, cand_proto)
    pass_rate = float(res.ok.float().mean().item())
    return orders, pass_rate


def batched_beam_plan_dag(
    problem: StepDAGProblem,
    beam_width: int = 32,
    require_tip_dropped: bool = False,
) -> DAGPlanResult:
    """Batched beam search over DAG step orderings gated at every depth by check_protocols_batched."""
    validate_dag_problem(problem)
    s = problem.steps.max_steps
    if s > 62:
        raise ValueError("Bitmask DAG planner supports up to 62 steps")
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")
    dev = problem.steps.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    pred_bits = torch.zeros((s,), dtype=torch.int64, device=dev)
    for u, v in problem.edges:
        pred_bits[v] |= 1 << u

    node_bit = (1 << torch.arange(s, dtype=torch.int64, device=dev)).unsqueeze(0)  # (1, S)
    step_pip = problem.steps.pipette[0]  # (S,)
    step_op = problem.steps.op[0]        # (S,)
    step_src = problem.steps.src[0]      # (S,)

    # Start with 1 empty beam
    beam_orders = torch.empty((1, 0), dtype=torch.int64, device=dev)
    beam_done = torch.zeros((1,), dtype=torch.int64, device=dev)
    beam_cost = torch.zeros((1,), dtype=torch.float32, device=dev)
    beam_last_pip = torch.full((1,), PIP_NONE, dtype=torch.int64, device=dev)
    beam_last_src = torch.full((1,), -1, dtype=torch.int64, device=dev)

    total_checked = 0

    for _depth in range(s):
        already = (beam_done.unsqueeze(1) & node_bit) != 0
        prereq_ok = (beam_done.unsqueeze(1) & pred_bits.unsqueeze(0)) == pred_bits.unsqueeze(0)
        avail = (~already) & prereq_ok  # (W, S)

        pairs = avail.nonzero(as_tuple=False)  # (N_cand, 2): [beam_idx, next_step]
        if pairs.numel() == 0:
            return DAGPlanResult(
                found=False,
                best_order=[],
                best_cost=float("inf"),
                best_protocol=None,
                num_valid_beams=0,
                total_candidates_checked=total_checked,
                elapsed_sec=time.perf_counter() - t0,
            )

        b_idx = pairs[:, 0]
        nxt = pairs[:, 1]
        cand_orders = torch.cat([beam_orders[b_idx], nxt.unsqueeze(1)], dim=1)
        n_cand = int(cand_orders.shape[0])
        total_checked += n_cand

        # Gate partial protocols in one batched checker pass
        cand_proto = _gather_orders(problem.steps, cand_orders)
        cand_state = problem.init_state.repeat_batch(n_cand)
        chk = check_protocols_batched(cand_state, cand_proto)

        ok_mask = chk.ok
        if require_tip_dropped and _depth == s - 1:
            ok_mask = ok_mask & (~chk.final_state.has_tip)
        valid_idx = ok_mask.nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return DAGPlanResult(
                found=False,
                best_order=[],
                best_cost=float("inf"),
                best_protocol=None,
                num_valid_beams=0,
                total_candidates_checked=total_checked,
                elapsed_sec=time.perf_counter() - t0,
            )

        b_idx = b_idx[valid_idx]
        nxt = nxt[valid_idx]
        cand_orders = cand_orders[valid_idx]

        # Heuristic step transition cost: penalize pipette tool switches & container jumps; reward early thaws
        nxt_pip = step_pip[nxt]
        nxt_op = step_op[nxt]
        nxt_src = step_src[nxt]
        prev_pip = beam_last_pip[b_idx]
        prev_src = beam_last_src[b_idx]

        pip_switch = (
            (nxt_pip != PIP_NONE) & (prev_pip != PIP_NONE) & (nxt_pip != prev_pip)
        ).to(torch.float32) * 1.0
        src_jump = ((prev_src >= 0) & (nxt_src != prev_src)).to(torch.float32) * 0.25
        early_thaw_bonus = (nxt_op == OP_THAW).to(torch.float32) * (-0.1)

        cand_cost = beam_cost[b_idx] + pip_switch + src_jump + early_thaw_bonus
        cand_done = beam_done[b_idx] | (1 << nxt)
        cand_last_pip = torch.where(nxt_pip != PIP_NONE, nxt_pip, prev_pip)
        cand_last_src = nxt_src

        # Keep top beam_width lowest-cost candidates
        k_keep = min(beam_width, int(cand_cost.numel()))
        top_idx = torch.topk(-cand_cost, k=k_keep, largest=True).indices
        beam_orders = cand_orders[top_idx]
        beam_done = cand_done[top_idx]
        beam_cost = cand_cost[top_idx]
        beam_last_pip = cand_last_pip[top_idx]
        beam_last_src = cand_last_src[top_idx]

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    best_i = int(beam_cost.argmin().item())
    best_ord_tensor = beam_orders[best_i : best_i + 1]
    best_proto = _gather_orders(problem.steps, best_ord_tensor)
    return DAGPlanResult(
        found=True,
        best_order=best_ord_tensor[0].to("cpu").tolist(),
        best_cost=float(beam_cost[best_i].item()),
        best_protocol=best_proto,
        num_valid_beams=int(beam_orders.shape[0]),
        total_candidates_checked=total_checked,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Part (b): Lab instrument scheduling (makespan minimization)
# ---------------------------------------------------------------------------


@dataclass
class ScheduleResult:
    method: str
    makespan: float
    assignment: Tensor  # (J,) int64 instrument id per job
    loads: Tensor       # (M,) float32 total duration per instrument
    elapsed_sec: float
    num_restarts: int = 1
    num_steps: int = 0


def _normalize_durations(
    durations: Tensor,
    num_instruments: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Normalize 1D (J,) or 2D (J, M) durations to `dtype` (J, M)."""
    if num_instruments < 1:
        raise ValueError(f"num_instruments must be >= 1, got {num_instruments}")
    if durations.ndim == 1:
        return durations.to(dtype).unsqueeze(1).expand(-1, num_instruments).contiguous()
    if durations.ndim == 2:
        if durations.shape[1] != num_instruments:
            raise ValueError(f"durations shape {tuple(durations.shape)} != (_, {num_instruments})")
        return durations.to(dtype).contiguous()
    raise ValueError("durations must be 1D (J,) or 2D (J, M)")


def make_lab_schedule_durations(
    num_jobs: int = 96,
    num_instruments: int = 8,
    heterogeneous: bool = True,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> Tensor:
    """Create synthetic lab job durations (seconds) across M instruments.

    If heterogeneous=True, instruments have mild speed differences and setup overheads per job
    (e.g. fast vs standard liquid handlers / thermocyclers), making assignment non-trivial.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    base = torch.randint(30, 420, (num_jobs,), generator=g).to(torch.float32)
    if not heterogeneous:
        return base.to(default_device(device))

    # Instrument speed factors (e.g. 0.85x .. 1.25x) + job-specific instrument overhead
    speed = torch.linspace(0.85, 1.25, num_instruments)
    jitter = torch.randint(0, 35, (num_jobs, num_instruments), generator=g).to(torch.float32)
    durs = (base.unsqueeze(1) * speed.unsqueeze(0) + jitter).round()
    return durs.to(default_device(device))


def compute_loads_and_makespan(
    durations: Tensor,
    assignment: Tensor,
    num_instruments: int,
) -> tuple[Tensor, float]:
    """Compute exact (M,) instrument loads and scalar makespan for one (J,) assignment."""
    d_jm64 = _normalize_durations(durations, num_instruments, dtype=torch.float64)
    j_idx = torch.arange(d_jm64.shape[0], device=d_jm64.device)
    picked = d_jm64[j_idx, assignment]
    loads64 = torch.zeros((num_instruments,), dtype=torch.float64, device=d_jm64.device)
    loads64.scatter_add_(0, assignment, picked)
    return loads64.to(torch.float32), float(loads64.max().item())


def schedule_greedy_list(
    durations: Tensor,
    num_instruments: int,
    sort_lpt: bool = False,
) -> ScheduleResult:
    """Greedy list-scheduling baseline (arrival order, or LPT when sort_lpt=True)."""
    dev = durations.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    d_jm = _normalize_durations(durations.to("cpu"), num_instruments)
    j_count = int(d_jm.shape[0])
    loads = [0.0] * num_instruments
    assign = [0] * j_count

    if sort_lpt:
        mean_dur = d_jm.mean(dim=1)
        job_order = torch.argsort(mean_dur, descending=True).tolist()
    else:
        job_order = list(range(j_count))

    d_list = d_jm.tolist()
    for j in job_order:
        row = d_list[j]
        # Pick instrument minimizing new completion time (loads[m] + d[j, m])
        best_m = min(range(num_instruments), key=lambda m: loads[m] + row[m])
        assign[j] = best_m
        loads[best_m] += row[best_m]

    assign_t = torch.tensor(assign, dtype=torch.int64, device=dev)
    loads_t = torch.tensor(loads, dtype=torch.float32, device=dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return ScheduleResult(
        method="greedy_lpt" if sort_lpt else "greedy_arrival",
        makespan=float(max(loads)),
        assignment=assign_t,
        loads=loads_t,
        elapsed_sec=elapsed,
    )


def schedule_batched_sa(
    durations: Tensor,
    num_instruments: int,
    num_restarts: int = 512,
    num_steps: int = 400,
    temp_start: float = 80.0,
    temp_end: float = 0.5,
    seed: int = 0,
) -> ScheduleResult:
    """Batched parallel simulated annealing over R restarts in PyTorch tensor ops."""
    dev = durations.device
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    d_jm = _normalize_durations(durations, num_instruments, dtype=torch.float64)
    num_jobs = int(d_jm.shape[0])
    r = num_restarts
    m = num_instruments

    g = torch.Generator(device=dev.type if dev.type == "cuda" else "cpu")
    g.manual_seed(seed)

    # Random initial assignments (R, J); accumulate loads in float64 to avoid rounding drift
    assign = torch.randint(0, m, (r, num_jobs), generator=g, device=dev)
    d_flat = d_jm.reshape(-1)
    j_base = (torch.arange(num_jobs, device=dev) * m).unsqueeze(0)  # (1, J)

    def _recompute_loads(cur_assign: Tensor) -> Tensor:
        job_durs = d_flat[(j_base + cur_assign).reshape(-1)].view(r, num_jobs)
        exact = torch.zeros((r, m), dtype=torch.float64, device=dev)
        exact.scatter_add_(1, cur_assign, job_durs)
        return exact

    loads = _recompute_loads(assign)
    cur_ms = loads.max(dim=1).values

    best_ms = cur_ms.clone()
    best_assign = assign.clone()

    # Pre-sample all random numbers in bulk; use O(1) 6-way tournament to pick bottleneck jobs
    tourn_j = torch.randint(0, num_jobs, (num_steps // 2 + 1, r, 6), generator=g, device=dev)
    rand_j1 = torch.randint(0, num_jobs, (num_steps, r), generator=g, device=dev)
    rand_j2_off = torch.randint(1, max(num_jobs, 2), (num_steps, r), generator=g, device=dev)
    rand_m_all = torch.randint(0, m, (num_steps, r), generator=g, device=dev)
    rand_u1 = torch.rand((num_steps, r), generator=g, device=dev)
    rand_u2 = torch.rand((num_steps, r), generator=g, device=dev)

    log_ratio = math.log(max(temp_end, 1e-4) / max(temp_start, 1e-4))

    for step in range(num_steps):
        frac = step / max(num_steps - 1, 1)
        temp = temp_start * math.exp(log_ratio * frac)

        if step % 2 == 0:
            # Move 1: O(1) 6-way tournament picks a job on a heavily loaded instrument
            tj = tourn_j[step // 2]                    # (R, 6)
            tm = assign.gather(1, tj)                  # (R, 6)
            tload = loads.gather(1, tm)                # (R, 6)
            win_k = tload.argmax(dim=1, keepdim=True)  # (R, 1)
            j1_2d = tj.gather(1, win_k)                # (R, 1)
            j1 = j1_2d.squeeze(1)
            old_m = tm.gather(1, win_k).squeeze(1)

            min_m = loads.argmin(dim=1)
            rand_m = rand_m_all[step]
            new_m = torch.where(rand_u1[step] < 0.6, min_m, rand_m)
            new_m = torch.where(new_m == old_m, (old_m + 1) % m, new_m)

            old_d = d_flat[j1 * m + old_m]
            new_d = d_flat[j1 * m + new_m]

            cand_loads = loads.clone()
            cand_loads.scatter_add_(1, old_m.unsqueeze(1), -old_d.unsqueeze(1))
            cand_loads.scatter_add_(1, new_m.unsqueeze(1), new_d.unsqueeze(1))
            cand_ms = cand_loads.max(dim=1).values

            delta = cand_ms - cur_ms
            prob = torch.exp((-delta / temp).clamp(min=-30.0, max=0.0))
            accept = (delta <= 0.0) | (rand_u2[step] < prob)

            loads = torch.where(accept.unsqueeze(1), cand_loads, loads)
            cur_ms = torch.where(accept, cand_ms, cur_ms)
            upd_m = torch.where(accept, new_m, old_m).unsqueeze(1)
            assign.scatter_(1, j1_2d, upd_m)

        else:
            # Move 2: swap instruments between two randomly chosen jobs j1 and j2
            j1 = rand_j1[step]
            j2 = (j1 + rand_j2_off[step]) % num_jobs
            j1_2d = j1.unsqueeze(1)
            j2_2d = j2.unsqueeze(1)
            m1 = assign.gather(1, j1_2d).squeeze(1)
            m2 = assign.gather(1, j2_2d).squeeze(1)
            m1_2d = m1.unsqueeze(1)
            m2_2d = m2.unsqueeze(1)

            d1_m1 = d_flat[j1 * m + m1].unsqueeze(1)
            d2_m2 = d_flat[j2 * m + m2].unsqueeze(1)
            d1_m2 = d_flat[j1 * m + m2].unsqueeze(1)
            d2_m1 = d_flat[j2 * m + m1].unsqueeze(1)

            cand_loads = loads.clone()
            cand_loads.scatter_add_(1, m1_2d, d2_m1 - d1_m1)
            cand_loads.scatter_add_(1, m2_2d, d1_m2 - d2_m2)
            cand_ms = cand_loads.max(dim=1).values

            delta = cand_ms - cur_ms
            prob = torch.exp((-delta / temp).clamp(min=-30.0, max=0.0))
            accept = (m1 != m2) & ((delta <= 0.0) | (rand_u2[step] < prob))

            loads = torch.where(accept.unsqueeze(1), cand_loads, loads)
            cur_ms = torch.where(accept, cand_ms, cur_ms)
            assign.scatter_(1, j1_2d, torch.where(accept, m2, m1).unsqueeze(1))
            assign.scatter_(1, j2_2d, torch.where(accept, m1, m2).unsqueeze(1))

        if (step + 1) % 64 == 0:
            loads = _recompute_loads(assign)
            cur_ms = loads.max(dim=1).values

        improved = cur_ms < best_ms
        if bool(improved.any()):
            best_ms = torch.where(improved, cur_ms, best_ms)
            imp_idx = improved.nonzero(as_tuple=False).squeeze(1)
            best_assign[imp_idx] = assign[imp_idx]

    winner = int(best_ms.argmin().item())
    win_assign = best_assign[winner]
    exact_loads, exact_ms = compute_loads_and_makespan(durations, win_assign, m)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return ScheduleResult(
        method="batched_sa",
        makespan=exact_ms,
        assignment=win_assign,
        loads=exact_loads,
        elapsed_sec=elapsed,
        num_restarts=r,
        num_steps=num_steps,
    )
