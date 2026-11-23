from __future__ import annotations

import pytest
import torch

from labfill.checker import check_protocols_batched
from labfill.planner import (
    StepDAGProblem,
    batched_beam_plan_dag,
    compute_loads_and_makespan,
    make_assay_dag_problem,
    make_lab_schedule_durations,
    sample_random_topological_orders,
    schedule_batched_sa,
    schedule_greedy_list,
    verify_dag_order,
)
from labfill.protocol import OP_PICK_TIP, OP_TRANSFER, PIP_P20, LabStateBatch, ProtocolBatch, ul


def test_dag_beam_planner() -> None:
    dag = make_assay_dag_problem()
    _, rand_rate = sample_random_topological_orders(dag, num_samples=128, seed=0)
    # Naive topological sorts almost always violate physical rules (seed 0 passes 2 of 128)
    assert 0.0 <= rand_rate <= 0.1

    plan = batched_beam_plan_dag(dag, beam_width=24)
    assert plan.found is True
    assert verify_dag_order(plan.best_order, dag.steps.max_steps, dag.edges)
    assert plan.best_protocol is not None
    chk = check_protocols_batched(dag.init_state, plan.best_protocol)
    assert bool(chk.ok.item()) is True


def test_instrument_makespan_schedulers() -> None:
    durs = make_lab_schedule_durations(num_jobs=64, num_instruments=6, heterogeneous=True, seed=3)
    g_arr = schedule_greedy_list(durs, 6, sort_lpt=False)
    g_lpt = schedule_greedy_list(durs, 6, sort_lpt=True)
    sa = schedule_batched_sa(durs, 6, num_restarts=128, num_steps=200, seed=3)

    for r in (g_arr, g_lpt, sa):
        assert r.assignment.shape == (64,)
        assert bool(((r.assignment >= 0) & (r.assignment < 6)).all().item())
        _, exact_ms = compute_loads_and_makespan(durs, r.assignment, 6)
        assert abs(exact_ms - r.makespan) < 1e-3

    # Batched SA should beat naive arrival-order greedy
    assert sa.makespan <= g_arr.makespan


def test_dag_validation_rejects_cycles_and_out_of_bounds_edges() -> None:
    dag = make_assay_dag_problem()
    with pytest.raises(ValueError, match="out of valid step range"):
        StepDAGProblem(init_state=dag.init_state, steps=dag.steps, edges=[(0, 99)])
    with pytest.raises(ValueError, match="out of valid step range"):
        StepDAGProblem(init_state=dag.init_state, steps=dag.steps, edges=[(-1, 2)])
    with pytest.raises(ValueError, match="Self-loop"):
        StepDAGProblem(init_state=dag.init_state, steps=dag.steps, edges=[(3, 3)])
    with pytest.raises(ValueError, match="cycle"):
        StepDAGProblem(init_state=dag.init_state, steps=dag.steps, edges=[(0, 1), (1, 2), (2, 0)])

    # Mutating edges after construction is still caught by planner entry points
    dag.edges.append((15, 0))
    with pytest.raises(ValueError, match="cycle"):
        sample_random_topological_orders(dag, num_samples=16)
    with pytest.raises(ValueError, match="cycle"):
        batched_beam_plan_dag(dag, beam_width=8)

    # Mounted-tip completion guard when require_tip_dropped=True
    st2 = LabStateBatch(
        volume=torch.tensor([[ul(100), 0]], dtype=torch.int64),
        capacity=torch.tensor([[ul(200), ul(200)]], dtype=torch.int64),
        sealed=torch.zeros((1, 2), dtype=torch.bool),
        frozen=torch.zeros((1, 2), dtype=torch.bool),
    )
    pr2 = ProtocolBatch.from_steps_list(
        [[{"op": OP_PICK_TIP}, {"op": OP_TRANSFER, "pipette": PIP_P20, "src": 0, "dst": 1, "vol": ul(10)}]]
    )
    dangling = StepDAGProblem(init_state=st2, steps=pr2, edges=[(0, 1)])
    assert batched_beam_plan_dag(dangling, beam_width=4, require_tip_dropped=False).found is True
    assert batched_beam_plan_dag(dangling, beam_width=4, require_tip_dropped=True).found is False


def test_schedule_batched_sa_non_integer_durations_no_drift() -> None:
    g = torch.Generator().manual_seed(123)
    # High dynamic range + fractional float32 values prone to float32 scatter_add_ drift
    durs = 10000.0 + torch.rand((48, 6), generator=g, dtype=torch.float32) * 3.137
    sa = schedule_batched_sa(durs, num_instruments=6, num_restarts=64, num_steps=2000, seed=7)
    exact_loads, exact_ms = compute_loads_and_makespan(durs, sa.assignment, 6)
    assert abs(sa.makespan - exact_ms) < 1e-6
    assert torch.allclose(sa.loads, exact_loads, atol=1e-4)
