from __future__ import annotations

from labfill.checker import check_protocols_batched
from labfill.planner import (
    batched_beam_plan_dag,
    compute_loads_and_makespan,
    make_assay_dag_problem,
    make_lab_schedule_durations,
    sample_random_topological_orders,
    schedule_batched_sa,
    schedule_greedy_list,
    verify_dag_order,
)


def test_dag_beam_planner() -> None:
    dag = make_assay_dag_problem()
    _, rand_rate = sample_random_topological_orders(dag, num_samples=128, seed=0)
    assert 0.0 <= rand_rate < 0.5  # Most naive topological sorts violate tip rules

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
