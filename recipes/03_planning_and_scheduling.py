#!/usr/bin/env python3
"""Recipe 3: Batched DAG step planning and parallel SA instrument scheduling."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labfill.planner import (
    batched_beam_plan_dag,
    make_assay_dag_problem,
    make_lab_schedule_durations,
    sample_random_topological_orders,
    schedule_batched_sa,
    schedule_greedy_list,
    verify_dag_order,
)
from labfill.protocol import OP_NAMES, default_device


def main() -> None:
    dev = default_device()
    print(f"=== Recipe 3: Batched Planning & Instrument Scheduling (device={dev}) ===\n")

    # Part (a): Batched beam search over DAG step orderings
    dag = make_assay_dag_problem(device=dev)
    _, rand_topo_pass = sample_random_topological_orders(dag, num_samples=512, seed=0)
    plan = batched_beam_plan_dag(dag, beam_width=32)
    dag_ok = verify_dag_order(plan.best_order, dag.steps.max_steps, dag.edges)

    print("--- (a) Step-Ordering Beam Search under Precedence DAG ---")
    print(f"Random topological order pass rate : {rand_topo_pass:.1%}")
    print(f"Checker-gated beam search found    : {plan.found} (DAG valid={dag_ok})")
    print(f"Candidates checked / elapsed       : {plan.total_candidates_checked} / {plan.elapsed_sec * 1000:.1f} ms")
    ops_ordered = [OP_NAMES[int(dag.steps.op[0, i].item())] for i in plan.best_order]
    print(f"Best step order ({len(plan.best_order)} steps)     : {plan.best_order}")
    print(f"Ordered ops                        : {', '.join(ops_ordered)}\n")

    # Part (b): Batched parallel simulated annealing for instrument makespan
    print("--- (b) Parallel Instrument Scheduling (96 jobs on 8 instruments) ---")
    print("Note: Use dedicated MILP / CP-SAT / GPU LP-routing solvers for large constrained instances.")
    durs = make_lab_schedule_durations(num_jobs=96, num_instruments=8, heterogeneous=True, seed=0, device=dev)
    g_arr = schedule_greedy_list(durs, 8, sort_lpt=False)
    g_lpt = schedule_greedy_list(durs, 8, sort_lpt=True)
    sa = schedule_batched_sa(durs, 8, num_restarts=256, num_steps=300, seed=0)

    for res in (g_arr, g_lpt, sa):
        print(
            f"  {res.method:<16s} | makespan = {res.makespan:7.1f} s | "
            f"time = {res.elapsed_sec * 1000:7.2f} ms"
        )


if __name__ == "__main__":
    main()
