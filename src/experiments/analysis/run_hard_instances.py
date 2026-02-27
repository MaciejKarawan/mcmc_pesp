"""Run cost calibration and profiling experiments on hard instances.

Uruchamia:
- run_barrier_sensitivity  na multi_hub_mesh/hard i random_network/hard
- run_lambda_sweep         na multi_hub_mesh/hard i random_network/hard
- run_profiling_analysis   na multi_hub_mesh/hard i random_network/hard

Wyniki zapisuje do results/hard_instances/<scenario>_{barrier|lambda|profiling}.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.analysis.diagnostic_runner import (
    BarrierSensitivityResult,
    LambdaSweepResult,
    ProfilingResult,
    compute_pareto_front,
    run_barrier_sensitivity,
    run_lambda_sweep,
    run_profiling_analysis,
)
from src.io.synthetic_scenarios import SyntheticScenarioFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Konfiguracja
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42

M_VALUES = [1e3, 1e4, 1e5, 1e6, 1e7]

LAMBDA_GRID = {
    "pax": [1.0],
    "ops": [0.1, 1.0, 10.0],
    "rob": [0.01, 0.1, 1.0],
}

N_STEPS_BARRIER = 5000
N_STEPS_LAMBDA = 5000
N_STEPS_PROFILING = 2000

SCENARIOS = [
    {"type": "multi_hub_mesh", "difficulty": "hard"},
    {"type": "random_network", "difficulty": "hard", "seed": 42},
]

OUTPUT_DIR = PROJECT_ROOT / "results" / "hard_instances"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _barrier_to_dict(result: BarrierSensitivityResult) -> dict:
    return {
        "scenario_name": result.scenario_name,
        "points": [
            {
                "M_barrier": p.M_barrier,
                "best_cost": p.best_cost,
                "mean_cost": p.mean_cost,
                "violation_rate": p.violation_rate,
                "acceptance_rate": p.acceptance_rate,
            }
            for p in result.points
        ],
    }


def _lambda_to_dict(result: LambdaSweepResult) -> dict:
    return {
        "scenario_name": result.scenario_name,
        "points": [
            {
                "lambda_pax": p.lambda_pax,
                "lambda_ops": p.lambda_ops,
                "lambda_rob": p.lambda_rob,
                "best_cost": p.best_cost,
                "cost_pax": p.cost_pax,
                "cost_ops": p.cost_ops,
                "cost_rob": p.cost_rob,
                "acceptance_rate": p.acceptance_rate,
            }
            for p in result.points
        ],
        "pareto_pax_ops": {
            "points": compute_pareto_front(result.points, "cost_pax", "cost_ops").points,
            "dominated": compute_pareto_front(result.points, "cost_pax", "cost_ops").dominated_points,
        },
        "pareto_pax_rob": {
            "points": compute_pareto_front(result.points, "cost_pax", "cost_rob").points,
            "dominated": compute_pareto_front(result.points, "cost_pax", "cost_rob").dominated_points,
        },
    }


def _profiling_to_dict(result: ProfilingResult) -> dict:
    return {
        "scenario_name": result.scenario_name,
        "n_steps": result.n_steps,
        "total_time_s": result.total_time_s,
        "breakdown": [
            {
                "component": p.component,
                "time_ms": p.time_ms,
                "percentage": p.percentage,
            }
            for p in result.breakdown
        ],
    }


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Saved → %s", path.relative_to(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    factory = SyntheticScenarioFactory()

    for spec in SCENARIOS:
        stype = spec["type"]
        diff = spec["difficulty"]
        kwargs = {k: v for k, v in spec.items() if k not in ("type", "difficulty")}

        scenario = factory.create(stype, difficulty=diff, **kwargs)
        name = scenario.metadata.name
        log.info(
            "Scenario: %s  [%dL %dT]",
            name,
            scenario.metadata.n_lines,
            scenario.metadata.n_transfers,
        )

        # ── 1. Barrier sensitivity ─────────────────────────────────────────
        log.info("  [barrier] M = %s  (%d steps each)", M_VALUES, N_STEPS_BARRIER)
        barrier_result = run_barrier_sensitivity(
            scenario,
            M_values=M_VALUES,
            n_steps=N_STEPS_BARRIER,
            beta=1.0,
            seed=SEED,
        )
        for p in barrier_result.points:
            log.info(
                "    M=%.0e  best=%.1f  viol=%.1f%%  acc=%.3f",
                p.M_barrier, p.best_cost, p.violation_rate * 100, p.acceptance_rate,
            )
        save(_barrier_to_dict(barrier_result), OUTPUT_DIR / f"{name}_barrier.json")

        # ── 2. Lambda sweep ───────────────────────────────────────────────
        n_configs = (
            len(LAMBDA_GRID["pax"]) * len(LAMBDA_GRID["ops"]) * len(LAMBDA_GRID["rob"])
        )
        log.info("  [lambda]  %d configs  (%d steps each)", n_configs, N_STEPS_LAMBDA)
        lambda_result = run_lambda_sweep(
            scenario,
            lambda_grid=LAMBDA_GRID,
            n_steps=N_STEPS_LAMBDA,
            beta=1.0,
            seed=SEED,
        )
        best_point = min(lambda_result.points, key=lambda p: p.best_cost)
        log.info(
            "    best: λ_pax=%.1f λ_ops=%.1f λ_rob=%.2f → cost=%.1f (PAX=%.1f OPS=%.1f ROB=%.1f)",
            best_point.lambda_pax,
            best_point.lambda_ops,
            best_point.lambda_rob,
            best_point.best_cost,
            best_point.cost_pax,
            best_point.cost_ops,
            best_point.cost_rob,
        )
        save(_lambda_to_dict(lambda_result), OUTPUT_DIR / f"{name}_lambda.json")

        # ── 3. Profiling ──────────────────────────────────────────────────
        log.info("  [profiling] %d steps", N_STEPS_PROFILING)
        profiling_result = run_profiling_analysis(
            scenario,
            n_steps=N_STEPS_PROFILING,
            seed=SEED,
        )
        for p in profiling_result.breakdown:
            log.info("    %-22s %7.3f ms/step (%5.1f%%)", p.component, p.time_ms, p.percentage)
        save(_profiling_to_dict(profiling_result), OUTPUT_DIR / f"{name}_profiling.json")

    log.info("Done. Results in %s", OUTPUT_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
