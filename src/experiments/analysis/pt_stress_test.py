"""PT stress test on hard scenarios with tuned parameters.

Applies recommendations from initial PT diagnostics:
- M=8 replicas with inertia as baseline sweet spot
- Narrower beta range (0.5–3.0, ratio 6x)
- Also test M=12, M=16 for hard instances
- 500k step budget
- Scenarios: multi_hub_mesh_hard, random_network_hard

Usage:
    python -m src.experiments.analysis.pt_stress_test
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.core.schedule import Schedule
from src.experiments.analysis.diagnostic_runner import build_cost_fn, build_proposals
from src.io.synthetic_scenarios import SyntheticScenario, SyntheticScenarioFactory
from src.mcmc.parallel_tempering import InertiaConfig, ParallelTempering

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


INERTIA_CONFIG = InertiaConfig(
    modality_inertia={"IC": 20.0, "Regional": 5.0, "bus": 1.0},
)


@dataclass
class PTStressConfig:
    """Configuration for a single PT stress test run."""

    label: str
    n_replicas: int
    beta_min: float
    beta_max: float
    use_inertia: bool
    n_steps: int  # total budget, divided by n_replicas


@dataclass
class PTStressResult:
    """Result of a single PT stress test run."""

    label: str
    scenario_name: str
    n_replicas: int
    beta_min: float
    beta_max: float
    use_inertia: bool
    steps_per_replica: int
    best_cost: float
    cold_chain_mean_cost: float
    swap_rates: list[float]
    per_replica_acceptance: list[float]
    temperature_schedule: list[float]
    wall_time_s: float
    # Cost trajectory (sampled)
    cost_checkpoints: list[dict[str, float]]


def run_single_config(
    scenario: SyntheticScenario,
    config: PTStressConfig,
    seed: int = 42,
    checkpoint_every: int = 5000,
) -> PTStressResult:
    """Run a single PT configuration."""
    rng = np.random.default_rng(seed)
    cost_fn = build_cost_fn(scenario)
    proposals, weights = build_proposals(scenario, rng)

    steps_per_replica = config.n_steps // max(config.n_replicas, 1)

    if config.n_replicas == 1:
        # Single chain baseline
        from src.mcmc.sampler import MCMCSampler

        t0 = time.perf_counter()
        sampler = MCMCSampler(
            network=scenario.network,
            cost_fn=cost_fn,
            proposals=proposals,
            proposal_weights=weights,
            beta=config.beta_max,
            mode="optimization",
            rng=rng,
        )
        sampler.initialize(Schedule.random(scenario.network, rng))

        checkpoints = []
        for step in range(1, config.n_steps + 1):
            sampler.step()
            if step % checkpoint_every == 0:
                checkpoints.append({
                    "step": step,
                    "best_cost": sampler.best_cost,
                    "current_cost": sampler.current_cost,
                })

        wall_time = time.perf_counter() - t0

        return PTStressResult(
            label=config.label,
            scenario_name=scenario.metadata.name,
            n_replicas=1,
            beta_min=config.beta_max,
            beta_max=config.beta_max,
            use_inertia=False,
            steps_per_replica=config.n_steps,
            best_cost=sampler.best_cost,
            cold_chain_mean_cost=checkpoints[-1]["current_cost"] if checkpoints else sampler.current_cost,
            swap_rates=[],
            per_replica_acceptance=[sampler.acceptance_rate],
            temperature_schedule=[config.beta_max],
            wall_time_s=wall_time,
            cost_checkpoints=checkpoints,
        )

    # PT run
    inertia = INERTIA_CONFIG if config.use_inertia else None

    t0 = time.perf_counter()
    pt = ParallelTempering(
        network=scenario.network,
        cost_fn=cost_fn,
        proposals=proposals,
        proposal_weights=weights,
        n_replicas=config.n_replicas,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        swap_interval=10,
        mode="optimization",
        inertia_config=inertia,
        rng=rng,
    )
    pt.initialize()

    checkpoints = []
    for step in range(1, steps_per_replica + 1):
        pt.step()
        if step % (checkpoint_every // config.n_replicas) == 0:
            checkpoints.append({
                "step": step * config.n_replicas,  # effective budget used
                "best_cost": pt.best_cost,
            })

    wall_time = time.perf_counter() - t0

    pt_result = pt.run(n_steps=0, log_every=0)  # just collect final stats

    return PTStressResult(
        label=config.label,
        scenario_name=scenario.metadata.name,
        n_replicas=config.n_replicas,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        use_inertia=config.use_inertia,
        steps_per_replica=steps_per_replica,
        best_cost=pt.best_cost,
        cold_chain_mean_cost=float(np.mean(pt_result.cold_chain_costs)) if pt_result.cold_chain_costs else pt.best_cost,
        swap_rates=list(pt_result.swap_acceptance_rates),
        per_replica_acceptance=list(pt_result.per_replica_acceptance_rates),
        temperature_schedule=list(pt.temperature_schedule),
        wall_time_s=wall_time,
        cost_checkpoints=checkpoints,
    )


def run_stress_test(seed: int = 42) -> dict[str, Any]:
    """Run the full PT stress test suite."""
    factory = SyntheticScenarioFactory()

    N_STEPS = 500_000

    # Configurations to test
    configs = [
        PTStressConfig("baseline_1chain", 1, 3.0, 3.0, False, N_STEPS),
        PTStressConfig("M8_narrow", 8, 0.5, 3.0, False, N_STEPS),
        PTStressConfig("M8_narrow_inertia", 8, 0.5, 3.0, True, N_STEPS),
        PTStressConfig("M12_narrow", 12, 0.5, 3.0, False, N_STEPS),
        PTStressConfig("M12_narrow_inertia", 12, 0.5, 3.0, True, N_STEPS),
        PTStressConfig("M16_narrow", 16, 0.5, 3.0, False, N_STEPS),
        PTStressConfig("M16_narrow_inertia", 16, 0.5, 3.0, True, N_STEPS),
    ]

    scenarios = [
        ("multi_hub_mesh", "hard", {}),
        ("random_network", "hard", {"seed": 42}),
    ]

    all_results: dict[str, list[dict]] = {}

    for stype, diff, kwargs in scenarios:
        scenario = factory.create(stype, difficulty=diff, **kwargs)
        scenario_key = scenario.metadata.name
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_key}")
        print(f"  Lines: {scenario.metadata.n_lines}, Transfers: {scenario.metadata.n_transfers}")
        print(f"  Budget: {N_STEPS} steps")
        print(f"{'='*80}")

        scenario_results = []

        for cfg in configs:
            print(f"\n  Running {cfg.label} (M={cfg.n_replicas}, "
                  f"beta=[{cfg.beta_min}, {cfg.beta_max}], "
                  f"inertia={cfg.use_inertia})...")

            result = run_single_config(scenario, cfg, seed=seed, checkpoint_every=10000)

            swap_avg = f"{np.mean(result.swap_rates):.3f}" if result.swap_rates else "N/A"
            swap_min = f"{min(result.swap_rates):.3f}" if result.swap_rates else "N/A"
            print(f"    best_cost={result.best_cost:.1f}  "
                  f"cold_mean={result.cold_chain_mean_cost:.1f}  "
                  f"swap_avg={swap_avg}  swap_min={swap_min}  "
                  f"time={result.wall_time_s:.1f}s")

            scenario_results.append({
                "label": result.label,
                "n_replicas": result.n_replicas,
                "beta_min": result.beta_min,
                "beta_max": result.beta_max,
                "use_inertia": result.use_inertia,
                "steps_per_replica": result.steps_per_replica,
                "best_cost": result.best_cost,
                "cold_chain_mean_cost": result.cold_chain_mean_cost,
                "swap_rates": result.swap_rates,
                "per_replica_acceptance": result.per_replica_acceptance,
                "temperature_schedule": result.temperature_schedule,
                "wall_time_s": result.wall_time_s,
                "cost_checkpoints": result.cost_checkpoints,
            })

        all_results[scenario_key] = scenario_results

    # Summary table
    print(f"\n\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"{'Scenario':<30} {'Config':<25} {'Best Cost':>12} {'Cold Mean':>12} "
          f"{'Swap Avg':>10} {'Swap Min':>10} {'Time(s)':>8}")
    print("-" * 120)
    for scenario_key, results in all_results.items():
        for r in results:
            swap_avg = f"{np.mean(r['swap_rates']):.3f}" if r['swap_rates'] else "N/A"
            swap_min = f"{min(r['swap_rates']):.3f}" if r['swap_rates'] else "N/A"
            print(f"{scenario_key:<30} {r['label']:<25} {r['best_cost']:>12.1f} "
                  f"{r['cold_chain_mean_cost']:>12.1f} {swap_avg:>10} {swap_min:>10} "
                  f"{r['wall_time_s']:>8.1f}")

    return all_results


if __name__ == "__main__":
    results = run_stress_test()

    out_path = Path("results/diagnostics/pt_stress_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
