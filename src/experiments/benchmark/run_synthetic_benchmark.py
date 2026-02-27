"""Synthetic benchmark runner for MCMC-PESP.

Runs MCMC optimization on synthetic scenarios (hub, chain, conflicting, etc.)
and computes convergence diagnostics. Supports multiple chains for R-hat.

Usage:
    python -m src.experiments.benchmark.run_synthetic_benchmark \
        --config configs/synthetic_benchmark.yaml --output results/synthetic_test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.random import Generator

from src.core.schedule import Schedule
from src.cost.operational import OperationalCost, TurnaroundCost
from src.cost.robustness import DelayDistribution, RobustnessCost
from src.cost.total import TotalCost
from src.cost.transfer_cost import TransferCost
from src.io.synthetic_scenarios import SyntheticScenario, SyntheticScenarioFactory
from src.mcmc.proposals.component import create_component_move
from src.mcmc.proposals.single_line import SingleLineShift, UniformLineShift
from src.mcmc.proposals.smart_repair import create_smart_repair
from src.mcmc.sampler import MCMCSampler
from src.metrics.convergence import MCMCDiagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SyntheticBenchmarkConfig:
    """Configuration for synthetic benchmark run.

    Args:
        scenarios: List of scenario specs (type, difficulty, extra kwargs).
        n_chains: Number of independent MCMC chains per scenario.
        n_steps_per_chain: MCMC steps per chain.
        beta: Inverse temperature.
        burn_in_fraction: Fraction of steps to discard.
        thin: Thinning interval.
        use_component_move: Whether to include ComponentMove.
        use_smart_repair: Whether to include SmartRepairMove.
        pax_weight: Passenger cost weight.
        ops_weight: Operational cost weight.
        rob_weight: Robustness cost weight.
    """

    scenarios: list[dict[str, Any]] = field(default_factory=list)
    n_chains: int = 4
    n_steps_per_chain: int = 20000
    beta: float = 1.0
    burn_in_fraction: float = 0.1
    thin: int = 10
    use_component_move: bool = True
    use_smart_repair: bool = True
    pax_weight: float = 1.0
    ops_weight: float = 0.1
    rob_weight: float = 0.05


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioBenchmarkResult:
    """Result from benchmarking a single scenario.

    Args:
        scenario_name: Name of the scenario.
        scenario_type: Type identifier.
        difficulty: Difficulty level.
        metadata: Scenario metadata dict.
        best_cost: Best cost found across all chains.
        best_solution: Offsets at best cost.
        optimal_cost: Known optimal cost (None if unknown).
        gap_to_optimum: Relative gap to optimum (None if unknown).
        acceptance_rate: Mean acceptance rate across chains.
        diagnostics: Convergence diagnostics dict.
        proposal_effectiveness: Per-proposal acceptance rates (placeholder).
        runtime_seconds: Total wall-clock time.
    """

    scenario_name: str
    scenario_type: str
    difficulty: str
    metadata: dict[str, Any]
    best_cost: float
    best_solution: dict[str, int]
    optimal_cost: float | None
    gap_to_optimum: float | None
    acceptance_rate: float
    diagnostics: dict[str, Any]
    proposal_effectiveness: dict[str, float]
    runtime_seconds: float


@dataclass
class SyntheticBenchmarkSuite:
    """Aggregated results from full benchmark suite.

    Args:
        config: Benchmark configuration.
        results: List of per-scenario results.
    """

    config: SyntheticBenchmarkConfig
    results: list[ScenarioBenchmarkResult]

    def summary_table(self) -> list[dict[str, Any]]:
        """Return summary as a list of dicts (one per scenario)."""
        rows = []
        for r in self.results:
            row = {
                "scenario": r.scenario_name,
                "type": r.scenario_type,
                "difficulty": r.difficulty,
                "n_lines": r.metadata.get("n_lines", ""),
                "n_transfers": r.metadata.get("n_transfers", ""),
                "best_cost": round(r.best_cost, 4),
                "optimal_cost": r.optimal_cost,
                "gap": round(r.gap_to_optimum, 6) if r.gap_to_optimum is not None else None,
                "acceptance_rate": round(r.acceptance_rate, 4),
                "r_hat": r.diagnostics.get("r_hat"),
                "ess_total": r.diagnostics.get("ess_total"),
                "converged": r.diagnostics.get("converged"),
                "runtime_s": round(r.runtime_seconds, 2),
            }
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Cost function builder
# ---------------------------------------------------------------------------


def build_cost_fn(
    scenario: SyntheticScenario,
    config: SyntheticBenchmarkConfig,
) -> TransferCost | TotalCost:
    """Build cost function appropriate for the scenario.

    Scenarios with turnarounds use TotalCost; others use TransferCost
    (optionally wrapped in TotalCost with robustness).

    Args:
        scenario: The synthetic scenario.
        config: Benchmark config with cost weights.

    Returns:
        A cost function (TransferCost or TotalCost).
    """
    transfer_cost = TransferCost(transfers=scenario.transfers)

    # If scenario has turnarounds or is full_cost/multi_hub_mesh, use TotalCost
    needs_full = (
        scenario.turnarounds is not None
        or scenario.metadata.scenario_type in ("full_cost", "multi_hub_mesh")
    )

    if not needs_full:
        # Simple scenarios: just transfer cost (optionally with robustness)
        if config.rob_weight > 0:
            delay_dist = DelayDistribution(rate=1.0 / 3.0, alpha=0.95)
            rob_cost = RobustnessCost(
                transfers=scenario.transfers,
                delay_distribution=delay_dist,
            )
            return TotalCost(
                transfer_cost=transfer_cost,
                robustness_cost=rob_cost,
                pax_weight=config.pax_weight,
                rob_weight=config.rob_weight,
            )
        return transfer_cost

    # Full cost with operational + robustness
    ops_cost = None
    if scenario.turnarounds is not None:
        turnaround_cost = TurnaroundCost(turnarounds=scenario.turnarounds)
        ops_cost = OperationalCost(turnaround_cost=turnaround_cost)

    delay_dist = DelayDistribution(rate=1.0 / 3.0, alpha=0.95)
    rob_cost = RobustnessCost(
        transfers=scenario.transfers,
        delay_distribution=delay_dist,
    )

    return TotalCost(
        transfer_cost=transfer_cost,
        operational_cost=ops_cost,
        robustness_cost=rob_cost,
        pax_weight=config.pax_weight,
        ops_weight=config.ops_weight,
        rob_weight=config.rob_weight,
    )


# ---------------------------------------------------------------------------
# Proposal builder
# ---------------------------------------------------------------------------


def build_proposals(
    scenario: SyntheticScenario,
    config: SyntheticBenchmarkConfig,
    rng: Generator,
) -> tuple[list, list[float]]:
    """Build list of proposals and weights for a scenario.

    Args:
        scenario: The synthetic scenario.
        config: Benchmark config.
        rng: Random number generator.

    Returns:
        Tuple of (proposals, weights).
    """
    proposals = []
    weights = []

    # Single line shift (always)
    proposals.append(SingleLineShift(sigma=5.0, rng=rng))
    weights.append(0.4)

    # Uniform line shift
    proposals.append(UniformLineShift(max_shift=10, rng=rng))
    weights.append(0.1)

    # Component move
    if config.use_component_move:
        component = create_component_move(
            scenario.transfers, sigma=5.0, rng=rng
        )
        proposals.append(component)
        weights.append(0.3)

    # Smart repair
    if config.use_smart_repair:
        smart_repair = create_smart_repair(
            scenario.transfers, sigma=2.0, rng=rng
        )
        proposals.append(smart_repair)
        weights.append(0.2)

    return proposals, weights


# ---------------------------------------------------------------------------
# Single scenario benchmark
# ---------------------------------------------------------------------------


def run_scenario_benchmark(
    scenario: SyntheticScenario,
    config: SyntheticBenchmarkConfig,
    base_seed: int = 42,
) -> ScenarioBenchmarkResult:
    """Run multi-chain MCMC benchmark on a single scenario.

    Args:
        scenario: The synthetic scenario to benchmark.
        config: Benchmark configuration.
        base_seed: Base random seed.

    Returns:
        ScenarioBenchmarkResult with costs, diagnostics, etc.
    """
    start_time = time.time()
    name = scenario.metadata.name

    logger.info(f"Running scenario: {name}")
    logger.info(
        f"  {scenario.metadata.n_lines} lines, "
        f"{scenario.metadata.n_transfers} transfers"
    )

    chain_costs: list[list[float]] = []
    chain_best_costs: list[float] = []
    chain_best_offsets: list[dict[str, int]] = []
    chain_acceptance: list[float] = []

    burn_in = int(config.n_steps_per_chain * config.burn_in_fraction)

    for c in range(config.n_chains):
        chain_seed = base_seed + c
        rng = np.random.default_rng(chain_seed)

        cost_fn = build_cost_fn(scenario, config)
        proposals, proposal_weights = build_proposals(scenario, config, rng)

        sampler = MCMCSampler(
            network=scenario.network,
            cost_fn=cost_fn,
            proposals=proposals,
            proposal_weights=proposal_weights,
            beta=config.beta,
            rng=rng,
        )

        initial = Schedule.random(scenario.network, rng)
        sampler.initialize(initial)

        result = sampler.run(
            n_steps=config.n_steps_per_chain,
            thin=config.thin,
            burn_in=burn_in,
            log_every=0,
        )

        chain_costs.append(result.costs)
        chain_best_costs.append(result.best_cost)
        chain_best_offsets.append(result.best_schedule.offsets.copy())
        chain_acceptance.append(result.acceptance_rate)

    # Find overall best
    best_idx = int(np.argmin(chain_best_costs))
    best_cost = chain_best_costs[best_idx]
    best_offsets = chain_best_offsets[best_idx]
    mean_accept = float(np.mean(chain_acceptance))

    # Compute gap to optimum
    optimal_cost = None
    gap = None
    if scenario.metadata.optimal_solution is not None:
        cost_fn = build_cost_fn(scenario, config)
        opt_schedule = Schedule(
            offsets=dict(scenario.metadata.optimal_solution),
            period=scenario.network.period,
        )
        optimal_cost = cost_fn(opt_schedule, scenario.network)
        if optimal_cost > 0:
            gap = (best_cost - optimal_cost) / optimal_cost
        elif best_cost == 0:
            gap = 0.0
        else:
            gap = best_cost  # absolute gap when optimal=0

    # Convergence diagnostics
    diagnostics: dict[str, Any] = {}
    if config.n_chains >= 2:
        try:
            diag = MCMCDiagnostics(chains=chain_costs)
            summary = diag.summary()
            diagnostics = {
                "r_hat": summary.get("r_hat"),
                "ess_total": summary.get("ess_total"),
                "converged": summary.get("converged"),
            }
        except Exception as e:
            diagnostics = {"error": str(e)}
    else:
        diagnostics = {"note": "single chain"}

    elapsed = time.time() - start_time

    logger.info(
        f"  Best cost: {best_cost:.4f}, "
        f"accept: {mean_accept:.3f}, "
        f"time: {elapsed:.1f}s"
    )

    return ScenarioBenchmarkResult(
        scenario_name=name,
        scenario_type=scenario.metadata.scenario_type,
        difficulty=scenario.metadata.difficulty,
        metadata={
            "n_lines": scenario.metadata.n_lines,
            "n_transfers": scenario.metadata.n_transfers,
            "n_turnarounds": scenario.metadata.n_turnarounds,
        },
        best_cost=best_cost,
        best_solution=best_offsets,
        optimal_cost=optimal_cost,
        gap_to_optimum=gap,
        acceptance_rate=mean_accept,
        diagnostics=diagnostics,
        proposal_effectiveness={},
        runtime_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------


def run_synthetic_benchmark_suite(
    config: SyntheticBenchmarkConfig,
    base_seed: int = 42,
) -> SyntheticBenchmarkSuite:
    """Run benchmark suite across all configured scenarios.

    Args:
        config: Full benchmark configuration.
        base_seed: Base random seed.

    Returns:
        SyntheticBenchmarkSuite with all results.
    """
    results: list[ScenarioBenchmarkResult] = []

    for i, spec in enumerate(config.scenarios):
        scenario_type = spec.get("type", spec.get("scenario_type", ""))
        difficulty = spec.get("difficulty", "easy")

        # Extract extra kwargs (anything except type/difficulty)
        extra = {
            k: v for k, v in spec.items()
            if k not in ("type", "scenario_type", "difficulty")
        }

        scenario = SyntheticScenarioFactory.create(
            scenario_type, difficulty=difficulty, **extra
        )

        scenario_seed = base_seed + i * 1000
        result = run_scenario_benchmark(scenario, config, scenario_seed)
        results.append(result)

    return SyntheticBenchmarkSuite(config=config, results=results)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(
    suite: SyntheticBenchmarkSuite,
    output_dir: str | Path,
) -> None:
    """Save benchmark results to JSON and CSV.

    Args:
        suite: Benchmark suite with results.
        output_dir: Directory for output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON with full results
    json_data = {
        "config": {
            "n_chains": suite.config.n_chains,
            "n_steps_per_chain": suite.config.n_steps_per_chain,
            "beta": suite.config.beta,
            "pax_weight": suite.config.pax_weight,
            "ops_weight": suite.config.ops_weight,
            "rob_weight": suite.config.rob_weight,
        },
        "results": [
            {
                "scenario_name": r.scenario_name,
                "scenario_type": r.scenario_type,
                "difficulty": r.difficulty,
                "best_cost": r.best_cost,
                "optimal_cost": r.optimal_cost,
                "gap_to_optimum": r.gap_to_optimum,
                "acceptance_rate": r.acceptance_rate,
                "diagnostics": r.diagnostics,
                "runtime_seconds": r.runtime_seconds,
                "best_solution": r.best_solution,
            }
            for r in suite.results
        ],
    }

    json_path = out / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"Results written to {json_path}")

    # CSV summary
    rows = suite.summary_table()
    if rows:
        csv_path = out / "summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Summary written to {csv_path}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> SyntheticBenchmarkConfig:
    """Load benchmark config from YAML file.

    Args:
        path: Path to YAML config.

    Returns:
        SyntheticBenchmarkConfig.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    mcmc = raw.get("mcmc", {})
    scenarios = raw.get("scenarios", [])

    return SyntheticBenchmarkConfig(
        scenarios=scenarios,
        n_chains=mcmc.get("n_chains", 4),
        n_steps_per_chain=mcmc.get("n_steps_per_chain", 20000),
        beta=mcmc.get("beta", 1.0),
        burn_in_fraction=mcmc.get("burn_in_fraction", 0.1),
        thin=mcmc.get("thin", 10),
        use_component_move=mcmc.get("use_component_move", True),
        use_smart_repair=mcmc.get("use_smart_repair", True),
        pax_weight=mcmc.get("pax_weight", 1.0),
        ops_weight=mcmc.get("ops_weight", 0.1),
        rob_weight=mcmc.get("rob_weight", 0.05),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point for synthetic benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run MCMC-PESP synthetic benchmark suite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/synthetic_benchmark.yaml",
        help="Path to YAML config (default: configs/synthetic_benchmark.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/synthetic_benchmark",
        help="Output directory (default: results/synthetic_benchmark)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    config = load_config(config_path)

    logger.info("=" * 60)
    logger.info("MCMC-PESP Synthetic Benchmark")
    logger.info("=" * 60)
    logger.info(f"Scenarios: {len(config.scenarios)}")
    logger.info(f"Chains: {config.n_chains}, Steps: {config.n_steps_per_chain}")

    suite = run_synthetic_benchmark_suite(config, base_seed=args.seed)

    save_results(suite, args.output)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for row in suite.summary_table():
        print(
            f"{row['scenario']:30s} | "
            f"cost={row['best_cost']:10.4f} | "
            f"gap={str(row['gap']):>10s} | "
            f"accept={row['acceptance_rate']:.3f} | "
            f"R-hat={str(row['r_hat']):>6s} | "
            f"{row['runtime_s']:.1f}s"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
