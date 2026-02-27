"""PESPlib benchmark runner for MCMC-PESP.

This script runs MCMC optimization on PESPlib instances and computes
convergence diagnostics. Supports multiple chains for R-hat computation.

Usage:
    python -m src.experiments.benchmark.run_pesplib --instance mini
    python -m src.experiments.benchmark.run_pesplib --instance toy --n_chains 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

from src.core.schedule import Schedule
from src.cost.operational import OperationalCost, TurnaroundConstraint
from src.cost.robustness import DelayDistribution, RobustnessCost
from src.cost.total import TotalCost
from src.cost.transfer_cost import TransferCost
from src.io.pesplib_loader import (
    PESPActivity,
    compute_pesplib_objective,
    load_pesplib_instance,
)
from src.io.synthetic_pesplib import ensure_synthetic_instances
from src.mcmc.proposals.component import create_component_move
from src.mcmc.proposals.single_line import SingleLineShift, UniformLineShift
from src.mcmc.proposals.smart_repair import create_smart_repair
from src.mcmc.sampler import MCMCResult, MCMCSampler
from src.metrics.convergence import MCMCDiagnostics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run.

    Args:
        n_chains: Number of independent MCMC chains.
        n_steps_per_chain: Number of MCMC steps per chain.
        beta: Inverse temperature for MCMC.
        burn_in_fraction: Fraction of samples to discard as burn-in.
        thin: Thinning interval for samples.
        use_component_move: Whether to use ComponentMove proposal.
        use_smart_repair: Whether to use SmartRepairMove proposal.
        pax_weight: Weight for passenger cost.
        ops_weight: Weight for operational cost.
        rob_weight: Weight for robustness cost.
    """

    n_chains: int = 4
    n_steps_per_chain: int = 25000
    beta: float = 1.0
    burn_in_fraction: float = 0.1
    thin: int = 10
    use_component_move: bool = True
    use_smart_repair: bool = True
    pax_weight: float = 1.0
    ops_weight: float = 0.1
    rob_weight: float = 0.05


@dataclass
class ChainResult:
    """Result from a single MCMC chain.

    Args:
        chain_id: Identifier for this chain.
        costs: List of sampled costs.
        best_cost: Best cost found in this chain.
        best_offsets: Offsets at best cost.
        acceptance_rate: Overall acceptance rate.
        per_proposal_accepts: Acceptance counts per proposal.
    """

    chain_id: int
    costs: list[float]
    best_cost: float
    best_offsets: dict[str, int]
    acceptance_rate: float
    per_proposal_accepts: dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated result from benchmark run.

    Args:
        instance_name: Name of the PESPlib instance.
        config: Benchmark configuration.
        chain_results: Results from each chain.
        best_solution: Overall best solution found.
        pesplib_objective: PESPlib objective at best solution.
        diagnostics: Convergence diagnostics.
    """

    instance_name: str
    config: BenchmarkConfig
    chain_results: list[ChainResult]
    best_solution: dict[str, int]
    best_cost: float
    pesplib_objective: float
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "instance": self.instance_name,
            "n_chains": self.config.n_chains,
            "n_steps_per_chain": self.config.n_steps_per_chain,
            "best_solution": {
                "offsets": self.best_solution,
                "our_cost": self.best_cost,
                "pesplib_objective": self.pesplib_objective,
            },
            "diagnostics": self.diagnostics,
            "per_chain_results": [
                {
                    "chain_id": cr.chain_id,
                    "best_cost": cr.best_cost,
                    "acceptance_rate": cr.acceptance_rate,
                    "n_samples": len(cr.costs),
                }
                for cr in self.chain_results
            ],
        }


def run_single_chain(
    chain_id: int,
    network,
    transfers,
    config: BenchmarkConfig,
    seed: int,
) -> ChainResult:
    """Run a single MCMC chain.

    Args:
        chain_id: Identifier for this chain.
        network: Network definition.
        transfers: Transfer set.
        config: Benchmark configuration.
        seed: Random seed for reproducibility.

    Returns:
        ChainResult with samples and diagnostics.
    """
    logger.info(f"Starting chain {chain_id} with seed {seed}")

    rng = np.random.default_rng(seed)

    # Create cost function
    transfer_cost = TransferCost(transfers=transfers)

    # Create operational cost (simplified - no turnarounds for now)
    operational_cost = None

    # Create robustness cost
    delay_dist = DelayDistribution(rate=1.0 / 3.0, alpha=0.95)  # Mean delay = 3 min
    robustness_cost = RobustnessCost(
        transfers=transfers,
        delay_distribution=delay_dist,
    )

    total_cost = TotalCost(
        transfer_cost=transfer_cost,
        operational_cost=operational_cost,
        robustness_cost=robustness_cost,
        pax_weight=config.pax_weight,
        ops_weight=config.ops_weight,
        rob_weight=config.rob_weight,
    )

    # Create proposals
    proposals = []
    proposal_weights = []

    # Single line shift (always included)
    proposals.append(SingleLineShift(sigma=5.0, rng=rng))
    proposal_weights.append(0.4)

    # Uniform line shift
    proposals.append(UniformLineShift(max_shift=10, rng=rng))
    proposal_weights.append(0.1)

    # Component move
    if config.use_component_move:
        component = create_component_move(transfers, sigma=5.0, rng=rng)
        proposals.append(component)
        proposal_weights.append(0.3)

    # Smart repair
    if config.use_smart_repair:
        smart_repair = create_smart_repair(transfers, sigma=2.0, rng=rng)
        proposals.append(smart_repair)
        proposal_weights.append(0.2)

    # Create sampler
    sampler = MCMCSampler(
        network=network,
        cost_fn=total_cost,
        proposals=proposals,
        proposal_weights=proposal_weights,
        beta=config.beta,
        rng=rng,
    )

    # Initialize with random schedule
    initial_schedule = Schedule.random(network, rng)
    sampler.initialize(initial_schedule)

    # Compute burn-in and run
    burn_in = int(config.n_steps_per_chain * config.burn_in_fraction)

    result = sampler.run(
        n_steps=config.n_steps_per_chain,
        thin=config.thin,
        burn_in=burn_in,
        log_every=5000,
    )

    logger.info(
        f"Chain {chain_id} complete: best_cost={result.best_cost:.2f}, "
        f"accept_rate={result.acceptance_rate:.3f}"
    )

    return ChainResult(
        chain_id=chain_id,
        costs=result.costs,
        best_cost=result.best_cost,
        best_offsets=result.best_schedule.offsets.copy(),
        acceptance_rate=result.acceptance_rate,
    )


def run_multi_chain_benchmark(
    instance_path: Path,
    config: BenchmarkConfig,
    base_seed: int = 42,
) -> BenchmarkResult:
    """Run multi-chain MCMC benchmark on a PESPlib instance.

    Args:
        instance_path: Path to PESPlib instance directory.
        config: Benchmark configuration.
        base_seed: Base random seed (chains use base_seed + chain_id).

    Returns:
        BenchmarkResult with aggregated results and diagnostics.
    """
    instance_name = instance_path.name

    logger.info("=" * 60)
    logger.info(f"MCMC-PESP Benchmark: {instance_name}")
    logger.info("=" * 60)

    # Load instance
    logger.info(f"Loading PESPlib instance from {instance_path}")
    network, transfers, event_mapping = load_pesplib_instance(instance_path)

    logger.info(f"Network: {len(network)} lines, period={network.period}")
    logger.info(f"Transfers: {len(transfers)}")

    # Run chains
    chain_results: list[ChainResult] = []

    for i in range(config.n_chains):
        chain_seed = base_seed + i
        result = run_single_chain(
            chain_id=i,
            network=network,
            transfers=transfers,
            config=config,
            seed=chain_seed,
        )
        chain_results.append(result)

    # Find overall best solution
    best_chain = min(chain_results, key=lambda r: r.best_cost)
    best_offsets = best_chain.best_offsets
    best_cost = best_chain.best_cost

    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATED RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best cost found: {best_cost:.4f}")
    logger.info(f"Best offsets: {best_offsets}")

    # Compute PESPlib objective for best solution
    best_schedule = Schedule(offsets=best_offsets, period=network.period)

    # We need the original activities for objective computation
    # Reload to get activities
    from src.io.pesplib_loader import PESPLibLoader

    loader = PESPLibLoader(instance_path)
    activities_path = loader._find_file("Activities.giv")
    activities = loader._parse_activities(activities_path)

    pesplib_obj = compute_pesplib_objective(
        best_schedule, network, activities, event_mapping
    )
    logger.info(f"PESPlib objective: {pesplib_obj:.2f}")

    # Compute convergence diagnostics
    cost_chains = [r.costs for r in chain_results]

    diagnostics: dict[str, Any] = {}

    if len(cost_chains) >= 2:
        try:
            mcmc_diag = MCMCDiagnostics(chains=cost_chains)
            diag_summary = mcmc_diag.summary()
            diagnostics = {
                "r_hat": diag_summary.get("r_hat"),
                "ess_total": diag_summary.get("ess_total"),
                "ess_per_chain": diag_summary.get("ess_per_chain"),
                "converged": diag_summary.get("converged"),
            }
            logger.info(f"R-hat: {diagnostics['r_hat']:.4f}")
            logger.info(f"ESS total: {diagnostics['ess_total']:.1f}")
            logger.info(f"Converged: {diagnostics['converged']}")
        except Exception as e:
            logger.warning(f"Could not compute diagnostics: {e}")
            diagnostics = {"error": str(e)}
    else:
        logger.info("Single chain - R-hat not computed")
        diagnostics = {"note": "Single chain, R-hat requires >= 2 chains"}

    # Per-chain acceptance rates
    mean_accept = np.mean([r.acceptance_rate for r in chain_results])
    logger.info(f"Mean acceptance rate: {mean_accept:.3f}")

    return BenchmarkResult(
        instance_name=instance_name,
        config=config,
        chain_results=chain_results,
        best_solution=best_offsets,
        best_cost=best_cost,
        pesplib_objective=pesplib_obj,
        diagnostics=diagnostics,
    )


def find_instance_path(instance_name: str) -> Path:
    """Find the path to a PESPlib instance.

    Searches in:
    1. data/pesplib/raw/{instance_name}/
    2. data/pesplib/converted/{instance_name}/
    3. data/pesplib/synthetic/{instance_name}/

    For synthetic instances (mini, triangle), generates them if needed.

    Args:
        instance_name: Name of the instance.

    Returns:
        Path to the instance directory.

    Raises:
        FileNotFoundError: If instance not found.
    """
    base = Path("data/pesplib")

    # Check for synthetic instances first
    if instance_name in ("mini", "triangle"):
        synthetic_instances = ensure_synthetic_instances()
        if instance_name in synthetic_instances:
            return synthetic_instances[instance_name]

    # Search paths
    search_paths = [
        base / "raw" / instance_name,
        base / "converted" / instance_name,
        base / "synthetic" / instance_name,
    ]

    for path in search_paths:
        if path.exists() and (path / "Events.giv").exists():
            return path

    raise FileNotFoundError(
        f"Instance '{instance_name}' not found. "
        f"Searched: {[str(p) for p in search_paths]}"
    )


def main() -> int:
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Run MCMC-PESP benchmark on PESPlib instances"
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="mini",
        help="Instance name (default: mini). Use 'mini' or 'triangle' for synthetic.",
    )
    parser.add_argument(
        "--n_chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=25000,
        help="Steps per chain (default: 25000)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Inverse temperature (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: stdout)",
    )

    args = parser.parse_args()

    # Find instance
    try:
        instance_path = find_instance_path(args.instance)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Create config
    config = BenchmarkConfig(
        n_chains=args.n_chains,
        n_steps_per_chain=args.n_steps,
        beta=args.beta,
    )

    # Run benchmark
    result = run_multi_chain_benchmark(
        instance_path=instance_path,
        config=config,
        base_seed=args.seed,
    )

    # Output results
    result_dict = result.to_dict()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        logger.info(f"Results written to {output_path}")
    else:
        print(json.dumps(result_dict, indent=2))

    # Return success if converged
    converged = result.diagnostics.get("converged", False)
    if isinstance(converged, bool) and converged:
        logger.info("SUCCESS: Chains converged (R-hat < 1.1, ESS > 100)")
        return 0
    else:
        logger.info("Note: Convergence criteria not fully met")
        return 0  # Still return success, just note it


if __name__ == "__main__":
    sys.exit(main())
