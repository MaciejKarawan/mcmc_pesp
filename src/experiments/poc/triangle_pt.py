"""Proof of Concept: Parallel Tempering on Triangle Network.

Demonstrates Parallel Tempering with inertia on a simple triangle network,
comparing performance against a single-chain MCMC run.

Usage:
    python -m src.experiments.poc.triangle_pt
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from src.core.schedule import Schedule
from src.cost.transfer_cost import TransferCost
from src.io.synthetic import TriangleNetwork
from src.mcmc.parallel_tempering import InertiaConfig, ParallelTempering
from src.mcmc.proposals.single_line import SingleLineShift
from src.mcmc.sampler import MCMCSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_chain(
    network,
    cost_fn,
    optimal_cost: float,
    n_steps: int = 10000,
    beta: float = 1.0,
    seed: int = 42,
) -> dict:
    """Run single-chain MCMC for comparison."""
    rng = np.random.default_rng(seed)
    proposal = SingleLineShift(sigma=5.0, rng=rng)

    sampler = MCMCSampler(
        network=network,
        cost_fn=cost_fn,
        proposals=[proposal],
        beta=beta,
        rng=rng,
    )
    sampler.initialize(Schedule.random(network, rng))

    result = sampler.run(n_steps=n_steps, log_every=0)

    return {
        "method": "single_chain",
        "best_cost": result.best_cost,
        "gap": result.best_cost - optimal_cost,
        "acceptance_rate": result.acceptance_rate,
    }


def run_parallel_tempering(
    network,
    cost_fn,
    optimal_cost: float,
    n_steps: int = 10000,
    n_replicas: int = 4,
    seed: int = 42,
) -> dict:
    """Run Parallel Tempering."""
    rng = np.random.default_rng(seed)

    pt = ParallelTempering(
        network=network,
        cost_fn=cost_fn,
        proposals=[SingleLineShift(sigma=5.0)],
        n_replicas=n_replicas,
        beta_min=0.1,
        beta_max=10.0,
        swap_interval=5,
        rng=rng,
    )
    pt.initialize()

    # Each replica does n_steps // n_replicas so total work is comparable
    steps_per_replica = n_steps // n_replicas
    result = pt.run(n_steps=steps_per_replica, log_every=0)

    return {
        "method": f"PT_{n_replicas}_replicas",
        "best_cost": result.best_cost,
        "gap": result.best_cost - optimal_cost,
        "per_replica_acceptance": result.per_replica_acceptance_rates,
        "swap_rates": result.swap_acceptance_rates,
    }


def run_pt_with_inertia(
    network,
    cost_fn,
    optimal_cost: float,
    n_steps: int = 10000,
    n_replicas: int = 4,
    seed: int = 42,
) -> dict:
    """Run Parallel Tempering with inertia."""
    rng = np.random.default_rng(seed)

    inertia = InertiaConfig(modality_inertia={"bus": 5})

    pt = ParallelTempering(
        network=network,
        cost_fn=cost_fn,
        proposals=[SingleLineShift(sigma=5.0)],
        n_replicas=n_replicas,
        beta_min=0.1,
        beta_max=10.0,
        swap_interval=5,
        inertia_config=inertia,
        rng=rng,
    )
    pt.initialize()

    steps_per_replica = n_steps // n_replicas
    result = pt.run(n_steps=steps_per_replica, log_every=0)

    return {
        "method": f"PT_{n_replicas}_inertia",
        "best_cost": result.best_cost,
        "gap": result.best_cost - optimal_cost,
        "per_replica_acceptance": result.per_replica_acceptance_rates,
        "swap_rates": result.swap_acceptance_rates,
    }


def main() -> int:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("MCMC-PESP PoC: Parallel Tempering on Triangle Network")
    logger.info("=" * 60)

    # Create network
    builder = TriangleNetwork()
    network, transfers = builder.create()
    cost_fn = TransferCost(transfers=transfers)

    optimal_offsets = builder.compute_optimal_offset()
    optimal_schedule = Schedule(offsets=optimal_offsets, period=network.period)
    optimal_cost = cost_fn(optimal_schedule, network)

    logger.info(f"Network: {len(network)} lines, {len(transfers)} transfers")
    logger.info(f"Optimal cost: {optimal_cost:.4f}")

    n_steps = 10000
    logger.info(f"\nTotal budget: {n_steps} MH evaluations")
    logger.info("-" * 60)

    # Run experiments
    results = []

    # Single chain
    logger.info("\n1. Single chain MCMC...")
    r1 = run_single_chain(network, cost_fn, optimal_cost, n_steps=n_steps)
    results.append(r1)
    logger.info(f"   Best cost: {r1['best_cost']:.4f}, gap: {r1['gap']:.4f}")

    # PT without inertia
    for m in [2, 4, 8]:
        logger.info(f"\n2. Parallel Tempering (M={m})...")
        r = run_parallel_tempering(
            network, cost_fn, optimal_cost, n_steps=n_steps, n_replicas=m
        )
        results.append(r)
        logger.info(
            f"   Best cost: {r['best_cost']:.4f}, gap: {r['gap']:.4f}, "
            f"swap_rates: {[f'{s:.2f}' for s in r['swap_rates']]}"
        )

    # PT with inertia
    logger.info("\n3. Parallel Tempering with inertia (M=4)...")
    r_inertia = run_pt_with_inertia(
        network, cost_fn, optimal_cost, n_steps=n_steps, n_replicas=4
    )
    results.append(r_inertia)
    logger.info(
        f"   Best cost: {r_inertia['best_cost']:.4f}, gap: {r_inertia['gap']:.4f}, "
        f"swap_rates: {[f'{s:.2f}' for s in r_inertia['swap_rates']]}"
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Method':<30} {'Best Cost':>12} {'Gap':>12}")
    logger.info("-" * 54)
    for r in results:
        logger.info(f"{r['method']:<30} {r['best_cost']:>12.4f} {r['gap']:>12.4f}")

    best = min(results, key=lambda x: x["best_cost"])
    logger.info(f"\nBest method: {best['method']} (cost={best['best_cost']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
