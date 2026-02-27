"""Proof of Concept: MCMC on Triangle Network.

This script demonstrates the MCMC sampler on a simple triangle network,
verifying that it can find the known optimal solution.

Usage:
    python -m src.experiments.poc.triangle
"""

from __future__ import annotations

import logging
import sys

import numpy as np

from src.core.schedule import Schedule
from src.cost.transfer_cost import TransferCost
from src.io.synthetic import TriangleNetwork
from src.mcmc.proposals.single_line import SingleLineShift
from src.mcmc.sampler import MCMCSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_triangle_poc(
    n_steps: int = 10000,
    beta: float = 1.0,
    sigma: float = 5.0,
    seed: int | None = 42,
) -> dict:
    """Run MCMC on the triangle network.

    Args:
        n_steps: Number of MCMC steps.
        beta: Inverse temperature.
        sigma: Standard deviation for line shifts.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with results and diagnostics.
    """
    logger.info("=" * 60)
    logger.info("MCMC-PESP Proof of Concept: Triangle Network")
    logger.info("=" * 60)

    # Create random generator
    rng = np.random.default_rng(seed)

    # Create triangle network
    logger.info("Creating triangle network...")
    builder = TriangleNetwork()
    network, transfers = builder.create()
    optimal_offsets = builder.compute_optimal_offset()

    logger.info(f"Network: {len(network)} lines, {len(transfers)} transfers")
    logger.info(f"Period: {network.period} minutes")
    logger.info(f"Known optimal offsets: {optimal_offsets}")

    # Create cost function
    cost_fn = TransferCost(transfers=transfers)

    # Compute optimal cost (should be 0)
    optimal_schedule = Schedule(offsets=optimal_offsets, period=network.period)
    optimal_cost = cost_fn(optimal_schedule, network)
    logger.info(f"Optimal cost: {optimal_cost:.4f}")

    # Create proposal
    proposal = SingleLineShift(sigma=sigma, rng=rng)

    # Create sampler
    sampler = MCMCSampler(
        network=network,
        cost_fn=cost_fn,
        proposals=[proposal],
        beta=beta,
        rng=rng,
    )

    # Initialize with random schedule
    initial_schedule = Schedule.random(network, rng)
    initial_cost = cost_fn(initial_schedule, network)
    logger.info(f"Initial random schedule: {initial_schedule.offsets}")
    logger.info(f"Initial cost: {initial_cost:.4f}")

    sampler.initialize(initial_schedule)

    # Run MCMC
    logger.info(f"\nRunning MCMC for {n_steps} steps (beta={beta}, sigma={sigma})...")
    result = sampler.run(n_steps=n_steps, log_every=2000)

    # Results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Acceptance rate: {result.acceptance_rate:.3f}")
    logger.info(f"Best cost found: {result.best_cost:.4f}")
    logger.info(f"Best schedule: {result.best_schedule.offsets}")
    logger.info(f"Optimal cost: {optimal_cost:.4f}")
    logger.info(f"Gap to optimum: {result.best_cost - optimal_cost:.4f}")

    # Check convergence
    converged = result.best_cost < 0.01  # Close to zero
    if converged:
        logger.info("\nSUCCESS: MCMC converged to optimal solution!")
    else:
        logger.info(f"\nNote: Best cost {result.best_cost:.4f} > 0 (not at optimum)")

    # Analyze final costs
    summary = result.summary()
    logger.info(f"\nCost statistics over samples:")
    logger.info(f"  Mean: {summary['mean_cost']:.4f}")
    logger.info(f"  Std:  {summary['std_cost']:.4f}")
    logger.info(f"  Min:  {summary['min_cost']:.4f}")
    logger.info(f"  Max:  {summary['max_cost']:.4f}")

    # Transfer details at best solution
    logger.info("\nTransfer details at best solution:")
    details = cost_fn.get_transfer_details(result.best_schedule, network)
    for d in details:
        status = "VIOLATED" if d["violated"] else "OK"
        logger.info(
            f"  {d['id']}: wait={d['waiting_time']:.1f}min, "
            f"cost={d['cost']:.2f}, [{status}]"
        )

    return {
        "converged": converged,
        "best_cost": result.best_cost,
        "optimal_cost": optimal_cost,
        "acceptance_rate": result.acceptance_rate,
        "best_schedule": result.best_schedule.offsets,
        "optimal_schedule": optimal_offsets,
        "summary": summary,
    }


def main() -> int:
    """Main entry point."""
    results = run_triangle_poc(
        n_steps=10000,
        beta=1.0,
        sigma=5.0,
        seed=42,
    )

    # Return 0 if converged, 1 otherwise
    return 0 if results["converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
