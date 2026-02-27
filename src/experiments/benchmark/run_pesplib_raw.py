"""Benchmark runner for raw PESPlib instances.

Usage:
    python -m src.experiments.benchmark.run_pesplib_raw --instance R1L1 --n_steps 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from src.io.pesplib_raw_loader import (
    PESPCostFunction,
    PESPSchedule,
    PESPSingleEventShift,
    load_pesplib_raw,
    run_pesp_mcmc,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Known best solutions for PESPlib Set-01 (from website)
KNOWN_SOLUTIONS = {
    "R1L1": {"objective": 29_894_745, "lower_bound": 20_230_655},
    "R1L2": {"objective": 31_081_380, "lower_bound": 23_093_710},
    "R1L3": {"objective": 35_697_890, "lower_bound": 28_135_880},
    "R1L4": {"objective": 54_104_635, "lower_bound": 47_227_510},
    "R2L1": {"objective": 42_161_450, "lower_bound": 26_316_195},
    "R2L2": {"objective": 43_506_390, "lower_bound": 29_936_230},
    "R2L3": {"objective": 49_961_540, "lower_bound": 39_215_050},
    "R2L4": {"objective": 115_627_335, "lower_bound": 87_549_645},
    "R3L1": {"objective": 59_408_020, "lower_bound": 33_890_315},
    "R3L2": {"objective": 61_346_440, "lower_bound": 42_207_825},
    "R3L3": {"objective": 71_877_820, "lower_bound": 55_082_680},
    "R3L4": {"objective": 172_251_580, "lower_bound": 126_685_195},
    "R4L1": {"objective": 75_588_190, "lower_bound": 52_310_810},
    "R4L2": {"objective": 77_627_590, "lower_bound": 49_862_760},
    "R4L3": {"objective": 97_006_160, "lower_bound": 66_371_585},
    "R4L4": {"objective": 233_312_290, "lower_bound": 145_976_695},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MCMC on PESPlib instances")
    parser.add_argument(
        "--instance",
        type=str,
        default="R1L1",
        help="Instance name (default: R1L1)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50000,
        help="Number of MCMC steps (default: 50000)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.0001,
        help="Inverse temperature (default: 0.0001)",
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

    # Find instance file
    instance_path = Path(f"data/pesplib/raw/{args.instance}.txt")
    if not instance_path.exists():
        logger.error(f"Instance not found: {instance_path}")
        return 1

    logger.info("=" * 60)
    logger.info(f"PESPlib Benchmark: {args.instance}")
    logger.info("=" * 60)

    # Load instance
    instance = load_pesplib_raw(instance_path)
    logger.info(f"Loaded: {instance.n_events} events, {len(instance.activities)} activities")

    # Get known optimum
    known = KNOWN_SOLUTIONS.get(args.instance, {})
    known_obj = known.get("objective")
    lower_bound = known.get("lower_bound")

    if known_obj:
        logger.info(f"Known optimum: {known_obj:,}")
    if lower_bound:
        logger.info(f"Lower bound: {lower_bound:,}")

    # Run MCMC
    logger.info(f"Running MCMC for {args.n_steps} steps, beta={args.beta}")
    best_schedule, best_cost, cost_history = run_pesp_mcmc(
        instance=instance,
        n_steps=args.n_steps,
        beta=args.beta,
        seed=args.seed,
        log_every=5000,
    )

    # Check feasibility
    is_feasible, violations = instance.check_feasibility(best_schedule.times)

    # Compute objective
    objective = instance.compute_objective(best_schedule.times)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best cost (with penalty): {best_cost:,.2f}")
    logger.info(f"Objective (weighted slack): {objective:,.2f}")
    logger.info(f"Feasible: {is_feasible} ({len(violations)} violations)")

    if known_obj:
        gap = (objective / known_obj - 1) * 100
        logger.info(f"Gap to optimum: {gap:.2f}%")

        if gap <= 10:
            logger.info("SUCCESS: Solution within 110% of optimum!")
        else:
            logger.info(f"Note: Gap is {gap:.2f}%, target is <= 10%")

    # Prepare result
    result = {
        "instance": args.instance,
        "n_events": instance.n_events,
        "n_activities": len(instance.activities),
        "n_steps": args.n_steps,
        "beta": args.beta,
        "best_cost": best_cost,
        "objective": objective,
        "feasible": is_feasible,
        "n_violations": len(violations),
        "known_optimum": known_obj,
        "lower_bound": lower_bound,
        "gap_percent": (objective / known_obj - 1) * 100 if known_obj else None,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results written to {output_path}")
    else:
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
