"""A/B benchmark on GTFS weekday profile (Mode B optimization).

A: Baseline schedule reconstructed from GTFS.
B: Best-so-far schedule from MCMC optimization.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.cost.robustness import DelayDistribution, RobustnessCost
from src.cost.total import TotalCost
from src.cost.transfer_cost import TransferCost
from src.io.gtfs_loader import GTFSLoadResult, load_gtfs_weekday
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


@dataclass
class GTFSABConfig:
    gtfs_path: str
    weekday: str = "monday"
    n_chains: int = 4
    n_steps_per_chain: int = 25000
    beta: float = 1.0
    burn_in_fraction: float = 0.1
    thin: int = 10
    seed: int = 42
    min_trips_per_pattern: int = 2
    walk_time: float = 3.0
    max_lines: int | None = None
    max_transfers_per_location: int | None = 400
    transfer_weight_mode: str = "uniform"
    focus_wroclaw_region: bool = False
    focus_nearest_count: int = 4
    focus_buffer_hubs_count: int = 4
    focus_include_keywords: list[str] = field(default_factory=lambda: ["olawa", "brzeg"])
    use_component_move: bool = True
    use_smart_repair: bool = True
    include_robustness: bool = True
    pax_weight: float = 1.0
    rob_weight: float = 0.05
    delay_rate: float = 1.0 / 3.0
    delay_alpha: float = 0.95


@dataclass
class ChainABResult:
    chain_id: int
    best_cost: float
    acceptance_rate: float
    best_offsets: dict[str, int]
    sampled_costs: list[float] = field(default_factory=list)


@dataclass
class GTFSABResult:
    config: GTFSABConfig
    metadata: dict[str, Any]
    baseline_cost: float
    baseline_violations: int
    best_cost: float
    best_violations: int
    best_offsets: dict[str, int]
    diagnostics: dict[str, Any]
    chain_results: list[ChainABResult]
    runtime_seconds: float

    def to_dict(self) -> dict[str, Any]:
        def _json_safe(value: Any) -> Any:
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    return value
            return value

        improvement_abs = self.baseline_cost - self.best_cost
        improvement_pct = 0.0
        if self.baseline_cost != 0:
            improvement_pct = 100.0 * improvement_abs / self.baseline_cost

        return {
            "gtfs_path": self.config.gtfs_path,
            "weekday": self.config.weekday,
            "n_lines": self.metadata.get("n_lines"),
            "n_transfers": self.metadata.get("n_transfers"),
            "active_service_ids": self.metadata.get("active_service_ids", []),
            "transfer_weight_mode": self.metadata.get("transfer_weight_mode"),
            "focus_wroclaw_region": self.metadata.get("focus_wroclaw_region"),
            "focus_locations": self.metadata.get("focus_locations", []),
            "focus_wroclaw_names": self.metadata.get("focus_wroclaw_names", []),
            "focus_nearest_names": self.metadata.get("focus_nearest_names", []),
            "focus_keyword_names": self.metadata.get("focus_keyword_names", []),
            "focus_buffer_names": self.metadata.get("focus_buffer_names", []),
            "baseline": {
                "cost": self.baseline_cost,
                "violations": self.baseline_violations,
            },
            "optimized": {
                "best_cost": self.best_cost,
                "violations": self.best_violations,
                "best_offsets": self.best_offsets,
            },
            "improvement": {
                "absolute": improvement_abs,
                "percent": improvement_pct,
            },
            "diagnostics": {
                key: _json_safe(value) for key, value in self.diagnostics.items()
            },
            "chains": [
                {
                    "chain_id": chain.chain_id,
                    "best_cost": chain.best_cost,
                    "acceptance_rate": chain.acceptance_rate,
                    "n_samples": len(chain.sampled_costs),
                }
                for chain in self.chain_results
            ],
            "runtime_seconds": self.runtime_seconds,
        }


def _build_cost_fn(
    load_result: GTFSLoadResult,
    config: GTFSABConfig,
) -> tuple[TransferCost, TransferCost | TotalCost]:
    transfer_cost = TransferCost(transfers=load_result.transfers)
    if not config.include_robustness:
        return transfer_cost, transfer_cost

    rob_cost = RobustnessCost(
        transfers=load_result.transfers,
        delay_distribution=DelayDistribution(
            rate=config.delay_rate,
            alpha=config.delay_alpha,
        ),
    )
    total = TotalCost(
        transfer_cost=transfer_cost,
        robustness_cost=rob_cost,
        pax_weight=config.pax_weight,
        rob_weight=config.rob_weight,
    )
    return transfer_cost, total


def _build_proposals(
    load_result: GTFSLoadResult,
    config: GTFSABConfig,
    rng: np.random.Generator,
) -> tuple[list[Any], list[float]]:
    proposals: list[Any] = []
    weights: list[float] = []

    proposals.append(SingleLineShift(sigma=5.0, rng=rng))
    weights.append(0.4)

    proposals.append(UniformLineShift(max_shift=10, rng=rng))
    weights.append(0.1)

    if config.use_component_move and len(load_result.network) > 1:
        proposals.append(create_component_move(load_result.transfers, sigma=5.0, rng=rng))
        weights.append(0.3)

    if config.use_smart_repair and len(load_result.transfers) > 0:
        proposals.append(create_smart_repair(load_result.transfers, sigma=2.0, rng=rng))
        weights.append(0.2)

    return proposals, weights


def run_gtfs_weekday_ab(config: GTFSABConfig) -> GTFSABResult:
    start_time = time.time()
    logger.info("Loading GTFS weekday profile")
    load_result = load_gtfs_weekday(
        feed_path=config.gtfs_path,
        weekday=config.weekday,
        min_trips_per_pattern=config.min_trips_per_pattern,
        walk_time=config.walk_time,
        max_lines=config.max_lines,
        max_transfers_per_location=config.max_transfers_per_location,
        transfer_weight_mode=config.transfer_weight_mode,
        focus_wroclaw_region=config.focus_wroclaw_region,
        focus_nearest_count=config.focus_nearest_count,
        focus_buffer_hubs_count=config.focus_buffer_hubs_count,
        focus_include_keywords=config.focus_include_keywords,
    )

    transfer_cost, full_cost = _build_cost_fn(load_result, config)
    baseline_cost = full_cost(load_result.baseline_schedule, load_result.network)
    baseline_violations = transfer_cost.count_violations(
        load_result.baseline_schedule,
        load_result.network,
    )

    logger.info(
        "GTFS slice: lines=%d transfers=%d",
        len(load_result.network),
        len(load_result.transfers),
    )
    logger.info(
        "Baseline: cost=%.4f violations=%d",
        baseline_cost,
        baseline_violations,
    )

    burn_in = int(config.n_steps_per_chain * config.burn_in_fraction)
    chain_results: list[ChainABResult] = []
    all_chains_costs: list[list[float]] = []

    for chain_id in range(config.n_chains):
        chain_seed = config.seed + chain_id
        rng = np.random.default_rng(chain_seed)
        proposals, proposal_weights = _build_proposals(load_result, config, rng)

        sampler = MCMCSampler(
            network=load_result.network,
            cost_fn=full_cost,
            proposals=proposals,
            proposal_weights=proposal_weights,
            beta=config.beta,
            mode="optimization",
            rng=rng,
        )
        sampler.initialize(load_result.baseline_schedule)
        run_result = sampler.run(
            n_steps=config.n_steps_per_chain,
            thin=config.thin,
            burn_in=burn_in,
            log_every=0,
        )

        all_chains_costs.append(run_result.costs)
        chain_results.append(
            ChainABResult(
                chain_id=chain_id,
                best_cost=run_result.best_cost,
                acceptance_rate=run_result.acceptance_rate,
                best_offsets=run_result.best_schedule.offsets.copy(),
                sampled_costs=run_result.costs,
            )
        )

    best_chain = min(chain_results, key=lambda chain: chain.best_cost)
    best_schedule = load_result.baseline_schedule.copy()
    best_schedule.offsets = dict(best_chain.best_offsets)
    best_violations = transfer_cost.count_violations(best_schedule, load_result.network)

    diagnostics: dict[str, Any] = {}
    if config.n_chains >= 2:
        try:
            summary = MCMCDiagnostics(chains=all_chains_costs).summary()
            diagnostics = {
                "r_hat": summary.get("r_hat"),
                "ess_total": summary.get("ess_total"),
                "converged": summary.get("converged"),
            }
        except Exception as exc:
            diagnostics = {"error": str(exc)}
    else:
        diagnostics = {"note": "single chain"}

    elapsed = time.time() - start_time
    logger.info(
        "Optimized: best_cost=%.4f violations=%d runtime=%.1fs",
        best_chain.best_cost,
        best_violations,
        elapsed,
    )

    return GTFSABResult(
        config=config,
        metadata=load_result.metadata,
        baseline_cost=baseline_cost,
        baseline_violations=baseline_violations,
        best_cost=best_chain.best_cost,
        best_violations=best_violations,
        best_offsets=best_chain.best_offsets,
        diagnostics=diagnostics,
        chain_results=chain_results,
        runtime_seconds=elapsed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GTFS weekday A/B benchmark in optimization mode",
    )
    parser.add_argument("--gtfs_path", required=True, type=str, help="GTFS folder or .zip")
    parser.add_argument("--weekday", type=str, default="monday", help="Weekday profile")
    parser.add_argument("--n_chains", type=int, default=4, help="Number of chains")
    parser.add_argument("--n_steps", type=int, default=25000, help="Steps per chain")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--thin", type=int, default=10, help="Thinning interval")
    parser.add_argument(
        "--burn_in_fraction",
        type=float,
        default=0.1,
        help="Burn-in fraction in [0, 1)",
    )
    parser.add_argument(
        "--min_trips_per_pattern",
        type=int,
        default=2,
        help="Minimum number of trips per line pattern",
    )
    parser.add_argument("--walk_time", type=float, default=3.0, help="Walk time in minutes")
    parser.add_argument("--max_lines", type=int, default=None, help="Optional line cap")
    parser.add_argument(
        "--max_transfers_per_location",
        type=int,
        default=400,
        help="Cap directed transfer pairs per location",
    )
    parser.add_argument(
        "--transfer_weight_mode",
        type=str,
        choices=["uniform", "frequency"],
        default="uniform",
        help="Transfer weight mode",
    )
    parser.add_argument(
        "--focus_wroclaw_region",
        action="store_true",
        help="Restrict transfer set to Wroclaw-centric focus region",
    )
    parser.add_argument(
        "--focus_nearest_count",
        type=int,
        default=4,
        help="Nearest stations around Wroclaw to include",
    )
    parser.add_argument(
        "--focus_buffer_hubs_count",
        type=int,
        default=4,
        help="Additional high-volume buffer hubs linked to Wroclaw",
    )
    parser.add_argument(
        "--focus_keywords",
        type=str,
        default="olawa,brzeg",
        help="Comma-separated location keywords to include in focus region",
    )
    parser.add_argument(
        "--disable_component_move",
        action="store_true",
        help="Disable component move proposal",
    )
    parser.add_argument(
        "--disable_smart_repair",
        action="store_true",
        help="Disable smart repair proposal",
    )
    parser.add_argument(
        "--disable_robustness",
        action="store_true",
        help="Use transfer cost only (no robustness component)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")

    args = parser.parse_args()
    focus_keywords = [
        token.strip() for token in args.focus_keywords.split(",") if token.strip()
    ]

    config = GTFSABConfig(
        gtfs_path=args.gtfs_path,
        weekday=args.weekday,
        n_chains=args.n_chains,
        n_steps_per_chain=args.n_steps,
        beta=args.beta,
        burn_in_fraction=args.burn_in_fraction,
        thin=args.thin,
        seed=args.seed,
        min_trips_per_pattern=args.min_trips_per_pattern,
        walk_time=args.walk_time,
        max_lines=args.max_lines,
        max_transfers_per_location=args.max_transfers_per_location,
        transfer_weight_mode=args.transfer_weight_mode,
        focus_wroclaw_region=args.focus_wroclaw_region,
        focus_nearest_count=args.focus_nearest_count,
        focus_buffer_hubs_count=args.focus_buffer_hubs_count,
        focus_include_keywords=focus_keywords,
        use_component_move=not args.disable_component_move,
        use_smart_repair=not args.disable_smart_repair,
        include_robustness=not args.disable_robustness,
    )

    result = run_gtfs_weekday_ab(config)
    payload = result.to_dict()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Wrote output to %s", output_path)
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
