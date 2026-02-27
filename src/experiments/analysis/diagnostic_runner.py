"""Diagnostic experiment runner for MCMC-PESP.

Provides functions to run systematic diagnostic experiments on synthetic
scenarios, measuring convergence, proposal effectiveness, PT performance,
and scaling behavior. Results are returned as structured dataclasses
suitable for analysis in Jupyter notebooks.

Usage:
    from src.experiments.analysis.diagnostic_runner import (
        sample_cost_landscape,
        run_convergence_diagnostic,
        run_proposal_comparison,
        run_pt_diagnostic,
        run_scaling_analysis,
        run_optimum_gap_tracking,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator

from src.core.network import Network
from src.core.schedule import Schedule
from src.core.transfers import TransferSet
from src.cost.operational import (
    FleetCost,
    OperationalCost,
    TurnaroundCost,
)
from src.cost.robustness import DelayDistribution, RobustnessCost
from src.cost.total import TotalCost
from src.cost.transfer_cost import TransferCost
from src.io.synthetic_scenarios import SyntheticScenario, SyntheticScenarioFactory
from src.mcmc.parallel_tempering import InertiaConfig, ParallelTempering
from src.mcmc.proposals.component import create_component_move
from src.mcmc.proposals.single_line import SingleLineShift, UniformLineShift
from src.mcmc.proposals.smart_repair import create_smart_repair
from src.mcmc.sampler import MCMCSampler
from src.metrics.convergence import MCMCDiagnostics, compute_ess, compute_r_hat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CostLandscapeResult:
    """Result of cost landscape sampling."""

    scenario_name: str
    scenario_type: str
    difficulty: str
    n_lines: int
    n_transfers: int
    costs: list[float]
    mean: float
    std: float
    median: float
    q05: float
    q25: float
    q75: float
    q95: float
    min_cost: float
    max_cost: float


@dataclass
class ConvergenceCheckpoint:
    """Snapshot of convergence metrics at a given step count."""

    step: int
    r_hat: float | None
    ess_total: float
    ess_per_chain: list[float]
    best_cost: float
    mean_cost: float
    acceptance_rate: float


@dataclass
class ConvergenceDiagnosticResult:
    """Result of convergence diagnostic experiment."""

    scenario_name: str
    n_chains: int
    total_steps: int
    checkpoints: list[ConvergenceCheckpoint]
    final_r_hat: float | None
    final_ess_total: float
    steps_to_convergence: int | None
    converged: bool
    wall_time_s: float


@dataclass
class ProposalComparisonEntry:
    """Per-proposal diagnostic stats."""

    proposal_name: str
    acceptance_rate: float
    best_cost: float
    mean_cost: float
    ess_total: float
    ess_per_step: float
    time_per_step_ms: float
    n_steps: int


@dataclass
class ProposalComparisonResult:
    """Result of proposal comparison experiment."""

    scenario_name: str
    entries: list[ProposalComparisonEntry]


@dataclass
class PTComparisonEntry:
    """Single PT configuration result."""

    n_replicas: int
    best_cost: float
    cold_chain_mean_cost: float
    swap_rates: list[float]
    per_replica_acceptance: list[float]
    wall_time_s: float
    use_inertia: bool


@dataclass
class PTComparisonResult:
    """Result of PT comparison experiment."""

    scenario_name: str
    entries: list[PTComparisonEntry]


@dataclass
class ScalingPoint:
    """Single data point in scaling analysis."""

    scenario_name: str
    n_lines: int
    n_transfers: int
    time_per_step_ms: float
    steps_measured: int


@dataclass
class ScalingResult:
    """Result of scaling analysis."""

    points: list[ScalingPoint]


@dataclass
class GapTrackingPoint:
    """Gap to optimum at a given step."""

    step: int
    best_cost: float
    optimal_cost: float
    gap: float


@dataclass
class GapTrackingResult:
    """Result of optimum gap tracking."""

    scenario_name: str
    optimal_cost: float
    points: list[GapTrackingPoint]
    final_gap: float
    wall_time_s: float


# ---------------------------------------------------------------------------
# Helper: build cost function and proposals for a scenario
# ---------------------------------------------------------------------------

def build_cost_fn(
    scenario: SyntheticScenario,
    pax_weight: float = 1.0,
    ops_weight: float = 0.1,
    rob_weight: float = 0.05,
) -> TransferCost | TotalCost:
    """Build cost function appropriate for the scenario type."""
    transfer_cost = TransferCost(transfers=scenario.transfers)

    if scenario.turnarounds is not None and len(scenario.turnarounds) > 0:
        turnaround_cost = TurnaroundCost(turnarounds=scenario.turnarounds)
        ops_cost = OperationalCost(
            turnaround_cost=turnaround_cost,
            fleet_cost=FleetCost(available_fleet=100, lambda_fleet=0.1),
        )
    else:
        ops_cost = None

    delay_dist = DelayDistribution(rate=1.0 / 3.0, alpha=0.95)
    rob_cost = RobustnessCost(
        transfers=scenario.transfers,
        delay_distribution=delay_dist,
    )

    return TotalCost(
        transfer_cost=transfer_cost,
        operational_cost=ops_cost,
        robustness_cost=rob_cost,
        pax_weight=pax_weight,
        ops_weight=ops_weight,
        rob_weight=rob_weight,
    )


def build_proposals(
    scenario: SyntheticScenario,
    rng: Generator,
    include_component: bool = True,
    include_smart_repair: bool = True,
) -> tuple[list, list[float]]:
    """Build proposal list and weights for a scenario."""
    proposals = []
    weights = []

    proposals.append(SingleLineShift(sigma=5.0, rng=rng))
    weights.append(0.4)

    proposals.append(UniformLineShift(max_shift=10, rng=rng))
    weights.append(0.1)

    if include_component and len(scenario.transfers) > 0:
        component = create_component_move(
            scenario.transfers, sigma=5.0, rng=rng,
        )
        proposals.append(component)
        weights.append(0.3)

    if include_smart_repair and len(scenario.transfers) > 0:
        smart_repair = create_smart_repair(
            scenario.transfers, sigma=2.0, rng=rng,
        )
        proposals.append(smart_repair)
        weights.append(0.2)

    return proposals, weights


# ---------------------------------------------------------------------------
# 1. Cost Landscape Sampling
# ---------------------------------------------------------------------------

def sample_cost_landscape(
    scenario: SyntheticScenario,
    n_samples: int = 1000,
    seed: int = 42,
) -> CostLandscapeResult:
    """Sample random schedules and compute costs to characterize the landscape.

    Args:
        scenario: Synthetic scenario to analyze.
        n_samples: Number of random schedules to sample.
        seed: Random seed.

    Returns:
        CostLandscapeResult with cost distribution statistics.
    """
    rng = np.random.default_rng(seed)
    cost_fn = build_cost_fn(scenario)

    costs = []
    for _ in range(n_samples):
        schedule = Schedule.random(scenario.network, rng)
        cost = cost_fn(schedule, scenario.network)
        costs.append(cost)

    costs_arr = np.array(costs)

    return CostLandscapeResult(
        scenario_name=scenario.metadata.name,
        scenario_type=scenario.metadata.scenario_type,
        difficulty=scenario.metadata.difficulty,
        n_lines=scenario.metadata.n_lines,
        n_transfers=scenario.metadata.n_transfers,
        costs=costs,
        mean=float(np.mean(costs_arr)),
        std=float(np.std(costs_arr)),
        median=float(np.median(costs_arr)),
        q05=float(np.percentile(costs_arr, 5)),
        q25=float(np.percentile(costs_arr, 25)),
        q75=float(np.percentile(costs_arr, 75)),
        q95=float(np.percentile(costs_arr, 95)),
        min_cost=float(np.min(costs_arr)),
        max_cost=float(np.max(costs_arr)),
    )


# ---------------------------------------------------------------------------
# 2. Convergence Diagnostic
# ---------------------------------------------------------------------------

def run_convergence_diagnostic(
    scenario: SyntheticScenario,
    n_chains: int = 4,
    n_steps: int = 20000,
    checkpoint_every: int = 2000,
    beta: float = 1.0,
    seed: int = 42,
) -> ConvergenceDiagnosticResult:
    """Run multi-chain MCMC and track convergence over time.

    Args:
        scenario: Scenario to test.
        n_chains: Number of independent chains.
        n_steps: Total steps per chain.
        checkpoint_every: Steps between convergence checkpoints.
        beta: Inverse temperature.
        seed: Base random seed.

    Returns:
        ConvergenceDiagnosticResult with R-hat/ESS checkpoints.
    """
    t0 = time.perf_counter()

    cost_fn = build_cost_fn(scenario)

    # Create and run chains, collecting costs at each step
    all_costs: list[list[float]] = [[] for _ in range(n_chains)]
    samplers: list[MCMCSampler] = []

    for c in range(n_chains):
        rng = np.random.default_rng(seed + c)
        proposals, weights = build_proposals(scenario, rng)
        sampler = MCMCSampler(
            network=scenario.network,
            cost_fn=cost_fn,
            proposals=proposals,
            proposal_weights=weights,
            beta=beta,
            mode="optimization",
            rng=rng,
        )
        sampler.initialize(Schedule.random(scenario.network, rng))
        samplers.append(sampler)

    # Run step by step, collecting checkpoints
    checkpoints: list[ConvergenceCheckpoint] = []
    steps_to_convergence: int | None = None

    for step in range(1, n_steps + 1):
        for c, sampler in enumerate(samplers):
            _, _ = sampler.step()
            all_costs[c].append(sampler.current_cost)

        if step % checkpoint_every == 0 and step >= checkpoint_every * 2:
            # Compute diagnostics on current data
            # Use second half for better estimates
            half = step // 2
            chains_for_diag = [costs[half:] for costs in all_costs]

            if len(chains_for_diag[0]) >= 4:
                try:
                    r_hat = float(compute_r_hat(
                        [np.array(c) for c in chains_for_diag]
                    )) if n_chains >= 2 else None
                except Exception:
                    r_hat = None

                ess_per = []
                for c_data in chains_for_diag:
                    try:
                        ess_per.append(float(compute_ess(np.array(c_data))))
                    except Exception:
                        ess_per.append(0.0)

                ess_total = sum(ess_per)
                best_cost = min(s.best_cost for s in samplers)
                mean_cost = float(np.mean(
                    [costs[-1] for costs in all_costs]
                ))
                accept_rate = float(np.mean(
                    [s.acceptance_rate for s in samplers]
                ))

                cp = ConvergenceCheckpoint(
                    step=step,
                    r_hat=r_hat,
                    ess_total=ess_total,
                    ess_per_chain=ess_per,
                    best_cost=best_cost,
                    mean_cost=mean_cost,
                    acceptance_rate=accept_rate,
                )
                checkpoints.append(cp)

                if (steps_to_convergence is None
                        and r_hat is not None
                        and r_hat < 1.1
                        and min(ess_per) > 50):
                    steps_to_convergence = step

    wall_time = time.perf_counter() - t0

    final_r_hat = checkpoints[-1].r_hat if checkpoints else None
    final_ess = checkpoints[-1].ess_total if checkpoints else 0.0
    converged = (
        final_r_hat is not None
        and final_r_hat < 1.1
        and checkpoints[-1].ess_per_chain
        and min(checkpoints[-1].ess_per_chain) > 50
    ) if checkpoints else False

    return ConvergenceDiagnosticResult(
        scenario_name=scenario.metadata.name,
        n_chains=n_chains,
        total_steps=n_steps,
        checkpoints=checkpoints,
        final_r_hat=final_r_hat,
        final_ess_total=final_ess,
        steps_to_convergence=steps_to_convergence,
        converged=converged,
        wall_time_s=wall_time,
    )


# ---------------------------------------------------------------------------
# 3. Proposal Comparison
# ---------------------------------------------------------------------------

def run_proposal_comparison(
    scenario: SyntheticScenario,
    n_steps: int = 10000,
    n_chains: int = 2,
    beta: float = 1.0,
    seed: int = 42,
) -> ProposalComparisonResult:
    """Test each proposal type separately and measure effectiveness.

    Args:
        scenario: Scenario to test on.
        n_steps: Steps per chain.
        n_chains: Number of chains for ESS computation.
        beta: Inverse temperature.
        seed: Random seed.

    Returns:
        ProposalComparisonResult with per-proposal stats.
    """
    cost_fn = build_cost_fn(scenario)
    entries: list[ProposalComparisonEntry] = []

    # Define proposals to test individually
    proposal_configs = [
        ("single_line_shift", lambda rng: ([SingleLineShift(sigma=5.0, rng=rng)], [1.0])),
        ("uniform_line_shift", lambda rng: ([UniformLineShift(max_shift=10, rng=rng)], [1.0])),
    ]

    if len(scenario.transfers) > 0:
        proposal_configs.append(
            ("component_move", lambda rng: (
                [create_component_move(scenario.transfers, sigma=5.0, rng=rng)],
                [1.0],
            ))
        )
        proposal_configs.append(
            ("smart_repair", lambda rng: (
                [create_smart_repair(scenario.transfers, sigma=2.0, rng=rng)],
                [1.0],
            ))
        )

    # Add the standard mix
    def make_mix(rng: Generator):
        return build_proposals(scenario, rng)

    proposal_configs.append(("full_mix", make_mix))

    for name, make_proposals in proposal_configs:
        chain_costs: list[list[float]] = []
        total_accept = 0
        total_steps_done = 0
        best_cost_overall = float("inf")
        mean_cost_last = 0.0

        t0 = time.perf_counter()

        for c in range(n_chains):
            rng = np.random.default_rng(seed + c * 100)
            proposals, weights = make_proposals(rng)
            sampler = MCMCSampler(
                network=scenario.network,
                cost_fn=cost_fn,
                proposals=proposals,
                proposal_weights=weights,
                beta=beta,
                mode="optimization",
                rng=rng,
            )
            sampler.initialize(Schedule.random(scenario.network, rng))
            result = sampler.run(n_steps=n_steps, log_every=0)

            chain_costs.append(result.costs if result.costs else [result.best_cost])
            total_accept += result.acceptance_rate
            total_steps_done += n_steps
            best_cost_overall = min(best_cost_overall, result.best_cost)
            mean_cost_last += np.mean(result.costs) if result.costs else result.best_cost

        wall_time = time.perf_counter() - t0
        time_per_step = (wall_time / total_steps_done) * 1000  # ms

        # ESS
        ess_total = 0.0
        for cc in chain_costs:
            if len(cc) >= 4:
                try:
                    ess_total += compute_ess(np.array(cc))
                except Exception:
                    pass

        entries.append(ProposalComparisonEntry(
            proposal_name=name,
            acceptance_rate=total_accept / n_chains,
            best_cost=best_cost_overall,
            mean_cost=mean_cost_last / n_chains,
            ess_total=ess_total,
            ess_per_step=ess_total / total_steps_done if total_steps_done > 0 else 0,
            time_per_step_ms=time_per_step,
            n_steps=n_steps,
        ))

    return ProposalComparisonResult(
        scenario_name=scenario.metadata.name,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# 4. Parallel Tempering Comparison
# ---------------------------------------------------------------------------

def run_pt_diagnostic(
    scenario: SyntheticScenario,
    replica_counts: list[int] | None = None,
    n_steps: int = 10000,
    beta_min: float = 0.1,
    beta_max: float = 10.0,
    test_inertia: bool = True,
    seed: int = 42,
) -> PTComparisonResult:
    """Compare PT performance across replica counts and inertia settings.

    Args:
        scenario: Scenario to test on.
        replica_counts: List of replica counts to test.
        n_steps: Total step budget (divided among replicas).
        beta_min: Minimum inverse temperature.
        beta_max: Maximum inverse temperature.
        test_inertia: Also test with inertia enabled.
        seed: Random seed.

    Returns:
        PTComparisonResult comparing configurations.
    """
    if replica_counts is None:
        replica_counts = [1, 4, 8]

    cost_fn = build_cost_fn(scenario)
    entries: list[PTComparisonEntry] = []

    configs_to_test: list[tuple[int, bool]] = []
    for m in replica_counts:
        configs_to_test.append((m, False))
    if test_inertia:
        for m in replica_counts:
            if m > 1:
                configs_to_test.append((m, True))

    for n_rep, use_inertia in configs_to_test:
        rng = np.random.default_rng(seed)

        if n_rep == 1:
            # Single chain baseline
            t0 = time.perf_counter()
            proposals, weights = build_proposals(scenario, rng)
            sampler = MCMCSampler(
                network=scenario.network,
                cost_fn=cost_fn,
                proposals=proposals,
                proposal_weights=weights,
                beta=beta_max,
                mode="optimization",
                rng=rng,
            )
            sampler.initialize(Schedule.random(scenario.network, rng))
            result = sampler.run(n_steps=n_steps, log_every=0)
            wall_time = time.perf_counter() - t0

            entries.append(PTComparisonEntry(
                n_replicas=1,
                best_cost=result.best_cost,
                cold_chain_mean_cost=float(np.mean(result.costs)) if result.costs else result.best_cost,
                swap_rates=[],
                per_replica_acceptance=[result.acceptance_rate],
                wall_time_s=wall_time,
                use_inertia=False,
            ))
            continue

        # PT with n_rep replicas
        steps_per_replica = n_steps // n_rep

        proposals, weights = build_proposals(scenario, rng)

        inertia_config = None
        if use_inertia:
            inertia_config = InertiaConfig(
                modality_inertia={"IC": 20.0, "Regional": 5.0, "bus": 1.0},
            )

        t0 = time.perf_counter()
        pt = ParallelTempering(
            network=scenario.network,
            cost_fn=cost_fn,
            proposals=proposals,
            proposal_weights=weights,
            n_replicas=n_rep,
            beta_min=beta_min,
            beta_max=beta_max,
            swap_interval=10,
            mode="optimization",
            inertia_config=inertia_config,
            rng=rng,
        )
        pt.initialize()

        pt_result = pt.run(
            n_steps=steps_per_replica,
            log_every=0,
        )
        wall_time = time.perf_counter() - t0

        entries.append(PTComparisonEntry(
            n_replicas=n_rep,
            best_cost=pt_result.best_cost,
            cold_chain_mean_cost=float(np.mean(pt_result.cold_chain_costs)) if pt_result.cold_chain_costs else pt_result.best_cost,
            swap_rates=list(pt_result.swap_acceptance_rates),
            per_replica_acceptance=list(pt_result.per_replica_acceptance_rates),
            wall_time_s=wall_time,
            use_inertia=use_inertia,
        ))

    return PTComparisonResult(
        scenario_name=scenario.metadata.name,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# 5. Scaling Analysis
# ---------------------------------------------------------------------------

def run_scaling_analysis(
    scenario_specs: list[dict[str, Any]] | None = None,
    n_steps: int = 2000,
    beta: float = 1.0,
    seed: int = 42,
) -> ScalingResult:
    """Measure time per MCMC step across different network sizes.

    Args:
        scenario_specs: List of {"type": ..., "difficulty": ...} dicts.
            Defaults to a range of scenarios.
        n_steps: Steps to run for timing.
        beta: Inverse temperature.
        seed: Random seed.

    Returns:
        ScalingResult with timing data per scenario.
    """
    if scenario_specs is None:
        scenario_specs = [
            {"type": "hub", "difficulty": "easy"},
            {"type": "hub", "difficulty": "medium"},
            {"type": "hub", "difficulty": "hard"},
            {"type": "chain", "difficulty": "easy"},
            {"type": "chain", "difficulty": "hard"},
            {"type": "multi_hub_mesh", "difficulty": "easy"},
            {"type": "multi_hub_mesh", "difficulty": "medium"},
            {"type": "random_network", "difficulty": "easy", "seed": 42},
            {"type": "random_network", "difficulty": "medium", "seed": 42},
            {"type": "random_network", "difficulty": "hard", "seed": 42},
        ]

    factory = SyntheticScenarioFactory()
    points: list[ScalingPoint] = []

    for spec in scenario_specs:
        stype = spec.pop("type")
        diff = spec.pop("difficulty", "easy")
        scenario = factory.create(stype, difficulty=diff, **spec)
        # Restore for potential re-use
        spec["type"] = stype
        spec["difficulty"] = diff

        cost_fn = build_cost_fn(scenario)
        rng = np.random.default_rng(seed)
        proposals, weights = build_proposals(scenario, rng)

        sampler = MCMCSampler(
            network=scenario.network,
            cost_fn=cost_fn,
            proposals=proposals,
            proposal_weights=weights,
            beta=beta,
            mode="optimization",
            rng=rng,
        )
        sampler.initialize(Schedule.random(scenario.network, rng))

        # Warm up
        for _ in range(100):
            sampler.step()

        # Time
        t0 = time.perf_counter()
        for _ in range(n_steps):
            sampler.step()
        elapsed = time.perf_counter() - t0

        points.append(ScalingPoint(
            scenario_name=scenario.metadata.name,
            n_lines=scenario.metadata.n_lines,
            n_transfers=scenario.metadata.n_transfers,
            time_per_step_ms=(elapsed / n_steps) * 1000,
            steps_measured=n_steps,
        ))

    return ScalingResult(points=points)


# ---------------------------------------------------------------------------
# 6. Optimum Gap Tracking
# ---------------------------------------------------------------------------

def run_optimum_gap_tracking(
    scenario: SyntheticScenario,
    n_steps: int = 20000,
    checkpoint_every: int = 500,
    beta: float = 2.0,
    seed: int = 42,
) -> GapTrackingResult:
    """Track how MCMC approaches the known optimum over time.

    Only meaningful for scenarios with known optimal solutions (hub, chain).

    Args:
        scenario: Scenario with known optimum.
        n_steps: Total MCMC steps.
        checkpoint_every: Steps between gap measurements.
        beta: Inverse temperature.
        seed: Random seed.

    Returns:
        GapTrackingResult with gap trajectory.

    Raises:
        ValueError: If scenario has no known optimal solution.
    """
    if scenario.metadata.optimal_solution is None:
        raise ValueError(
            f"Scenario '{scenario.metadata.name}' has no known optimal solution"
        )

    t0 = time.perf_counter()

    # Compute optimal cost
    cost_fn = build_cost_fn(scenario)
    optimal_schedule = Schedule(
        offsets=scenario.metadata.optimal_solution,
        period=scenario.network.period,
    )
    optimal_cost = cost_fn(optimal_schedule, scenario.network)

    # Run MCMC
    rng = np.random.default_rng(seed)
    proposals, weights = build_proposals(scenario, rng)

    sampler = MCMCSampler(
        network=scenario.network,
        cost_fn=cost_fn,
        proposals=proposals,
        proposal_weights=weights,
        beta=beta,
        mode="optimization",
        rng=rng,
    )
    sampler.initialize(Schedule.random(scenario.network, rng))

    points: list[GapTrackingPoint] = []

    for step in range(1, n_steps + 1):
        sampler.step()

        if step % checkpoint_every == 0:
            best = sampler.best_cost
            if optimal_cost > 1e-9:
                gap = (best - optimal_cost) / optimal_cost
            else:
                gap = best - optimal_cost

            points.append(GapTrackingPoint(
                step=step,
                best_cost=best,
                optimal_cost=optimal_cost,
                gap=gap,
            ))

    wall_time = time.perf_counter() - t0

    return GapTrackingResult(
        scenario_name=scenario.metadata.name,
        optimal_cost=optimal_cost,
        points=points,
        final_gap=points[-1].gap if points else float("inf"),
        wall_time_s=wall_time,
    )


# ---------------------------------------------------------------------------
# 7. Cost Calibration
# ---------------------------------------------------------------------------

@dataclass
class BarrierSensitivityPoint:
    """Single measurement in barrier sensitivity analysis."""

    M_barrier: float
    best_cost: float
    mean_cost: float
    violation_rate: float  # Fraction of steps where constraints violated
    acceptance_rate: float


@dataclass
class BarrierSensitivityResult:
    """Results from barrier sensitivity analysis."""

    scenario_name: str
    points: list[BarrierSensitivityPoint]


@dataclass
class LambdaSweepPoint:
    """Single measurement in lambda weights sweep."""

    lambda_pax: float
    lambda_ops: float
    lambda_rob: float
    best_cost: float
    cost_pax: float  # PAX component only
    cost_ops: float  # OPS component only
    cost_rob: float  # ROB component only
    acceptance_rate: float


@dataclass
class LambdaSweepResult:
    """Results from lambda weights sweep."""

    scenario_name: str
    points: list[LambdaSweepPoint]


@dataclass
class ParetoFrontResult:
    """Pareto front for multi-objective optimization."""

    scenario_name: str
    points: list[tuple[float, float]]  # (objective1, objective2)
    labels: tuple[str, str]  # Names of objectives
    dominated_points: list[tuple[float, float]]  # Non-Pareto points


def run_barrier_sensitivity(
    scenario: SyntheticScenario,
    M_values: list[float] | None = None,
    n_steps: int = 5000,
    beta: float = 1.0,
    seed: int = 42,
) -> BarrierSensitivityResult:
    """Test sensitivity to barrier parameter M.

    Args:
        scenario: Test scenario.
        M_values: Barrier values to test. Defaults to [1e3, 1e4, 1e5, 1e6, 1e7].
        n_steps: MCMC steps per M value.
        beta: Inverse temperature.
        seed: Random seed.

    Returns:
        BarrierSensitivityResult with measurements per M.
    """
    if M_values is None:
        M_values = [1e3, 1e4, 1e5, 1e6, 1e7]

    points: list[BarrierSensitivityPoint] = []

    for M in M_values:
        # Create new TransferSet with updated M in params
        from src.core.transfers import Transfer, TransferParams, TransferSet
        from dataclasses import replace

        modified_transfers = {}
        for tid, t in scenario.transfers.transfers.items():
            # Create new params with updated M
            new_params = replace(t.params, M=M)
            # Create new transfer with new params
            new_transfer = replace(t, params=new_params)
            modified_transfers[tid] = new_transfer

        modified_transfer_set = TransferSet(transfers=modified_transfers)

        # Build cost function with modified transfers
        transfer_cost = TransferCost(transfers=modified_transfer_set)

        rng = np.random.default_rng(seed)
        proposals, weights = build_proposals(scenario, rng)

        sampler = MCMCSampler(
            network=scenario.network,
            cost_fn=transfer_cost,
            proposals=proposals,
            proposal_weights=weights,
            beta=beta,
            mode="optimization",
            rng=rng,
        )
        sampler.initialize(Schedule.random(scenario.network, rng))

        # Track violations
        violation_count = 0

        for _ in range(n_steps):
            sampler.step()
            # Check if current schedule violates any constraint
            # (waiting time outside [w_min, w_max])
            current_schedule = sampler.current_schedule
            for transfer in modified_transfers.values():
                wait_time = transfer.compute_waiting_time(current_schedule, scenario.network)
                if wait_time < transfer.params.w_min or wait_time > transfer.params.w_max:
                    violation_count += 1
                    break  # Count once per step

        result = sampler.run(n_steps=0, log_every=0)  # Just get final stats

        points.append(
            BarrierSensitivityPoint(
                M_barrier=M,
                best_cost=result.best_cost,
                mean_cost=float(np.mean(result.costs)) if result.costs else result.best_cost,
                violation_rate=violation_count / n_steps,
                acceptance_rate=result.acceptance_rate,
            )
        )

    return BarrierSensitivityResult(
        scenario_name=scenario.metadata.name,
        points=points,
    )


def run_lambda_sweep(
    scenario: SyntheticScenario,
    lambda_grid: dict[str, list[float]] | None = None,
    n_steps: int = 5000,
    beta: float = 1.0,
    seed: int = 42,
) -> LambdaSweepResult:
    """Sweep over cost function weights λ.

    Args:
        scenario: Test scenario (must have turnarounds for OPS).
        lambda_grid: Dict with keys "pax", "ops", "rob", values are lists to test.
            Defaults to pax=[1.0], ops=[0.1, 1.0, 10.0], rob=[0.01, 0.1, 1.0].
        n_steps: MCMC steps per configuration.
        beta: Inverse temperature.
        seed: Random seed.

    Returns:
        LambdaSweepResult with measurements per λ combination.
    """
    if lambda_grid is None:
        lambda_grid = {
            "pax": [1.0],
            "ops": [0.1, 1.0, 10.0],
            "rob": [0.01, 0.1, 1.0],
        }

    # Create component cost functions
    transfer_cost_fn = TransferCost(transfers=scenario.transfers)

    turnaround_cost_fn = None
    if scenario.turnarounds is not None:
        turnaround_cost_fn = TurnaroundCost(turnarounds=scenario.turnarounds)

    robustness_cost_fn = RobustnessCost(
        transfers=scenario.transfers,
        delay_distribution=DelayDistribution(rate=0.5, alpha=0.95),  # rate=1/mean
    )

    points: list[LambdaSweepPoint] = []

    for lambda_pax in lambda_grid["pax"]:
        for lambda_ops in lambda_grid["ops"]:
            for lambda_rob in lambda_grid["rob"]:
                # Build total cost with these weights
                total_cost = TotalCost(
                    transfer_cost=transfer_cost_fn,
                    operational_cost=turnaround_cost_fn,
                    robustness_cost=robustness_cost_fn,
                    pax_weight=lambda_pax,
                    ops_weight=lambda_ops,
                    rob_weight=lambda_rob,
                )

                # Run MCMC
                rng = np.random.default_rng(seed)
                proposals, weights = build_proposals(scenario, rng)

                sampler = MCMCSampler(
                    network=scenario.network,
                    cost_fn=total_cost,
                    proposals=proposals,
                    proposal_weights=weights,
                    beta=beta,
                    mode="optimization",
                    rng=rng,
                )
                sampler.initialize(Schedule.random(scenario.network, rng))
                result = sampler.run(n_steps=n_steps, log_every=0)

                # Compute component costs for best solution
                best_schedule = result.best_schedule
                cost_pax = transfer_cost_fn(best_schedule, scenario.network)
                cost_rob = robustness_cost_fn(best_schedule, scenario.network)
                cost_ops = (
                    turnaround_cost_fn(best_schedule, scenario.network)
                    if turnaround_cost_fn
                    else 0.0
                )

                points.append(
                    LambdaSweepPoint(
                        lambda_pax=lambda_pax,
                        lambda_ops=lambda_ops,
                        lambda_rob=lambda_rob,
                        best_cost=result.best_cost,
                        cost_pax=cost_pax,
                        cost_ops=cost_ops,
                        cost_rob=cost_rob,
                        acceptance_rate=result.acceptance_rate,
                    )
                )

    return LambdaSweepResult(
        scenario_name=scenario.metadata.name,
        points=points,
    )


def compute_pareto_front(
    points: list[LambdaSweepPoint],
    objective1: str,
    objective2: str,
) -> ParetoFrontResult:
    """Compute Pareto front from lambda sweep results.

    Args:
        points: Results from lambda sweep.
        objective1: Name of first objective ("cost_pax", "cost_ops", "cost_rob").
        objective2: Name of second objective.

    Returns:
        ParetoFrontResult with Pareto-optimal points.
    """
    # Extract objective values
    obj_values = [
        (getattr(p, objective1), getattr(p, objective2))
        for p in points
    ]

    # Compute Pareto front (minimize both objectives)
    pareto_front = []
    dominated = []

    for i, (x1, y1) in enumerate(obj_values):
        is_dominated = False
        for j, (x2, y2) in enumerate(obj_values):
            if i != j and x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                is_dominated = True
                break

        if is_dominated:
            dominated.append((x1, y1))
        else:
            pareto_front.append((x1, y1))

    # Sort Pareto front by first objective
    pareto_front.sort(key=lambda p: p[0])

    return ParetoFrontResult(
        scenario_name=points[0].lambda_pax if points else "unknown",  # dummy
        points=pareto_front,
        labels=(objective1, objective2),
        dominated_points=dominated,
    )


# ---------------------------------------------------------------------------
# 8. Benchmark Methods
# ---------------------------------------------------------------------------

@dataclass
class RandomSearchResult:
    """Results from random search baseline."""

    scenario_name: str
    n_samples: int
    best_cost: float
    best_solution: dict[str, int]
    mean_cost: float
    median_cost: float
    wall_time_s: float


@dataclass
class SimulatedAnnealingResult:
    """Results from simulated annealing."""

    scenario_name: str
    n_steps: int
    best_cost: float
    best_solution: dict[str, int]
    acceptance_rate: float
    final_temperature: float
    wall_time_s: float
    cost_trajectory: list[float]  # Best cost at each checkpoint


@dataclass
class BenchmarkComparisonResult:
    """Comparison of multiple methods."""

    scenario_name: str
    methods: list[str]
    best_costs: list[float]
    wall_times: list[float]
    n_evaluations: list[int]  # Cost function evaluations per method


def run_random_search(
    scenario: SyntheticScenario,
    n_samples: int = 10000,
    seed: int = 42,
) -> RandomSearchResult:
    """Run random search baseline.

    Args:
        scenario: Test scenario.
        n_samples: Number of random schedules to evaluate.
        seed: Random seed.

    Returns:
        RandomSearchResult with best found solution.
    """
    cost_fn = build_cost_fn(scenario)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()

    best_cost = float("inf")
    best_solution = None
    costs = []

    for _ in range(n_samples):
        schedule = Schedule.random(scenario.network, rng)
        cost = cost_fn(schedule, scenario.network)
        costs.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_solution = schedule.offsets.copy()

    wall_time = time.perf_counter() - t0

    return RandomSearchResult(
        scenario_name=scenario.metadata.name,
        n_samples=n_samples,
        best_cost=best_cost,
        best_solution=best_solution,
        mean_cost=float(np.mean(costs)),
        median_cost=float(np.median(costs)),
        wall_time_s=wall_time,
    )


def run_simulated_annealing(
    scenario: SyntheticScenario,
    n_steps: int = 10000,
    T_init: float = 100.0,
    T_final: float = 0.01,
    cooling_schedule: str = "geometric",
    seed: int = 42,
) -> SimulatedAnnealingResult:
    """Run simulated annealing with same proposals as MCMC.

    Args:
        scenario: Test scenario.
        n_steps: Total annealing steps.
        T_init: Initial temperature.
        T_final: Final temperature.
        cooling_schedule: "geometric" or "linear".
        seed: Random seed.

    Returns:
        SimulatedAnnealingResult with best found solution.
    """
    cost_fn = build_cost_fn(scenario)
    rng = np.random.default_rng(seed)
    proposals, weights = build_proposals(scenario, rng, include_smart_repair=False)

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Initialize
    current_schedule = Schedule.random(scenario.network, rng)
    current_cost = cost_fn(current_schedule, scenario.network)

    best_cost = current_cost
    best_solution = current_schedule.offsets.copy()

    accepted = 0
    cost_trajectory = []
    checkpoint_interval = n_steps // 100

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Temperature schedule
        if cooling_schedule == "geometric":
            alpha = (T_final / T_init) ** (1 / n_steps)
            T = T_init * (alpha ** step)
        else:  # linear
            T = T_init + (T_final - T_init) * (step / n_steps)

        # Select proposal
        proposal_idx = rng.choice(len(proposals), p=weights)
        proposal = proposals[proposal_idx]

        # Propose move
        new_schedule, hastings, changed_lines = proposal.propose(current_schedule, scenario.network)
        new_cost = cost_fn(new_schedule, scenario.network)

        # Metropolis acceptance with temperature
        delta = new_cost - current_cost
        beta = 1.0 / T if T > 0 else float("inf")
        log_accept_prob = -beta * delta + np.log(hastings)
        accept = log_accept_prob >= 0 or rng.random() < np.exp(log_accept_prob)

        if accept:
            current_schedule = new_schedule
            current_cost = new_cost
            accepted += 1

            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = current_schedule.offsets.copy()

        if step % checkpoint_interval == 0:
            cost_trajectory.append(best_cost)

    wall_time = time.perf_counter() - t0

    return SimulatedAnnealingResult(
        scenario_name=scenario.metadata.name,
        n_steps=n_steps,
        best_cost=best_cost,
        best_solution=best_solution,
        acceptance_rate=accepted / n_steps,
        final_temperature=T,
        wall_time_s=wall_time,
        cost_trajectory=cost_trajectory,
    )


def run_benchmark_comparison(
    scenario: SyntheticScenario,
    methods: list[str] | None = None,
    budget: int = 10000,
    seed: int = 42,
) -> BenchmarkComparisonResult:
    """Compare multiple optimization methods on same scenario.

    Args:
        scenario: Test scenario.
        methods: List of method names to compare. Options: "mcmc", "pt", "sa", "random".
            Defaults to ["random", "mcmc", "sa"].
        budget: Total cost evaluations budget (approximately).
        seed: Random seed.

    Returns:
        BenchmarkComparisonResult with comparison data.
    """
    if methods is None:
        methods = ["random", "mcmc", "sa"]

    method_names = []
    best_costs = []
    wall_times = []
    n_evals = []

    for method in methods:
        if method == "random":
            result = run_random_search(scenario, n_samples=budget, seed=seed)
            method_names.append("Random Search")
            best_costs.append(result.best_cost)
            wall_times.append(result.wall_time_s)
            n_evals.append(result.n_samples)

        elif method == "mcmc":
            cost_fn = build_cost_fn(scenario)
            rng = np.random.default_rng(seed)
            proposals, weights = build_proposals(scenario, rng)

            sampler = MCMCSampler(
                network=scenario.network,
                cost_fn=cost_fn,
                proposals=proposals,
                proposal_weights=weights,
                beta=1.0,
                mode="optimization",
                rng=rng,
            )
            sampler.initialize(Schedule.random(scenario.network, rng))

            t0 = time.perf_counter()
            result = sampler.run(n_steps=budget, log_every=0)
            wall_time = time.perf_counter() - t0

            method_names.append("MCMC (single chain)")
            best_costs.append(result.best_cost)
            wall_times.append(wall_time)
            n_evals.append(budget)

        elif method == "sa":
            result = run_simulated_annealing(scenario, n_steps=budget, seed=seed)
            method_names.append("Simulated Annealing")
            best_costs.append(result.best_cost)
            wall_times.append(result.wall_time_s)
            n_evals.append(result.n_steps)

        elif method == "pt":
            cost_fn = build_cost_fn(scenario)
            rng = np.random.default_rng(seed)
            proposals, weights = build_proposals(scenario, rng)

            n_replicas = 4
            steps_per_replica = budget // n_replicas

            pt = ParallelTempering(
                network=scenario.network,
                cost_fn=cost_fn,
                proposals=proposals,
                proposal_weights=weights,
                n_replicas=n_replicas,
                beta_min=0.1,
                beta_max=5.0,
                swap_interval=10,
                mode="optimization",
                rng=rng,
            )
            pt.initialize()

            t0 = time.perf_counter()
            result = pt.run(n_steps=steps_per_replica, log_every=0)
            wall_time = time.perf_counter() - t0

            method_names.append("Parallel Tempering (M=4)")
            best_costs.append(result.best_cost)
            wall_times.append(wall_time)
            n_evals.append(budget)

    return BenchmarkComparisonResult(
        scenario_name=scenario.metadata.name,
        methods=method_names,
        best_costs=best_costs,
        wall_times=wall_times,
        n_evaluations=n_evals,
    )


# ---------------------------------------------------------------------------
# 9. Profiling & Optimization
# ---------------------------------------------------------------------------

@dataclass
class ProfilingPoint:
    """Timing breakdown for a single run."""

    component: str
    time_ms: float
    percentage: float


@dataclass
class ProfilingResult:
    """Profiling analysis result."""

    scenario_name: str
    n_steps: int
    total_time_s: float
    breakdown: list[ProfilingPoint]


def run_profiling_analysis(
    scenario: SyntheticScenario,
    n_steps: int = 1000,
    seed: int = 42,
) -> ProfilingResult:
    """Profile MCMC execution to identify bottlenecks.

    Args:
        scenario: Test scenario.
        n_steps: Steps to profile.
        seed: Random seed.

    Returns:
        ProfilingResult with timing breakdown.
    """
    cost_fn = build_cost_fn(scenario)
    rng = np.random.default_rng(seed)
    proposals, weights = build_proposals(scenario, rng)

    sampler = MCMCSampler(
        network=scenario.network,
        cost_fn=cost_fn,
        proposals=proposals,
        proposal_weights=weights,
        beta=1.0,
        mode="optimization",
        rng=rng,
    )
    sampler.initialize(Schedule.random(scenario.network, rng))

    # Time proposal + cost evaluation
    time_propose = 0.0
    time_cost = 0.0
    time_accept = 0.0

    for _ in range(n_steps):
        # Propose
        t0 = time.perf_counter()
        proposal_idx = rng.choice(len(proposals), p=weights)
        proposal = proposals[proposal_idx]
        new_schedule, hastings, changed_lines = proposal.propose(sampler.current_schedule, scenario.network)
        time_propose += time.perf_counter() - t0

        # Cost evaluation
        t0 = time.perf_counter()
        new_cost = cost_fn(new_schedule, scenario.network)
        time_cost += time.perf_counter() - t0

        # Acceptance
        t0 = time.perf_counter()
        delta = new_cost - sampler.current_cost
        log_accept = -sampler.beta * delta + np.log(hastings)
        accept = log_accept >= 0 or rng.random() < np.exp(log_accept)
        if accept:
            sampler._current_schedule = new_schedule
            sampler._current_cost = new_cost
        time_accept += time.perf_counter() - t0

    total_time = time_propose + time_cost + time_accept

    breakdown = [
        ProfilingPoint(
            component="Proposal",
            time_ms=(time_propose / n_steps) * 1000,
            percentage=(time_propose / total_time) * 100,
        ),
        ProfilingPoint(
            component="Cost Evaluation",
            time_ms=(time_cost / n_steps) * 1000,
            percentage=(time_cost / total_time) * 100,
        ),
        ProfilingPoint(
            component="Acceptance Decision",
            time_ms=(time_accept / n_steps) * 1000,
            percentage=(time_accept / total_time) * 100,
        ),
    ]

    return ProfilingResult(
        scenario_name=scenario.metadata.name,
        n_steps=n_steps,
        total_time_s=total_time,
        breakdown=breakdown,
    )
