"""Auto-calibrated PT framework for networks of arbitrary cost scale.

Problem: PT swap probability = exp(-Δβ × ΔE). For random_network_hard
(costs ~90k-96k), betas tuned for small networks (0.5-3.0) produce
swap rates < 0.01% — the replicas never communicate.

Solution: auto-calibrate beta schedule from short cost-variance estimates.

Key formula (Gaussian energy-difference approximation):
  swap_rate ≈ erfc(Δβ · σ_E / √(4))
  → target_swap ≈ 0.23 ⟹ Δβ ≈ 1.6 / σ_E

where σ_E is the std of energy differences between adjacent replicas.
For stationary replicas, σ_E ≈ √2 · σ_single where σ_single = std of costs
at that beta (since Δcost ≈ N(0, σ_i² + σ_{i+1}²) and σ_i ≈ σ_{i+1}).

Usage:
    python -m src.experiments.analysis.pt_calibration
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erfinv

from src.core.schedule import Schedule
from src.experiments.analysis.diagnostic_runner import build_cost_fn, build_proposals
from src.io.synthetic_scenarios import SyntheticScenario, SyntheticScenarioFactory
from src.mcmc.parallel_tempering import InertiaConfig, ParallelTempering
from src.mcmc.sampler import MCMCSampler

logging.basicConfig(level=logging.WARNING)

INERTIA_CONFIG = InertiaConfig(
    modality_inertia={"IC": 20.0, "Regional": 5.0, "bus": 1.0},
)


# ---------------------------------------------------------------------------
# Step 1: Estimate cost statistics at a given beta
# ---------------------------------------------------------------------------

@dataclass
class CostStats:
    """Cost statistics at a given inverse temperature."""
    beta: float
    mean: float
    std: float
    acceptance_rate: float
    n_samples: int


def estimate_cost_stats(
    scenario: SyntheticScenario,
    beta: float,
    n_warmup: int = 3000,
    n_sample: int = 5000,
    seed: int = 42,
) -> CostStats:
    """Run short MCMC at given beta, return cost mean/std after warmup."""
    rng = np.random.default_rng(seed)
    cost_fn = build_cost_fn(scenario)
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

    for _ in range(n_warmup):
        sampler.step()

    costs = []
    for _ in range(n_sample):
        sampler.step()
        costs.append(sampler.current_cost)

    return CostStats(
        beta=beta,
        mean=float(np.mean(costs)),
        std=float(np.std(costs)),
        acceptance_rate=sampler.acceptance_rate,
        n_samples=n_sample,
    )


# ---------------------------------------------------------------------------
# Step 2: Calibrate beta schedule
# ---------------------------------------------------------------------------

def calibrate_beta_schedule(
    scenario: SyntheticScenario,
    beta_max: float,
    n_replicas: int,
    target_swap: float = 0.23,
    n_warmup: int = 3000,
    n_sample: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> list[float]:
    """Compute a beta schedule targeting a given swap acceptance rate.

    Algorithm:
    1. Estimate σ_E (cost std) at beta_max with short MCMC run.
    2. Compute Δβ from Gaussian swap-rate formula:
       Δβ = sqrt(2) * erfinv(1 - 2*target_swap) / (sqrt(2) * σ_E)
         = erfinv(1 - 2*target_swap) / σ_E
    3. Build schedule: beta_k = beta_max - k * Δβ (clip at 1e-6).
    4. If beta_min ends up negative, scale down Δβ proportionally.

    Returns:
        Sorted list of betas [beta_min, ..., beta_max] (ascending).
    """
    stats = estimate_cost_stats(
        scenario, beta_max,
        n_warmup=n_warmup, n_sample=n_sample, seed=seed,
    )
    if verbose:
        print(f"    [calibrate] beta={beta_max:.4f}: mean={stats.mean:.1f}, "
              f"std={stats.std:.1f}, accept={stats.acceptance_rate:.3f}")

    sigma = stats.std
    if sigma < 1e-6:
        # Degenerate: chain not mixing, use heuristic fallback
        sigma = abs(stats.mean) * 0.01 + 1.0
        if verbose:
            print(f"    [calibrate] σ≈0 — using fallback σ={sigma:.1f}")

    # Gaussian approx: swap_rate ≈ erfc(Δβ·σ / √2) / 2 for one-sided tail
    # More precisely for symmetric ΔE ~ N(0, 2σ²):
    #   swap_rate = erfc(Δβ·σ / sqrt(2)) / 2  [Kofke, 2002 approximation]
    # Inverting: Δβ = sqrt(2) * erfinv(1 - 2*target_swap) / σ
    raw_erfinv = float(erfinv(1.0 - 2.0 * target_swap))
    delta_beta = abs(np.sqrt(2.0) * raw_erfinv / sigma)

    if verbose:
        print(f"    [calibrate] sigma_E={sigma:.2f}  d_beta={delta_beta:.6f}  "
              f"-> beta_min={beta_max - (n_replicas-1)*delta_beta:.6f}")

    # Build schedule (descending index = ascending beta)
    betas_raw = [beta_max - i * delta_beta for i in range(n_replicas)]
    # Clip negative betas — scale down if needed
    if betas_raw[-1] <= 0:
        max_span = beta_max * 0.999
        delta_beta = max_span / (n_replicas - 1)
        betas_raw = [beta_max - i * delta_beta for i in range(n_replicas)]
        if verbose:
            print(f"    [calibrate] clipped: d_beta rescaled to {delta_beta:.6f}")

    betas = sorted(max(b, 1e-8) for b in betas_raw)
    return betas


# ---------------------------------------------------------------------------
# Step 3: Run PT with explicit beta list (bypasses geometric schedule)
# ---------------------------------------------------------------------------

def run_pt_with_betas(
    scenario: SyntheticScenario,
    betas: list[float],
    n_steps_per_replica: int,
    use_inertia: bool = False,
    seed: int = 42,
    checkpoint_every: int = 10000,
) -> dict[str, Any]:
    """Run PT using given explicit beta schedule.

    Since ParallelTempering only supports geometric schedules, we map the
    custom betas to the closest geometric parameters (beta_min, beta_max)
    and use the same n_replicas — the internal schedule will differ slightly,
    but for the purposes of this experiment we accept that and report the
    actual schedule computed.

    For accurate custom schedules, PT is run with beta_min=betas[0],
    beta_max=betas[-1], and n_replicas=len(betas). The geometric spacing
    approximates our calibrated spacing when betas are nearly geometric.
    """
    rng = np.random.default_rng(seed)
    cost_fn = build_cost_fn(scenario)
    proposals, weights = build_proposals(scenario, rng)

    n_replicas = len(betas)
    beta_min = betas[0]
    beta_max = betas[-1]

    inertia = INERTIA_CONFIG if use_inertia else None

    t0 = time.perf_counter()
    pt = ParallelTempering(
        network=scenario.network,
        cost_fn=cost_fn,
        proposals=proposals,
        proposal_weights=weights,
        n_replicas=n_replicas,
        beta_min=beta_min,
        beta_max=beta_max,
        swap_interval=10,
        mode="optimization",
        inertia_config=inertia,
        rng=rng,
    )
    pt.initialize()

    checkpoints = []
    ckpt_interval = max(1, checkpoint_every // n_replicas)
    for step in range(1, n_steps_per_replica + 1):
        pt.step()
        if step % ckpt_interval == 0:
            checkpoints.append({
                "step_budget": step * n_replicas,
                "best_cost": pt.best_cost,
            })

    # Final stats
    result = pt.run(n_steps=0, log_every=0)
    wall_time = time.perf_counter() - t0

    return {
        "n_replicas": n_replicas,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "actual_betas": list(pt.temperature_schedule),
        "calibrated_betas": betas,
        "use_inertia": use_inertia,
        "steps_per_replica": n_steps_per_replica,
        "best_cost": pt.best_cost,
        "cold_chain_mean": float(np.mean(result.cold_chain_costs)) if result.cold_chain_costs else pt.best_cost,
        "swap_rates": list(result.swap_acceptance_rates),
        "swap_avg": float(np.mean(result.swap_acceptance_rates)) if result.swap_acceptance_rates else 0.0,
        "swap_min": float(min(result.swap_acceptance_rates)) if result.swap_acceptance_rates else 0.0,
        "per_replica_acceptance": list(result.per_replica_acceptance_rates),
        "wall_time_s": wall_time,
        "checkpoints": checkpoints,
    }


# ---------------------------------------------------------------------------
# Step 4: Iterative calibration loop
# ---------------------------------------------------------------------------

def run_iterative_calibration(
    scenario: SyntheticScenario,
    beta_max: float = 3.0,
    n_replicas_list: list[int] | None = None,
    target_swap: float = 0.23,
    n_steps: int = 500_000,
    max_iterations: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Iteratively calibrate beta and run PT until swap rates are healthy.

    In each iteration:
    1. Estimate σ_E at beta_max → compute Δβ.
    2. Run PT with calibrated betas.
    3. Check actual swap rates. If still too low, halve Δβ and retry.

    Args:
        scenario: Scenario to optimize.
        beta_max: Fixed cold-chain beta (quality of cold chain).
        n_replicas_list: Replica counts to compare.
        target_swap: Target swap acceptance rate (~0.23 optimal for MCMC).
        n_steps: Total step budget per run.
        max_iterations: Max calibration iterations per replica count.
        seed: Random seed.

    Returns:
        List of result dicts, one per (n_replicas, iteration) combination.
    """
    if n_replicas_list is None:
        n_replicas_list = [8, 12, 16]

    all_results = []

    for n_rep in n_replicas_list:
        print(f"\n  -- M={n_rep} replicas --")
        steps_per_replica = n_steps // n_rep

        for iteration in range(1, max_iterations + 1):
            print(f"    Iteration {iteration}/{max_iterations}: calibrating betas...")
            betas = calibrate_beta_schedule(
                scenario, beta_max, n_replicas=n_rep,
                target_swap=target_swap,
                n_warmup=3000, n_sample=8000,
                seed=seed + iteration,
                verbose=True,
            )
            print(f"    Betas: {[f'{b:.5f}' for b in betas]}")

            for use_inertia in [False, True]:
                label = f"M{n_rep}_iter{iteration}" + ("_inertia" if use_inertia else "")
                print(f"    Running {label} (inertia={use_inertia})...")

                result = run_pt_with_betas(
                    scenario, betas,
                    n_steps_per_replica=steps_per_replica,
                    use_inertia=use_inertia,
                    seed=seed,
                    checkpoint_every=20000,
                )
                result["label"] = label
                result["iteration"] = iteration
                result["calibrated_target_swap"] = target_swap

                print(f"      best={result['best_cost']:.1f}  "
                      f"swap_avg={result['swap_avg']:.3f}  "
                      f"swap_min={result['swap_min']:.3f}  "
                      f"time={result['wall_time_s']:.1f}s")
                all_results.append(result)

            # Check if swap rates are healthy after first inertia=False run
            first_result = all_results[-2]  # no-inertia result
            if first_result["swap_avg"] >= 0.15 and first_result["swap_min"] >= 0.05:
                print(f"    ✓ Swap rates healthy — skipping remaining iterations")
                break
            elif iteration < max_iterations:
                print(f"    ✗ swap_avg={first_result['swap_avg']:.3f} < 0.15 "
                      f"— narrowing range further")
                # Halve the effective Δβ by increasing target_swap
                target_swap = min(target_swap + 0.10, 0.40)

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_calibration_experiment(seed: int = 42) -> None:
    factory = SyntheticScenarioFactory()

    BETA_MAX = 3.0
    N_STEPS = 500_000
    SCENARIOS = [
        ("multi_hub_mesh", "hard", {}),
        ("random_network", "hard", {"seed": 42}),
    ]

    output: dict[str, Any] = {}

    for stype, diff, kwargs in SCENARIOS:
        scenario = factory.create(stype, difficulty=diff, **kwargs)
        name = scenario.metadata.name
        print(f"\n{'='*80}")
        print(f"Scenario: {name}  ({scenario.metadata.n_lines}L, "
              f"{scenario.metadata.n_transfers}T)  budget={N_STEPS}")
        print(f"{'='*80}")

        # Quick landscape: cost stats at several betas
        print("\n  Cost statistics at probe betas:")
        probe_betas = [0.001, 0.01, 0.1, 0.5, 1.0, 3.0]
        stats_table = []
        for pb in probe_betas:
            s = estimate_cost_stats(scenario, pb, n_warmup=1000, n_sample=3000, seed=seed)
            print(f"    b={pb:6.3f}: mean={s.mean:12.1f}  std={s.std:12.1f}  "
                  f"accept={s.acceptance_rate:.3f}")
            stats_table.append({
                "beta": pb, "mean": s.mean, "std": s.std,
                "acceptance_rate": s.acceptance_rate,
            })

        # Run iterative calibration
        results = run_iterative_calibration(
            scenario,
            beta_max=BETA_MAX,
            n_replicas_list=[8, 12, 16],
            target_swap=0.23,
            n_steps=N_STEPS,
            max_iterations=3,
            seed=seed,
        )

        # Summary table
        print(f"\n  SUMMARY for {name}:")
        print(f"  {'Label':<30} {'Best':>10} {'Swap Avg':>10} {'Swap Min':>10} "
              f"{'b_min':>10} {'b_max':>8} {'Time':>8}")
        print(f"  {'-'*90}")
        for r in results:
            print(f"  {r['label']:<30} {r['best_cost']:>10.1f} "
                  f"{r['swap_avg']:>10.3f} {r['swap_min']:>10.3f} "
                  f"{r['beta_min']:>10.6f} {r['beta_max']:>8.3f} "
                  f"{r['wall_time_s']:>8.1f}s")

        output[name] = {
            "n_lines": scenario.metadata.n_lines,
            "n_transfers": scenario.metadata.n_transfers,
            "probe_stats": stats_table,
            "pt_results": results,
        }

    out_path = Path("results/diagnostics/pt_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n\nSaved to {out_path}")


if __name__ == "__main__":
    run_calibration_experiment()
