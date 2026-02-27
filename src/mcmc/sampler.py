"""MCMC sampler with Metropolis-Hastings algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from src.mcmc.adaptation import ProposalAdapter

if TYPE_CHECKING:
    from numpy.random import Generator

    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import TransferSet
    from src.cost.transfer_cost import CostFunction
    from src.mcmc.proposals.base import Proposal

logger = logging.getLogger(__name__)


@dataclass
class MCMCResult:
    """Result of an MCMC run.

    Args:
        samples: List of sampled schedules.
        costs: Cost at each sample.
        acceptance_rate: Fraction of accepted proposals.
        best_schedule: Schedule with lowest cost found.
        best_cost: Lowest cost found.
    """

    samples: list[Schedule]
    costs: list[float]
    acceptance_rate: float
    best_schedule: Schedule
    best_cost: float

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "n_samples": len(self.samples),
            "acceptance_rate": self.acceptance_rate,
            "best_cost": self.best_cost,
            "mean_cost": np.mean(self.costs),
            "std_cost": np.std(self.costs),
            "min_cost": np.min(self.costs),
            "max_cost": np.max(self.costs),
        }


@dataclass
class MCMCSampler:
    """Metropolis-Hastings MCMC sampler.

    Implements the basic MH algorithm for sampling from the Boltzmann
    distribution over schedules.

    Args:
        network: Transport network definition.
        cost_fn: Cost function to minimize.
        proposals: List of proposal move generators.
        proposal_weights: Weights for selecting proposals (will be normalized).
        beta: Inverse temperature (higher = more greedy).
        mode: "sampling" (formal MCMC) or "optimization" (best-so-far tracking).
        rng: NumPy random generator.
    """

    network: Network
    cost_fn: CostFunction
    proposals: list[Proposal]
    proposal_weights: list[float] | None = None
    beta: float = 1.0
    mode: Literal["sampling", "optimization"] = "optimization"
    rng: Generator = field(default_factory=lambda: np.random.default_rng())

    # Internal state
    _current_schedule: Schedule | None = field(default=None, init=False)
    _current_cost: float = field(default=float("inf"), init=False)
    _best_schedule: Schedule | None = field(default=None, init=False)
    _best_cost: float = field(default=float("inf"), init=False)
    _n_accepted: int = field(default=0, init=False)
    _n_total: int = field(default=0, init=False)
    _adapter: ProposalAdapter | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.proposal_weights is None:
            self.proposal_weights = [1.0] * len(self.proposals)

        # Normalize weights
        total = sum(self.proposal_weights)
        self._normalized_weights = [w / total for w in self.proposal_weights]

        # Create adapter for Mode A (sampling)
        if self.mode == "sampling":
            self._adapter = ProposalAdapter(
                n_proposals=len(self.proposals),
                mode="sampling"
            )
        else:
            self._adapter = None

    def initialize(self, schedule: Schedule) -> None:
        """Initialize the sampler with a starting schedule.

        Args:
            schedule: Initial schedule state.
        """
        self._current_schedule = schedule.copy()
        self._current_cost = self.cost_fn(schedule, self.network)
        self._n_accepted = 0
        self._n_total = 0

        if self.mode == "optimization":
            self._best_schedule = schedule.copy()
            self._best_cost = self._current_cost

        logger.info(f"Initialized MCMC with cost {self._current_cost:.2f}")

    def step(self) -> tuple[Schedule, bool]:
        """Perform one MCMC step.

        Returns:
            Tuple of (current_schedule, was_accepted).

        Raises:
            RuntimeError: If sampler not initialized.
        """
        if self._current_schedule is None:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        # Select a proposal (use adapter in sampling mode)
        if self._adapter is not None:
            weights = self._adapter.get_weights()
        else:
            weights = self._normalized_weights

        proposal_idx = self.rng.choice(len(self.proposals), p=weights)
        proposal = self.proposals[proposal_idx]

        # Generate proposed move
        new_schedule, hastings_ratio, changed_lines = proposal.propose(
            self._current_schedule, self.network
        )

        # Compute cost difference (use incremental if available)
        delta_cost = self.cost_fn.delta(
            self._current_schedule,
            new_schedule,
            changed_lines,
            self.network,
        )
        new_cost = self._current_cost + delta_cost

        # Compute acceptance probability
        # alpha = min(1, exp(-beta * delta_cost) * hastings_ratio)
        log_alpha = -self.beta * delta_cost + np.log(hastings_ratio)
        accept = log_alpha >= 0 or self.rng.random() < np.exp(log_alpha)

        self._n_total += 1

        if accept:
            self._current_schedule = new_schedule
            self._current_cost = new_cost
            self._n_accepted += 1

        # Track best (optimization mode only)
        if self.mode == "optimization" and new_cost < self._best_cost:
            self._best_schedule = new_schedule.copy()
            self._best_cost = new_cost

        # Update adapter (sampling mode only)
        if self._adapter is not None:
            self._adapter.update(proposal_idx, accept)

        return self._current_schedule, accept

    def run(
        self,
        n_steps: int,
        thin: int = 1,
        burn_in: int = 0,
        log_every: int = 1000,
    ) -> MCMCResult:
        """Run the MCMC sampler for a fixed number of steps.

        Args:
            n_steps: Number of MCMC steps to run.
            thin: Keep every thin-th sample (default 1 = keep all).
            burn_in: Number of initial steps to discard (default 0).
            log_every: Log progress every N steps (0 = no logging).

        Returns:
            MCMCResult with samples and diagnostics.

        Raises:
            RuntimeError: If sampler not initialized.
        """
        if self._current_schedule is None:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        samples: list[Schedule] = []
        costs: list[float] = []

        for i in range(n_steps):
            schedule, accepted = self.step()

            # Collect samples after burn-in, respecting thinning
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(schedule.copy())
                costs.append(self._current_cost)

            # Logging
            if log_every > 0 and (i + 1) % log_every == 0:
                ar = self._n_accepted / self._n_total if self._n_total > 0 else 0
                logger.info(
                    f"Step {i + 1}/{n_steps}: "
                    f"cost={self._current_cost:.2f}, "
                    f"best={self._best_cost:.2f}, "
                    f"accept_rate={ar:.3f}"
                )

        acceptance_rate = self._n_accepted / self._n_total if self._n_total > 0 else 0

        if self._best_schedule is not None:
            best_schedule = self._best_schedule.copy()
            best_cost = self._best_cost
        else:
            best_schedule = self._current_schedule.copy()
            best_cost = self._current_cost

        return MCMCResult(
            samples=samples,
            costs=costs,
            acceptance_rate=acceptance_rate,
            best_schedule=best_schedule,
            best_cost=best_cost,
        )

    @property
    def current_schedule(self) -> Schedule | None:
        """Current schedule state."""
        return self._current_schedule

    @property
    def current_cost(self) -> float:
        """Current cost."""
        return self._current_cost

    @property
    def best_schedule(self) -> Schedule | None:
        """Best schedule found so far."""
        return self._best_schedule

    @property
    def best_cost(self) -> float:
        """Best cost found so far."""
        return self._best_cost

    @property
    def acceptance_rate(self) -> float:
        """Current acceptance rate."""
        if self._n_total == 0:
            return 0.0
        return self._n_accepted / self._n_total
