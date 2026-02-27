"""MCMC history recording and playback."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import TransferSet
    from src.cost.transfer_cost import TransferCost
    from src.mcmc.proposals.base import Proposal


logger = logging.getLogger(__name__)


@dataclass
class MCMCStep:
    """Record of a single MCMC step."""

    step_number: int
    schedule_before: dict[str, int]
    schedule_after: dict[str, int]
    cost_before: float
    cost_after: float
    proposed_line: str | None
    proposed_delta: int
    accepted: bool
    acceptance_prob: float
    proposal_name: str


@dataclass
class MCMCHistory:
    """Complete history of an MCMC run for visualization."""

    steps: list[MCMCStep] = field(default_factory=list)
    period: int = 60
    initial_schedule: dict[str, int] = field(default_factory=dict)
    optimal_schedule: dict[str, int] | None = None
    optimal_cost: float | None = None

    def __len__(self) -> int:
        return len(self.steps)

    def get_schedule_at_step(self, step: int) -> dict[str, int]:
        """Get schedule state at a given step."""
        if step < 0:
            return dict(self.initial_schedule)
        if step >= len(self.steps):
            step = len(self.steps) - 1
        return dict(self.steps[step].schedule_after)

    def get_cost_at_step(self, step: int) -> float:
        """Get cost at a given step."""
        if step < 0 or step >= len(self.steps):
            return float("inf")
        return self.steps[step].cost_after

    def get_costs(self) -> list[float]:
        """Get all costs as a list."""
        return [s.cost_after for s in self.steps]

    def get_acceptance_rate_up_to(self, step: int) -> float:
        """Get acceptance rate up to a given step."""
        if step <= 0:
            return 0.0
        accepted = sum(1 for s in self.steps[:step + 1] if s.accepted)
        return accepted / (step + 1)

    def get_best_up_to(self, step: int) -> tuple[dict[str, int], float]:
        """Get best schedule and cost up to a given step."""
        if step < 0:
            return dict(self.initial_schedule), float("inf")

        best_cost = float("inf")
        best_schedule = dict(self.initial_schedule)

        for s in self.steps[:step + 1]:
            if s.cost_after < best_cost:
                best_cost = s.cost_after
                best_schedule = dict(s.schedule_after)

        return best_schedule, best_cost

    def summary_at_step(self, step: int) -> dict:
        """Get summary statistics at a given step."""
        if step < 0:
            return {
                "step": -1,
                "cost": float("inf"),
                "best_cost": float("inf"),
                "acceptance_rate": 0.0,
                "accepted": False,
            }

        step = min(step, len(self.steps) - 1)
        s = self.steps[step]
        _, best_cost = self.get_best_up_to(step)

        return {
            "step": step,
            "cost": s.cost_after,
            "best_cost": best_cost,
            "acceptance_rate": self.get_acceptance_rate_up_to(step),
            "accepted": s.accepted,
            "proposed_line": s.proposed_line,
            "proposed_delta": s.proposed_delta,
            "proposal_name": s.proposal_name,
            "acceptance_prob": s.acceptance_prob,
        }


@dataclass
class MCMCRecorder:
    """MCMC sampler that records full history for visualization.

    Similar to MCMCSampler but stores every step for later playback.
    """

    network: Network
    cost_fn: TransferCost
    proposals: list[Proposal]
    proposal_weights: list[float] | None = None
    beta: float = 1.0
    rng: Generator = field(default_factory=lambda: np.random.default_rng())

    # Internal state
    _history: MCMCHistory = field(default_factory=MCMCHistory, init=False)
    _current_schedule: Schedule | None = field(default=None, init=False)
    _current_cost: float = field(default=float("inf"), init=False)

    def __post_init__(self) -> None:
        if self.proposal_weights is None:
            self.proposal_weights = [1.0] * len(self.proposals)
        total = sum(self.proposal_weights)
        self._normalized_weights = [w / total for w in self.proposal_weights]

    def initialize(
        self,
        schedule: Schedule,
        optimal_schedule: dict[str, int] | None = None,
        optimal_cost: float | None = None,
    ) -> None:
        """Initialize the recorder with a starting schedule.

        Args:
            schedule: Initial schedule state.
            optimal_schedule: Known optimal for comparison (optional).
            optimal_cost: Known optimal cost (optional).
        """
        from src.core.schedule import Schedule

        self._current_schedule = schedule.copy()
        self._current_cost = self.cost_fn(schedule, self.network)

        self._history = MCMCHistory(
            steps=[],
            period=self.network.period,
            initial_schedule=dict(schedule.offsets),
            optimal_schedule=optimal_schedule,
            optimal_cost=optimal_cost,
        )

        logger.info(f"MCMCRecorder initialized with cost {self._current_cost:.2f}")

    def run(self, n_steps: int, log_every: int = 1000) -> MCMCHistory:
        """Run MCMC and record full history.

        Args:
            n_steps: Number of MCMC steps.
            log_every: Log progress every N steps (0 = no logging).

        Returns:
            MCMCHistory with all recorded steps.
        """
        if self._current_schedule is None:
            raise RuntimeError("Recorder not initialized. Call initialize() first.")

        for i in range(n_steps):
            # Select proposal
            proposal_idx = self.rng.choice(
                len(self.proposals), p=self._normalized_weights
            )
            proposal = self.proposals[proposal_idx]

            # Record state before
            schedule_before = dict(self._current_schedule.offsets)
            cost_before = self._current_cost

            # Generate proposal
            new_schedule, hastings_ratio, changed_lines = proposal.propose(
                self._current_schedule, self.network
            )

            # Compute delta cost
            delta_cost = self.cost_fn.delta(
                self._current_schedule,
                new_schedule,
                changed_lines,
                self.network,
            )
            new_cost = self._current_cost + delta_cost

            # Compute acceptance probability
            log_alpha = -self.beta * delta_cost + np.log(hastings_ratio)
            alpha = min(1.0, np.exp(log_alpha))

            # Accept/reject
            u = self.rng.random()
            accept = u < alpha

            # Determine what changed
            proposed_line = list(changed_lines)[0] if changed_lines else None
            if proposed_line:
                proposed_delta = (
                    new_schedule.get_offset(proposed_line)
                    - self._current_schedule.get_offset(proposed_line)
                )
                # Handle wraparound
                if proposed_delta > self.network.period // 2:
                    proposed_delta -= self.network.period
                elif proposed_delta < -self.network.period // 2:
                    proposed_delta += self.network.period
            else:
                proposed_delta = 0

            # Update state if accepted
            if accept:
                self._current_schedule = new_schedule
                self._current_cost = new_cost

            # Record step
            step = MCMCStep(
                step_number=i,
                schedule_before=schedule_before,
                schedule_after=dict(self._current_schedule.offsets),
                cost_before=cost_before,
                cost_after=self._current_cost,
                proposed_line=proposed_line,
                proposed_delta=proposed_delta,
                accepted=accept,
                acceptance_prob=alpha,
                proposal_name=proposal.name,
            )
            self._history.steps.append(step)

            # Logging
            if log_every > 0 and (i + 1) % log_every == 0:
                ar = self._history.get_acceptance_rate_up_to(i)
                _, best = self._history.get_best_up_to(i)
                logger.info(
                    f"Step {i + 1}/{n_steps}: "
                    f"cost={self._current_cost:.2f}, "
                    f"best={best:.2f}, "
                    f"accept_rate={ar:.3f}"
                )

        return self._history

    @property
    def history(self) -> MCMCHistory:
        """Get the recorded history."""
        return self._history
