"""Single line shift proposal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from src.core.network import Network
    from src.core.schedule import Schedule


@dataclass
class SingleLineShift:
    """Proposal that shifts a single line's offset by a Gaussian perturbation.

    This is the simplest MCMC proposal: randomly select one line and shift
    its offset by a value drawn from N(0, sigma^2).

    The proposal is symmetric, so the Hastings ratio is always 1.0.
    When line_weights are provided (for inertia), selection is biased
    but weights are state-independent, so the Hastings ratio remains 1.0.

    Args:
        sigma: Standard deviation of the Gaussian perturbation (in minutes).
        rng: NumPy random generator for reproducibility.
        line_weights: Optional dict mapping line_id to selection weight.
            When provided, P(L) is proportional to line_weights[L].
            Used by Parallel Tempering inertia mechanism.
    """

    sigma: float = 5.0
    rng: Generator = field(default_factory=lambda: np.random.default_rng())
    line_weights: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return "single_line_shift"

    def propose(
        self,
        schedule: Schedule,
        network: Network,
    ) -> tuple[Schedule, float, set[str]]:
        """Propose shifting a single line.

        Args:
            schedule: Current schedule state.
            network: Network definition.

        Returns:
            Tuple of (new_schedule, hastings_ratio=1.0, {changed_line_id}).
        """
        # Select a random line (with optional inertia weights)
        line_ids = network.line_ids
        if self.line_weights is not None:
            weights = np.array([self.line_weights.get(lid, 1.0) for lid in line_ids])
            weights = weights / weights.sum()
            line_id = self.rng.choice(line_ids, p=weights)
        else:
            line_id = self.rng.choice(line_ids)

        # Generate Gaussian perturbation
        delta = self.rng.normal(0, self.sigma)
        delta_int = int(round(delta))

        # Create new schedule with shifted line
        new_schedule = schedule.copy()
        new_schedule.shift_line(line_id, delta_int)

        # Symmetric proposal -> Hastings ratio = 1.0
        # (line_weights are state-independent, so forward/backward P(L) are equal)
        return new_schedule, 1.0, {line_id}


@dataclass
class UniformLineShift:
    """Proposal that shifts a single line's offset uniformly.

    Shifts a randomly selected line by a value drawn uniformly from
    [-max_shift, max_shift].

    The proposal is symmetric, so the Hastings ratio is always 1.0.

    Args:
        max_shift: Maximum shift in either direction (in minutes).
        rng: NumPy random generator for reproducibility.
        line_weights: Optional dict mapping line_id to selection weight.
    """

    max_shift: int = 10
    rng: Generator = field(default_factory=lambda: np.random.default_rng())
    line_weights: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return "uniform_line_shift"

    def propose(
        self,
        schedule: Schedule,
        network: Network,
    ) -> tuple[Schedule, float, set[str]]:
        """Propose shifting a single line uniformly.

        Args:
            schedule: Current schedule state.
            network: Network definition.

        Returns:
            Tuple of (new_schedule, hastings_ratio=1.0, {changed_line_id}).
        """
        # Select a random line (with optional inertia weights)
        line_ids = network.line_ids
        if self.line_weights is not None:
            weights = np.array([self.line_weights.get(lid, 1.0) for lid in line_ids])
            weights = weights / weights.sum()
            line_id = self.rng.choice(line_ids, p=weights)
        else:
            line_id = self.rng.choice(line_ids)

        # Generate uniform perturbation
        delta = self.rng.integers(-self.max_shift, self.max_shift + 1)

        # Create new schedule with shifted line
        new_schedule = schedule.copy()
        new_schedule.shift_line(line_id, int(delta))

        # Symmetric proposal -> Hastings ratio = 1.0
        return new_schedule, 1.0, {line_id}
