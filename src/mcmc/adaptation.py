"""Adaptive proposal weight and scale tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ProposalAdapter:
    """Adaptive proposal weight and scale tuning.

    Mode A (sampling): diminishing gamma_n = n^{-0.7}, containment p_j >= epsilon,
                       scale sigma in [sigma_min, sigma_max].
    Mode B (optimization): continuous adaptation, no constraints.

    Args:
        n_proposals: Number of proposal types.
        mode: "sampling" or "optimization".
        epsilon: Minimum weight per proposal (Mode A only).
        sigma_min: Minimum proposal scale (Mode A only).
        sigma_max: Maximum proposal scale (Mode A only).
        target_accept: Target acceptance rate for scale tuning.
    """

    n_proposals: int
    mode: Literal["sampling", "optimization"]
    epsilon: float = 0.05
    sigma_min: float = 0.5
    sigma_max: float = 50.0
    target_accept: float = 0.234

    _weights: list[float] = field(default_factory=list, init=False)
    _sigmas: list[float] = field(default_factory=list, init=False)
    _n_step: int = field(default=0, init=False)
    _accept_counts: list[int] = field(default_factory=list, init=False)
    _total_counts: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._weights = [1.0 / self.n_proposals] * self.n_proposals
        self._sigmas = [5.0] * self.n_proposals
        self._accept_counts = [0] * self.n_proposals
        self._total_counts = [0] * self.n_proposals

    def gamma(self) -> float:
        """Diminishing step size (Mode A): gamma_n = (n+1)^{-0.7} -> 0."""
        return (self._n_step + 1) ** (-0.7)

    def update(self, proposal_idx: int, accepted: bool) -> None:
        """Update weights and scales after a step.

        Args:
            proposal_idx: Index of the proposal that was used.
            accepted: Whether the proposal was accepted.
        """
        self._n_step += 1
        self._total_counts[proposal_idx] += 1
        if accepted:
            self._accept_counts[proposal_idx] += 1

        # Compute acceptance rate for this proposal
        tc = self._total_counts[proposal_idx]
        ac = self._accept_counts[proposal_idx]
        accept_rate = ac / tc if tc > 0 else self.target_accept

        if self.mode == "sampling":
            step_size = self.gamma()
        else:
            step_size = 0.01  # Fixed learning rate for optimization

        # Update sigma: increase if accept rate too high, decrease if too low
        log_sigma = self._sigmas[proposal_idx]
        adjustment = step_size * (accept_rate - self.target_accept)
        new_sigma = log_sigma * (1.0 + adjustment)

        if self.mode == "sampling":
            new_sigma = max(self.sigma_min, min(self.sigma_max, new_sigma))

        self._sigmas[proposal_idx] = new_sigma

        # Update weights based on acceptance rate
        # Higher acceptance → higher weight
        self._weights[proposal_idx] *= 1.0 + step_size * (accept_rate - self.target_accept)

        # Normalize weights
        total = sum(self._weights)
        if total > 0:
            self._weights = [w / total for w in self._weights]

    def get_weights(self) -> list[float]:
        """Get current normalized weights (with containment in Mode A).

        In sampling mode, ensures p_j >= epsilon for all j by projecting
        onto the simplex with minimum-weight constraints.

        Returns:
            List of proposal weights summing to 1.
        """
        if self.mode == "sampling":
            n = self.n_proposals
            # Ensure minimum epsilon for each proposal
            # Distribute remaining weight (1 - n*epsilon) proportionally
            floor = self.epsilon
            remaining = 1.0 - n * floor
            if remaining <= 0:
                # epsilon too large, just use uniform
                return [1.0 / n] * n

            total = sum(self._weights)
            if total <= 0:
                return [1.0 / n] * n

            proportional = [w / total * remaining for w in self._weights]
            return [floor + p for p in proportional]
        return list(self._weights)

    def get_sigma(self, proposal_idx: int) -> float:
        """Get current scale for proposal (bounded in Mode A).

        Args:
            proposal_idx: Index of the proposal.

        Returns:
            Current scale value.
        """
        sigma = self._sigmas[proposal_idx]
        if self.mode == "sampling":
            return max(self.sigma_min, min(self.sigma_max, sigma))
        return sigma
