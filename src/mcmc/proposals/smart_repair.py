"""Smart repair move proposal for targeted constraint improvement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from numpy.random import Generator

    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import Transfer, TransferSet


@dataclass
class SmartRepairMove:
    """Targeted repair move that biases toward ideal waiting time.

    This proposal selects a random transfer, then shifts one of its
    lines to bring the waiting time closer to the ideal value.

    The move is ASYMMETRIC because the shift is biased toward the repair
    point (δ*), so the Hastings ratio must be computed properly:

        H = φ(δ - δ*_forward) / φ(δ' - δ*_backward)

    where φ is the Gaussian density and δ' = -δ is the reverse move.

    Algorithm:
        1. Select transfer γ uniformly at random
        2. Select line L from {from_line, to_line} uniformly
        3. Compute δ* = shift that would give w = w_ideal
        4. Sample δ = δ* + ε, where ε ~ N(0, σ²)
        5. Compute Hastings ratio for detailed balance

    Args:
        transfers: Set of transfers to consider for repair.
        sigma: Standard deviation for perturbation around repair point.
        rng: NumPy random generator.
    """

    transfers: TransferSet
    sigma: float = 2.0
    rng: Generator = field(default_factory=lambda: np.random.default_rng())

    @property
    def name(self) -> str:
        return "smart_repair"

    def _compute_repair_delta(
        self,
        transfer: Transfer,
        schedule: Schedule,
        network: Network,
        target_line: str,
    ) -> float:
        """Compute δ* using minimal norm rule: |δ*| <= freq/2.

        The effect of shifting a line depends on whether it's the
        arriving (from) or departing (to) line:
        - Shifting from_line by +δ moves arrival later → decreases wait
        - Shifting to_line by +δ moves departure later → increases wait

        The base delta is then reduced mod frequency to minimize |δ*|.

        Args:
            transfer: The transfer to repair.
            schedule: Current schedule.
            network: Network definition.
            target_line: Which line to shift.

        Returns:
            The repair delta δ* with minimal norm.
        """
        current_wait = transfer.compute_waiting_time(schedule, network)
        target_wait = transfer.params.w_ideal
        diff = target_wait - current_wait  # How much we want to increase wait

        if target_line == transfer.from_line:
            base_delta = -diff
        else:
            base_delta = diff

        # Apply minimal norm: reduce mod frequency to [-freq/2, freq/2)
        to_line_obj = network.get_line(transfer.to_line)
        freq = to_line_obj.frequency
        delta_star = base_delta % freq
        if delta_star > freq / 2:
            delta_star -= freq
        return delta_star

    def _compute_discrete_mass(self, k: int, delta_star: float) -> float:
        """Compute probability mass for integer shift k under discrete normal.

        For a continuous Gaussian N(δ*, σ²), the probability mass of integer k
        is obtained by integrating over [k - 0.5, k + 0.5):

            q_k = Φ((k + 0.5 - δ*)/σ) - Φ((k - 0.5 - δ*)/σ)

        where Φ is the standard normal CDF.

        Args:
            k: Integer shift value
            delta_star: Center of the Gaussian (repair point)

        Returns:
            Probability mass in [0, 1] (clipped for numerical safety)
        """
        lower = (k - 0.5 - delta_star) / self.sigma
        upper = (k + 0.5 - delta_star) / self.sigma

        mass = norm.cdf(upper) - norm.cdf(lower)

        # Clip to avoid numerical issues
        return max(1e-300, min(1.0, mass))

    def propose(
        self,
        schedule: Schedule,
        network: Network,
    ) -> tuple[Schedule, float, set[str]]:
        """Propose a smart repair move.

        Args:
            schedule: Current schedule state.
            network: Network definition.

        Returns:
            Tuple of (new_schedule, hastings_ratio, changed_lines).
        """
        # Get list of transfers
        transfer_list = list(self.transfers)
        if not transfer_list:
            return schedule.copy(), 1.0, set()

        # Select random transfer
        transfer_idx = self.rng.integers(0, len(transfer_list))
        transfer = transfer_list[transfer_idx]

        # Select which line to modify
        from_line = transfer.from_line
        to_line = transfer.to_line

        if from_line == to_line:
            # Self-loop (unusual but possible)
            target_line = from_line
        else:
            # Choose uniformly between the two lines
            if self.rng.random() < 0.5:
                target_line = from_line
            else:
                target_line = to_line

        # Check if target line exists in network
        if target_line not in network.lines:
            return schedule.copy(), 1.0, set()

        # Compute repair delta (δ*)
        delta_star_forward = self._compute_repair_delta(
            transfer, schedule, network, target_line
        )

        # Add Gaussian noise: δ = δ* + ε
        epsilon = self.rng.normal(0, self.sigma)
        delta = delta_star_forward + epsilon
        delta_int = int(round(delta))

        # Apply shift
        new_schedule = schedule.copy()
        new_schedule.shift_line(target_line, delta_int)

        # Compute Hastings ratio using discrete probability masses
        # Forward: we sampled delta_int from q_{delta_int}(·|δ*_forward)
        # Backward: reverse move would sample -delta_int from q_{-delta_int}(·|δ*_backward)
        # H = q_backward / q_forward

        delta_star_backward = self._compute_repair_delta(
            transfer, new_schedule, network, target_line
        )

        forward_mass = self._compute_discrete_mass(delta_int, delta_star_forward)
        backward_mass = self._compute_discrete_mass(-delta_int, delta_star_backward)

        if forward_mass > 0:
            hastings_ratio = backward_mass / forward_mass
        else:
            # Numerical safety: if forward mass is zero, reject
            hastings_ratio = 0.0

        # Clamp to prevent extreme values
        hastings_ratio = max(1e-10, min(1e10, hastings_ratio))

        return new_schedule, hastings_ratio, {target_line}

    def compute_repair_info(
        self,
        transfer: Transfer,
        schedule: Schedule,
        network: Network,
    ) -> dict:
        """Get diagnostic information about potential repair.

        Useful for understanding what repair would do.

        Args:
            transfer: The transfer to analyze.
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Dictionary with repair analysis.
        """
        current_wait = transfer.compute_waiting_time(schedule, network)
        ideal_wait = transfer.params.w_ideal
        min_wait = transfer.params.w_min

        delta_from = self._compute_repair_delta(
            transfer, schedule, network, transfer.from_line
        )
        delta_to = self._compute_repair_delta(
            transfer, schedule, network, transfer.to_line
        )

        return {
            "transfer_id": transfer.id,
            "current_wait": current_wait,
            "ideal_wait": ideal_wait,
            "min_wait": min_wait,
            "is_violated": current_wait < min_wait,
            "gap_to_ideal": ideal_wait - current_wait,
            "repair_delta_from_line": delta_from,
            "repair_delta_to_line": delta_to,
        }


def create_smart_repair(
    transfers: TransferSet,
    sigma: float = 2.0,
    rng: Generator | None = None,
) -> SmartRepairMove:
    """Factory function to create SmartRepairMove.

    Args:
        transfers: Transfer set for repair candidates.
        sigma: Standard deviation for perturbation.
        rng: Random generator (None = new default).

    Returns:
        Configured SmartRepairMove instance.
    """
    if rng is None:
        rng = np.random.default_rng()

    return SmartRepairMove(
        transfers=transfers,
        sigma=sigma,
        rng=rng,
    )
