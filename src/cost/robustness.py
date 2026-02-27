"""Heuristic robustness proxy based on CVaR of delay distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import Transfer, TransferSet


@dataclass(frozen=True)
class DelayDistribution:
    """Exponential delay distribution parameters for CVaR calculation.

    Models delays as an exponential distribution with rate parameter λ.
    Provides closed-form expressions for VaR and CVaR.

    Args:
        rate: λ parameter (1/mean_delay). Higher rate = smaller delays.
        alpha: Confidence level for CVaR calculation (default 0.95).
    """

    rate: float
    alpha: float = 0.95

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"rate must be positive, got {self.rate}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    @property
    def mean(self) -> float:
        """Mean delay (1/λ)."""
        return 1.0 / self.rate

    @property
    def var_alpha(self) -> float:
        """Value at Risk at confidence α.

        For exponential distribution: VaR_α = -ln(1-α) / λ
        """
        return -math.log(1 - self.alpha) / self.rate

    @property
    def cvar_alpha(self) -> float:
        """Conditional Value at Risk at confidence α.

        For exponential distribution: CVaR_α = (-ln(1-α) + 1) / λ

        This is the expected value of delays exceeding VaR_α.
        """
        return (-math.log(1 - self.alpha) + 1) / self.rate


# Default delay distribution (mean delay = 3 minutes)
DEFAULT_DELAY_DISTRIBUTION = DelayDistribution(rate=1.0 / 3.0, alpha=0.95)


@dataclass
class RobustnessCost:
    """Heuristic robustness proxy based on CVaR of delay distributions.

    This is a heuristic proxy, NOT formal CVaR minimization of passenger loss.
    Measures the risk of missed connections due to delays.
    The cost penalizes transfers where the buffer time (waiting - walk)
    is insufficient to absorb the CVaR of delay distribution.

    Cost formula per transfer:
        Cost_ROB(γ) = ω_γ * max(0, CVaR_α - (w_γ - ρ_γ))

    where:
        - ω_γ is the transfer weight
        - CVaR_α is the conditional value at risk of delays
        - w_γ is the waiting time
        - ρ_γ is the walk time (minimum required connection time)

    Args:
        transfers: Set of transfers to evaluate.
        delay_distribution: Delay distribution for CVaR calculation.
    """

    transfers: TransferSet
    delay_distribution: DelayDistribution

    def compute_buffer(
        self,
        transfer: Transfer,
        schedule: Schedule,
        network: Network,
    ) -> float:
        """Compute buffer time for a transfer.

        Buffer = waiting_time - walk_time
        This is the time available to absorb delays.

        Args:
            transfer: The transfer to evaluate.
            schedule: Current schedule with line offsets.
            network: Network containing line definitions.

        Returns:
            Buffer time in minutes.
        """
        waiting_time = transfer.compute_waiting_time(schedule, network)
        return waiting_time - transfer.walk_time

    def compute_single_robustness_cost(
        self,
        transfer: Transfer,
        schedule: Schedule,
        network: Network,
    ) -> float:
        """Heuristic penalty for insufficient buffer against delays.

        Args:
            transfer: The transfer to evaluate.
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Weighted robustness cost for this transfer.
        """
        buffer = self.compute_buffer(transfer, schedule, network)
        cvar = self.delay_distribution.cvar_alpha

        # Shortfall: how much buffer is lacking compared to CVaR
        shortfall = max(0.0, cvar - buffer)

        return transfer.weight * shortfall

    def __call__(self, schedule: Schedule, network: Network) -> float:
        """Compute total robustness cost for a schedule.

        Args:
            schedule: Current schedule with line offsets.
            network: Network containing line definitions.

        Returns:
            Total weighted robustness cost across all transfers.
        """
        total = 0.0
        for transfer in self.transfers:
            total += self.compute_single_robustness_cost(transfer, schedule, network)
        return total

    def delta(
        self,
        old_schedule: Schedule,
        new_schedule: Schedule,
        changed_lines: set[str],
        network: Network,
    ) -> float:
        """Compute incremental robustness cost change.

        Only recomputes cost for transfers involving changed lines.

        Args:
            old_schedule: Previous schedule.
            new_schedule: New schedule after move.
            changed_lines: Set of line IDs that were modified.
            network: Network definition.

        Returns:
            Cost difference (new_cost - old_cost).
        """
        affected = self.transfers.get_transfers_involving_lines(changed_lines)

        old_cost = 0.0
        new_cost = 0.0

        for transfer in affected:
            old_cost += self.compute_single_robustness_cost(
                transfer, old_schedule, network
            )
            new_cost += self.compute_single_robustness_cost(
                transfer, new_schedule, network
            )

        return new_cost - old_cost

    def get_robustness_details(
        self,
        schedule: Schedule,
        network: Network,
    ) -> list[dict]:
        """Get detailed robustness information for each transfer.

        Useful for debugging and analysis.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            List of dicts with robustness details.
        """
        cvar = self.delay_distribution.cvar_alpha
        details = []

        for transfer in self.transfers:
            buffer = self.compute_buffer(transfer, schedule, network)
            cost = self.compute_single_robustness_cost(transfer, schedule, network)
            shortfall = max(0.0, cvar - buffer)

            details.append({
                "id": transfer.id,
                "from_line": transfer.from_line,
                "to_line": transfer.to_line,
                "buffer": buffer,
                "cvar": cvar,
                "shortfall": shortfall,
                "cost": cost,
                "weight": transfer.weight,
                "at_risk": buffer < cvar,
            })

        return details

    def count_at_risk_transfers(
        self,
        schedule: Schedule,
        network: Network,
    ) -> int:
        """Count transfers with insufficient buffer for CVaR.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Number of transfers with buffer < CVaR.
        """
        cvar = self.delay_distribution.cvar_alpha
        count = 0

        for transfer in self.transfers:
            buffer = self.compute_buffer(transfer, schedule, network)
            if buffer < cvar:
                count += 1

        return count
