"""Transfer cost function with soft constraints and barriers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import Transfer, TransferParams, TransferSet


logger = logging.getLogger(__name__)


class CostFunction(Protocol):
    """Protocol for cost functions."""

    def __call__(self, schedule: Schedule, network: Network) -> float:
        """Compute total cost for a schedule."""
        ...

    def delta(
        self,
        old_schedule: Schedule,
        new_schedule: Schedule,
        changed_lines: set[str],
        network: Network,
    ) -> float:
        """Compute cost difference between schedules.

        For efficiency, only considers transfers involving changed lines.
        """
        ...


def barrier_cost(violation: float, M: float) -> float:
    """Compute quadratic barrier cost for constraint violation.

    Args:
        violation: Amount of violation (positive means violated).
        M: Barrier coefficient.

    Returns:
        Quadratic penalty M * violation^2 if violation > 0, else 0.
    """
    if violation <= 0:
        return 0.0
    return M * violation * violation


def soft_transfer_cost(waiting_time: float, params: TransferParams) -> float:
    """Compute soft cost for a single transfer.

    The cost function is piecewise:
    - w < w_min: Barrier penalty M * (w_min - w)^2
    - w_min <= w < w_ideal: Quadratic approach alpha * (w_ideal - w)^2
    - w >= w_ideal: Quadratic departure beta * (w - w_ideal)^2

    Args:
        waiting_time: Actual waiting time in minutes.
        params: Transfer parameters defining the cost shape.

    Returns:
        Cost value (non-negative).
    """
    w = waiting_time
    w_min = params.w_min
    w_ideal = params.w_ideal

    if w < w_min:
        # Barrier region - constraint violation
        violation = w_min - w
        return barrier_cost(violation, params.M)
    elif w < w_ideal:
        # Below ideal - penalize with alpha
        diff = w_ideal - w
        return params.alpha * diff * diff
    else:
        # Above ideal - penalize with beta
        diff = w - w_ideal
        return params.beta * diff * diff


@dataclass
class TransferCost:
    """Cost function for transfer quality.

    Computes the weighted sum of soft transfer costs across all transfers.

    Args:
        transfers: Set of transfers to evaluate.
    """

    transfers: TransferSet

    def compute_single_transfer_cost(
        self,
        transfer: Transfer,
        schedule: Schedule,
        network: Network,
    ) -> float:
        """Compute cost for a single transfer.

        Args:
            transfer: The transfer to evaluate.
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Weighted cost for this transfer.
        """
        waiting_time = transfer.compute_waiting_time(schedule, network)
        raw_cost = soft_transfer_cost(waiting_time, transfer.params)
        return transfer.weight * raw_cost

    def __call__(self, schedule: Schedule, network: Network) -> float:
        """Compute total transfer cost for a schedule.

        Args:
            schedule: Current schedule with line offsets.
            network: Network containing line definitions.

        Returns:
            Total weighted cost across all transfers.
        """
        total = 0.0
        for transfer in self.transfers:
            total += self.compute_single_transfer_cost(transfer, schedule, network)
        return total

    def delta(
        self,
        old_schedule: Schedule,
        new_schedule: Schedule,
        changed_lines: set[str],
        network: Network,
    ) -> float:
        """Compute incremental cost change.

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
            old_cost += self.compute_single_transfer_cost(transfer, old_schedule, network)
            new_cost += self.compute_single_transfer_cost(transfer, new_schedule, network)

        return new_cost - old_cost

    def get_transfer_details(
        self,
        schedule: Schedule,
        network: Network,
    ) -> list[dict]:
        """Get detailed information about each transfer.

        Useful for debugging and analysis.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            List of dicts with transfer details.
        """
        details = []
        for transfer in self.transfers:
            waiting_time = transfer.compute_waiting_time(schedule, network)
            cost = self.compute_single_transfer_cost(transfer, schedule, network)
            details.append({
                "id": transfer.id,
                "from_line": transfer.from_line,
                "to_line": transfer.to_line,
                "waiting_time": waiting_time,
                "cost": cost,
                "weight": transfer.weight,
                "w_min": transfer.params.w_min,
                "w_ideal": transfer.params.w_ideal,
                "violated": waiting_time < transfer.params.w_min,
            })
        return details

    def count_violations(self, schedule: Schedule, network: Network) -> int:
        """Count number of transfers violating minimum waiting time.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Number of violated transfers.
        """
        count = 0
        for transfer in self.transfers:
            waiting_time = transfer.compute_waiting_time(schedule, network)
            if waiting_time < transfer.params.w_min:
                count += 1
        return count

    def get_worst_transfer(
        self,
        schedule: Schedule,
        network: Network,
    ) -> tuple[Transfer | None, float]:
        """Find the transfer with highest cost.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Tuple of (worst transfer, its cost) or (None, 0) if no transfers.
        """
        worst_transfer = None
        worst_cost = 0.0

        for transfer in self.transfers:
            cost = self.compute_single_transfer_cost(transfer, schedule, network)
            if cost > worst_cost:
                worst_cost = cost
                worst_transfer = transfer

        return worst_transfer, worst_cost
