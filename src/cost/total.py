"""Aggregated total cost function combining PAX, OPS, and ROB components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule

from src.cost.operational import OperationalCost
from src.cost.robustness import RobustnessCost
from src.cost.transfer_cost import TransferCost


@dataclass
class TotalCost:
    """Aggregated cost function: PAX + OPS + ROB.

    Combines three cost components:
    - PAX (passenger): Transfer quality cost
    - OPS (operational): Turnaround and fleet constraints
    - ROB (robustness): Heuristic robustness proxy

    The total cost is a weighted sum:
        Cost = pax_weight * Cost_PAX + ops_weight * Cost_OPS + rob_weight * Cost_ROB

    Args:
        transfer_cost: Transfer quality cost function (required).
        operational_cost: Operational constraints cost (optional).
        robustness_cost: CVaR robustness cost (optional).
        pax_weight: Weight for passenger cost component.
        ops_weight: Weight for operational cost component.
        rob_weight: Weight for robustness cost component.
    """

    transfer_cost: TransferCost
    operational_cost: OperationalCost | None = None
    robustness_cost: RobustnessCost | None = None
    pax_weight: float = 1.0
    ops_weight: float = 1.0
    rob_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.pax_weight < 0 or self.ops_weight < 0 or self.rob_weight < 0:
            raise ValueError("All weights must be non-negative")

    def __call__(self, schedule: Schedule, network: Network) -> float:
        """Compute total cost for a schedule.

        Args:
            schedule: Current schedule with line offsets.
            network: Network containing line definitions.

        Returns:
            Total weighted cost.
        """
        total = self.pax_weight * self.transfer_cost(schedule, network)

        if self.operational_cost is not None:
            total += self.ops_weight * self.operational_cost(schedule, network)

        if self.robustness_cost is not None:
            total += self.rob_weight * self.robustness_cost(schedule, network)

        return total

    def delta(
        self,
        old_schedule: Schedule,
        new_schedule: Schedule,
        changed_lines: set[str],
        network: Network,
    ) -> float:
        """Compute incremental cost change.

        CRITICAL: This method MUST return EXACTLY the same value as:
            self(new_schedule, network) - self(old_schedule, network)

        Only recomputes cost for affected constraints.

        Args:
            old_schedule: Previous schedule.
            new_schedule: New schedule after move.
            changed_lines: Set of line IDs that were modified.
            network: Network definition.

        Returns:
            Cost difference (new_cost - old_cost).
        """
        d = self.pax_weight * self.transfer_cost.delta(
            old_schedule, new_schedule, changed_lines, network
        )

        if self.operational_cost is not None:
            d += self.ops_weight * self.operational_cost.delta(
                old_schedule, new_schedule, changed_lines, network
            )

        if self.robustness_cost is not None:
            d += self.rob_weight * self.robustness_cost.delta(
                old_schedule, new_schedule, changed_lines, network
            )

        return d

    def decompose(
        self,
        schedule: Schedule,
        network: Network,
    ) -> dict[str, float]:
        """Return breakdown of costs by component.

        Useful for analysis and debugging.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Dictionary with 'pax', 'ops', 'rob', and 'total' costs.
        """
        pax = self.pax_weight * self.transfer_cost(schedule, network)

        ops = 0.0
        if self.operational_cost is not None:
            ops = self.ops_weight * self.operational_cost(schedule, network)

        rob = 0.0
        if self.robustness_cost is not None:
            rob = self.rob_weight * self.robustness_cost(schedule, network)

        return {
            "pax": pax,
            "ops": ops,
            "rob": rob,
            "total": pax + ops + rob,
        }

    def get_component_weights(self) -> dict[str, float]:
        """Return the current component weights.

        Returns:
            Dictionary with 'pax', 'ops', 'rob' weights.
        """
        return {
            "pax": self.pax_weight,
            "ops": self.ops_weight,
            "rob": self.rob_weight,
        }


@dataclass
class TotalCostConfig:
    """Configuration for creating TotalCost from parameters.

    Args:
        pax_weight: Weight for passenger cost.
        ops_weight: Weight for operational cost.
        rob_weight: Weight for robustness cost.
        delay_rate: Rate parameter for delay distribution (1/mean_delay).
        delay_alpha: Confidence level for CVaR.
        fleet_available: Available fleet size (optional).
        fleet_lambda: Fleet penalty coefficient (optional).
    """

    pax_weight: float = 1.0
    ops_weight: float = 1.0
    rob_weight: float = 0.1
    delay_rate: float = 1.0 / 3.0  # Mean delay = 3 min
    delay_alpha: float = 0.95
    fleet_available: int | None = None
    fleet_lambda: float = 0.5


def create_total_cost(
    transfer_cost: TransferCost,
    operational_cost: OperationalCost | None = None,
    robustness_cost: RobustnessCost | None = None,
    config: TotalCostConfig | None = None,
) -> TotalCost:
    """Factory function to create TotalCost with configuration.

    Args:
        transfer_cost: Transfer quality cost (required).
        operational_cost: Operational cost (optional).
        robustness_cost: Robustness cost (optional).
        config: Configuration object (uses defaults if None).

    Returns:
        Configured TotalCost instance.
    """
    if config is None:
        config = TotalCostConfig()

    return TotalCost(
        transfer_cost=transfer_cost,
        operational_cost=operational_cost,
        robustness_cost=robustness_cost,
        pax_weight=config.pax_weight,
        ops_weight=config.ops_weight,
        rob_weight=config.rob_weight,
    )


def create_total_cost_from_dict(
    transfer_cost: TransferCost,
    operational_cost: OperationalCost | None = None,
    robustness_cost: RobustnessCost | None = None,
    **kwargs: Any,
) -> TotalCost:
    """Create TotalCost from keyword arguments.

    Convenience function for configuration from YAML/JSON.

    Args:
        transfer_cost: Transfer quality cost (required).
        operational_cost: Operational cost (optional).
        robustness_cost: Robustness cost (optional).
        **kwargs: Configuration parameters (pax_weight, ops_weight, rob_weight).

    Returns:
        Configured TotalCost instance.
    """
    return TotalCost(
        transfer_cost=transfer_cost,
        operational_cost=operational_cost,
        robustness_cost=robustness_cost,
        pax_weight=kwargs.get("pax_weight", 1.0),
        ops_weight=kwargs.get("ops_weight", 1.0),
        rob_weight=kwargs.get("rob_weight", 0.1),
    )
