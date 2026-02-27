"""Operational cost functions for turnaround and fleet constraints."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.cost.transfer_cost import barrier_cost

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnaroundConstraint:
    """A turnaround constraint between line end and line start.

    Represents the minimum time required for a vehicle to turn around
    at a terminal and start a new trip.

    Args:
        id: Unique identifier for this constraint.
        end_line: Line whose last station is the turnaround point.
        end_station_idx: Index of the last station on end_line.
        start_line: Line whose first station starts from same point.
        start_station_idx: Index of first station on start_line (usually 0).
        min_turnaround: Minimum turnaround time in minutes.
        M: Barrier coefficient for violation penalty (critical constraint).
    """

    id: str
    end_line: str
    end_station_idx: int
    start_line: str
    start_station_idx: int
    min_turnaround: float
    M: float = 1e7  # Critical constraint - higher penalty than transfers


@dataclass
class TurnaroundSet:
    """Collection of turnaround constraints with efficient lookup.

    Args:
        constraints: Dictionary mapping constraint ID to TurnaroundConstraint.
    """

    constraints: dict[str, TurnaroundConstraint] = field(default_factory=dict)

    # Lazy-built indices
    _by_end_line: dict[str, list[TurnaroundConstraint]] | None = field(
        default=None, init=False, repr=False
    )
    _by_start_line: dict[str, list[TurnaroundConstraint]] | None = field(
        default=None, init=False, repr=False
    )

    def _build_indices(self) -> None:
        """Build lookup indices by line."""
        self._by_end_line = {}
        self._by_start_line = {}

        for constraint in self.constraints.values():
            # Index by end_line
            if constraint.end_line not in self._by_end_line:
                self._by_end_line[constraint.end_line] = []
            self._by_end_line[constraint.end_line].append(constraint)

            # Index by start_line
            if constraint.start_line not in self._by_start_line:
                self._by_start_line[constraint.start_line] = []
            self._by_start_line[constraint.start_line].append(constraint)

    def get_constraints_involving_line(
        self, line_id: str
    ) -> list[TurnaroundConstraint]:
        """Get all constraints involving a specific line.

        Args:
            line_id: The line ID to search for.

        Returns:
            List of constraints where the line is either end_line or start_line.
        """
        if self._by_end_line is None:
            self._build_indices()

        result = []
        if self._by_end_line and line_id in self._by_end_line:
            result.extend(self._by_end_line[line_id])
        if self._by_start_line and line_id in self._by_start_line:
            for c in self._by_start_line[line_id]:
                if c not in result:
                    result.append(c)
        return result

    def get_constraints_involving_lines(
        self, line_ids: set[str]
    ) -> list[TurnaroundConstraint]:
        """Get all constraints involving any of the specified lines.

        Args:
            line_ids: Set of line IDs to search for.

        Returns:
            List of constraints involving any of the lines.
        """
        result = []
        seen_ids: set[str] = set()

        for line_id in line_ids:
            for constraint in self.get_constraints_involving_line(line_id):
                if constraint.id not in seen_ids:
                    seen_ids.add(constraint.id)
                    result.append(constraint)

        return result

    def __iter__(self):
        """Iterate over constraints."""
        return iter(self.constraints.values())

    def __len__(self) -> int:
        """Return number of constraints."""
        return len(self.constraints)


@dataclass
class TurnaroundCost:
    """Cost function for turnaround constraints.

    Computes barrier penalties when turnaround times are too short.

    Args:
        turnarounds: Set of turnaround constraints.
    """

    turnarounds: TurnaroundSet

    def compute_turnaround_time(
        self,
        constraint: TurnaroundConstraint,
        schedule: "Schedule",
        network: "Network",
    ) -> float:
        """Compute available turnaround time for a constraint.

        The turnaround time is the gap between when the end_line arrives
        at the terminal and when the start_line departs.

        Args:
            constraint: The turnaround constraint.
            schedule: Current schedule with offsets.
            network: Network containing line definitions.

        Returns:
            Available turnaround time in minutes.
        """
        end_line = network.get_line(constraint.end_line)
        start_line = network.get_line(constraint.start_line)

        # Time when end_line arrives at terminal
        end_offset = schedule.get_offset(constraint.end_line)
        arrival_time = end_offset + end_line.arrival_offset(constraint.end_station_idx)

        # Time when start_line departs from terminal
        start_offset = schedule.get_offset(constraint.start_line)
        departure_time = start_offset + start_line.departure_offset(
            constraint.start_station_idx
        )

        # Turnaround time repeats every start_line.frequency minutes
        period = start_line.frequency
        turnaround = (departure_time - arrival_time) % period

        return float(turnaround)

    def compute_single_turnaround_cost(
        self,
        constraint: TurnaroundConstraint,
        schedule: "Schedule",
        network: "Network",
    ) -> float:
        """Compute cost for a single turnaround constraint.

        Args:
            constraint: The turnaround constraint.
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Barrier cost if violated, 0 otherwise.
        """
        turnaround_time = self.compute_turnaround_time(constraint, schedule, network)
        violation = constraint.min_turnaround - turnaround_time

        return barrier_cost(violation, constraint.M)

    def __call__(self, schedule: "Schedule", network: "Network") -> float:
        """Compute total turnaround cost.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Total barrier cost across all turnaround constraints.
        """
        total = 0.0
        for constraint in self.turnarounds:
            total += self.compute_single_turnaround_cost(constraint, schedule, network)
        return total

    def delta(
        self,
        old_schedule: "Schedule",
        new_schedule: "Schedule",
        changed_lines: set[str],
        network: "Network",
    ) -> float:
        """Compute incremental turnaround cost change.

        Only recomputes cost for constraints involving changed lines.

        Args:
            old_schedule: Previous schedule.
            new_schedule: New schedule after move.
            changed_lines: Set of line IDs that were modified.
            network: Network definition.

        Returns:
            Cost difference (new_cost - old_cost).
        """
        affected = self.turnarounds.get_constraints_involving_lines(changed_lines)

        old_cost = 0.0
        new_cost = 0.0

        for constraint in affected:
            old_cost += self.compute_single_turnaround_cost(
                constraint, old_schedule, network
            )
            new_cost += self.compute_single_turnaround_cost(
                constraint, new_schedule, network
            )

        return new_cost - old_cost

    def count_violations(self, schedule: "Schedule", network: "Network") -> int:
        """Count number of violated turnaround constraints.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Number of constraints with turnaround < min_turnaround.
        """
        count = 0
        for constraint in self.turnarounds:
            turnaround_time = self.compute_turnaround_time(constraint, schedule, network)
            if turnaround_time < constraint.min_turnaround:
                count += 1
        return count

    def get_turnaround_details(
        self,
        schedule: "Schedule",
        network: "Network",
    ) -> list[dict]:
        """Get detailed information about each turnaround.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            List of dicts with turnaround details.
        """
        details = []
        for constraint in self.turnarounds:
            turnaround_time = self.compute_turnaround_time(
                constraint, schedule, network
            )
            cost = self.compute_single_turnaround_cost(constraint, schedule, network)
            details.append({
                "id": constraint.id,
                "end_line": constraint.end_line,
                "start_line": constraint.start_line,
                "turnaround_time": turnaround_time,
                "min_turnaround": constraint.min_turnaround,
                "cost": cost,
                "violated": turnaround_time < constraint.min_turnaround,
            })
        return details


@dataclass
class FleetCost:
    """Average intensity proxy for fleet size estimation via Little's Law.

    NOT a hard constraint on actual fleet. N_conc^peak is reported
    separately as a diagnostic metric.

    Args:
        available_fleet: Maximum available fleet size.
        lambda_fleet: Penalty coefficient for over-capacity.
    """

    available_fleet: int
    lambda_fleet: float = 0.5

    def compute_line_cycle_time(self, line: "Line", frequency: int) -> float:
        """Compute cycle time for a single line.

        Cycle time = sum of travel times + sum of dwell times.

        Args:
            line: Line definition.
            frequency: Service frequency in minutes.

        Returns:
            Total cycle time in minutes.
        """
        travel_sum = sum(line.travel_times)
        dwell_sum = sum(line.dwell_times)
        return float(travel_sum + dwell_sum)

    def compute_fleet_size(
        self,
        schedule: "Schedule",
        network: "Network",
    ) -> float:
        """Estimate fleet size needed for schedule.

        N_line = ceil(cycle_time / frequency) vehicles per line.
        This is an average intensity proxy (Little's Law), not peak concurrency.

        Args:
            schedule: Current schedule (not used in this simple model).
            network: Network definition.

        Returns:
            Estimated number of vehicles needed.
        """
        total_vehicles = 0.0

        for line in network.lines.values():
            freq = line.frequency
            cycle_time = self.compute_line_cycle_time(line, freq)
            vehicles_for_line = (cycle_time + freq - 1) / freq
            total_vehicles += vehicles_for_line

        return total_vehicles

    def __call__(self, schedule: "Schedule", network: "Network") -> float:
        """Compute fleet penalty cost.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Quadratic penalty if fleet exceeds capacity, 0 otherwise.
        """
        fleet_size = self.compute_fleet_size(schedule, network)
        excess = fleet_size - self.available_fleet

        if excess <= 0:
            return 0.0

        return self.lambda_fleet * excess * excess

    def delta(
        self,
        old_schedule: "Schedule",
        new_schedule: "Schedule",
        changed_lines: set[str],
        network: "Network",
    ) -> float:
        """Compute incremental fleet cost change.

        Note: In this simplified model, fleet size doesn't depend on
        schedule offsets, only on line structure. So delta is 0.

        Args:
            old_schedule: Previous schedule.
            new_schedule: New schedule after move.
            changed_lines: Set of line IDs that were modified.
            network: Network definition.

        Returns:
            Cost difference (typically 0 in this model).
        """
        # Fleet size depends on line structure, not offsets
        # For the simple model, this is always 0
        return 0.0


@dataclass
class OperationalCost:
    """Combined operational cost: turnarounds + fleet.

    Args:
        turnaround_cost: Turnaround cost component.
        fleet_cost: Fleet cost component (optional).
    """

    turnaround_cost: TurnaroundCost | None = None
    fleet_cost: FleetCost | None = None

    def __call__(self, schedule: "Schedule", network: "Network") -> float:
        """Compute total operational cost.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Sum of turnaround and fleet costs.
        """
        total = 0.0

        if self.turnaround_cost is not None:
            total += self.turnaround_cost(schedule, network)

        if self.fleet_cost is not None:
            total += self.fleet_cost(schedule, network)

        return total

    def delta(
        self,
        old_schedule: "Schedule",
        new_schedule: "Schedule",
        changed_lines: set[str],
        network: "Network",
    ) -> float:
        """Compute incremental operational cost change.

        Args:
            old_schedule: Previous schedule.
            new_schedule: New schedule after move.
            changed_lines: Set of line IDs that were modified.
            network: Network definition.

        Returns:
            Cost difference (new_cost - old_cost).
        """
        delta = 0.0

        if self.turnaround_cost is not None:
            delta += self.turnaround_cost.delta(
                old_schedule, new_schedule, changed_lines, network
            )

        if self.fleet_cost is not None:
            delta += self.fleet_cost.delta(
                old_schedule, new_schedule, changed_lines, network
            )

        return delta

    def count_violations(self, schedule: "Schedule", network: "Network") -> int:
        """Count total operational constraint violations.

        Args:
            schedule: Current schedule.
            network: Network definition.

        Returns:
            Number of violated constraints.
        """
        count = 0

        if self.turnaround_cost is not None:
            count += self.turnaround_cost.count_violations(schedule, network)

        return count
