"""Tests for operational cost functions."""

from __future__ import annotations

import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.cost.operational import (
    FleetCost,
    OperationalCost,
    TurnaroundConstraint,
    TurnaroundCost,
    TurnaroundSet,
)


@pytest.fixture
def simple_network() -> Network:
    """Create a simple network with two lines."""
    network = Network()

    # Line 1: A -> B -> C
    stations_l1 = [
        Station(id="A", location="A", modality="bus"),
        Station(id="B", location="B", modality="bus"),
        Station(id="C", location="C", modality="bus"),
    ]
    line1 = Line(
        id="L1",
        modality="bus",
        stations=stations_l1,
        travel_times=[10, 15],  # A->B: 10min, B->C: 15min
        dwell_times=[2, 2, 2],
        frequency=60,
    )
    network.add_line(line1)

    # Line 2: C -> D -> E
    stations_l2 = [
        Station(id="C2", location="C", modality="bus"),
        Station(id="D", location="D", modality="bus"),
        Station(id="E", location="E", modality="bus"),
    ]
    line2 = Line(
        id="L2",
        modality="bus",
        stations=stations_l2,
        travel_times=[12, 8],
        dwell_times=[2, 2, 2],
        frequency=60,
    )
    network.add_line(line2)

    return network


@pytest.fixture
def turnaround_constraint() -> TurnaroundConstraint:
    """Create a turnaround constraint: L1 ends, L2 starts at same terminal."""
    return TurnaroundConstraint(
        id="T1",
        end_line="L1",
        end_station_idx=2,  # C is the terminal of L1
        start_line="L2",
        start_station_idx=0,  # C is the start of L2
        min_turnaround=5.0,
        M=1e7,
    )


@pytest.fixture
def turnaround_set(turnaround_constraint: TurnaroundConstraint) -> TurnaroundSet:
    """Create a TurnaroundSet with one constraint."""
    return TurnaroundSet(constraints={turnaround_constraint.id: turnaround_constraint})


class TestTurnaroundConstraint:
    """Tests for TurnaroundConstraint dataclass."""

    def test_creation(self) -> None:
        constraint = TurnaroundConstraint(
            id="T1",
            end_line="L1",
            end_station_idx=2,
            start_line="L2",
            start_station_idx=0,
            min_turnaround=5.0,
        )

        assert constraint.id == "T1"
        assert constraint.end_line == "L1"
        assert constraint.end_station_idx == 2
        assert constraint.start_line == "L2"
        assert constraint.start_station_idx == 0
        assert constraint.min_turnaround == 5.0
        assert constraint.M == 1e7  # Default

    def test_is_frozen(self) -> None:
        constraint = TurnaroundConstraint(
            id="T1",
            end_line="L1",
            end_station_idx=2,
            start_line="L2",
            start_station_idx=0,
            min_turnaround=5.0,
        )
        with pytest.raises(AttributeError):
            constraint.id = "T2"  # type: ignore


class TestTurnaroundSet:
    """Tests for TurnaroundSet."""

    def test_iteration(self, turnaround_set: TurnaroundSet) -> None:
        constraints = list(turnaround_set)
        assert len(constraints) == 1
        assert constraints[0].id == "T1"

    def test_len(self, turnaround_set: TurnaroundSet) -> None:
        assert len(turnaround_set) == 1

    def test_get_constraints_involving_line(
        self, turnaround_set: TurnaroundSet
    ) -> None:
        # L1 is the end_line
        l1_constraints = turnaround_set.get_constraints_involving_line("L1")
        assert len(l1_constraints) == 1
        assert l1_constraints[0].id == "T1"

        # L2 is the start_line
        l2_constraints = turnaround_set.get_constraints_involving_line("L2")
        assert len(l2_constraints) == 1
        assert l2_constraints[0].id == "T1"

        # L3 is not involved
        l3_constraints = turnaround_set.get_constraints_involving_line("L3")
        assert len(l3_constraints) == 0

    def test_get_constraints_involving_lines(
        self, turnaround_set: TurnaroundSet
    ) -> None:
        constraints = turnaround_set.get_constraints_involving_lines({"L1", "L3"})
        assert len(constraints) == 1

        # Should not duplicate
        constraints = turnaround_set.get_constraints_involving_lines({"L1", "L2"})
        assert len(constraints) == 1


class TestTurnaroundCost:
    """Tests for TurnaroundCost."""

    def test_compute_turnaround_time(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        cost_fn = TurnaroundCost(turnarounds=turnaround_set)
        schedule = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)
        constraint = list(turnaround_set)[0]

        # L1 arrives at C (station 2) at: 0 + 10 + 2 + 15 = 27
        # L2 departs from C (station 0) at: 40
        # Turnaround = (40 - 27) % 60 = 13
        turnaround_time = cost_fn.compute_turnaround_time(
            constraint, schedule, simple_network
        )

        assert abs(turnaround_time - 13.0) < 0.01

    def test_no_violation_zero_cost(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        cost_fn = TurnaroundCost(turnarounds=turnaround_set)
        # With turnaround = 13 > min_turnaround = 5, no violation
        schedule = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)

        cost = cost_fn(schedule, simple_network)
        assert cost == 0.0

    def test_violation_barrier_cost(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        cost_fn = TurnaroundCost(turnarounds=turnaround_set)
        # Set L2 to depart just 2 minutes after L1 arrives
        # L1 arrives at 27, L2 departs at 29 -> turnaround = 2 < 5
        schedule = Schedule(offsets={"L1": 0, "L2": 29}, period=1440)

        constraint = list(turnaround_set)[0]
        turnaround_time = cost_fn.compute_turnaround_time(
            constraint, schedule, simple_network
        )

        # Should be about 2 minutes
        assert turnaround_time < constraint.min_turnaround

        cost = cost_fn(schedule, simple_network)
        # Violation = 5 - 2 = 3, cost = 1e7 * 3^2 = 9e7
        assert cost > 0.0
        assert cost > 1e6  # Should be a large penalty

    def test_delta_matches_full_computation(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        cost_fn = TurnaroundCost(turnarounds=turnaround_set)

        old_schedule = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)
        new_schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)

        old_cost = cost_fn(old_schedule, simple_network)
        new_cost = cost_fn(new_schedule, simple_network)
        expected_delta = new_cost - old_cost

        # Test delta when changing L2
        delta = cost_fn.delta(
            old_schedule, new_schedule, {"L2"}, simple_network
        )

        assert abs(delta - expected_delta) < 1e-10

    def test_delta_only_considers_affected(
        self,
        simple_network: Network,
    ) -> None:
        # Create two turnaround constraints
        constraints = {
            "T1": TurnaroundConstraint(
                id="T1",
                end_line="L1",
                end_station_idx=2,
                start_line="L2",
                start_station_idx=0,
                min_turnaround=5.0,
            ),
        }
        turnaround_set = TurnaroundSet(constraints=constraints)
        cost_fn = TurnaroundCost(turnarounds=turnaround_set)

        old_schedule = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)
        new_schedule = Schedule(offsets={"L1": 5, "L2": 40}, period=1440)

        # Only L1 changed, which affects T1
        delta = cost_fn.delta(
            old_schedule, new_schedule, {"L1"}, simple_network
        )

        # Verify against full computation
        old_cost = cost_fn(old_schedule, simple_network)
        new_cost = cost_fn(new_schedule, simple_network)

        assert abs(delta - (new_cost - old_cost)) < 1e-10

    def test_count_violations(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        cost_fn = TurnaroundCost(turnarounds=turnaround_set)

        # No violation
        schedule_ok = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)
        assert cost_fn.count_violations(schedule_ok, simple_network) == 0

        # Violation
        schedule_bad = Schedule(offsets={"L1": 0, "L2": 29}, period=1440)
        assert cost_fn.count_violations(schedule_bad, simple_network) == 1


class TestFleetCost:
    """Tests for FleetCost."""

    def test_compute_line_cycle_time(self, simple_network: Network) -> None:
        fleet_cost = FleetCost(available_fleet=10)
        line = simple_network.get_line("L1")

        # L1: travel = [10, 15], dwell = [2, 2, 2]
        # cycle = 10 + 15 + 2 + 2 + 2 = 31
        cycle_time = fleet_cost.compute_line_cycle_time(line, 60)
        assert cycle_time == 31.0

    def test_compute_fleet_size(self, simple_network: Network) -> None:
        fleet_cost = FleetCost(available_fleet=10)
        schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)

        fleet_size = fleet_cost.compute_fleet_size(schedule, simple_network)

        # L1: 31 min cycle -> needs 31/60 ≈ 0.52 vehicles per period
        # L2: 12 + 8 + 2 + 2 + 2 = 26 min -> 26/60 ≈ 0.43 vehicles
        # Total ≈ 0.95 vehicles
        assert fleet_size > 0
        assert fleet_size < 10  # Should be well under capacity

    def test_under_capacity_zero_cost(self, simple_network: Network) -> None:
        fleet_cost = FleetCost(available_fleet=10)
        schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)

        cost = fleet_cost(schedule, simple_network)
        assert cost == 0.0

    def test_over_capacity_penalty(self, simple_network: Network) -> None:
        # Set very low fleet capacity
        fleet_cost = FleetCost(available_fleet=0, lambda_fleet=1.0)
        schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)

        cost = fleet_cost(schedule, simple_network)

        # Should have positive cost since fleet > 0 but available = 0
        assert cost > 0.0

    def test_delta_is_zero_for_offset_changes(self, simple_network: Network) -> None:
        """Fleet cost doesn't depend on offsets in the simplified model."""
        fleet_cost = FleetCost(available_fleet=1)

        old_schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)
        new_schedule = Schedule(offsets={"L1": 10, "L2": 20}, period=1440)

        delta = fleet_cost.delta(
            old_schedule, new_schedule, {"L1", "L2"}, simple_network
        )

        assert delta == 0.0


class TestOperationalCost:
    """Tests for combined OperationalCost."""

    def test_combines_turnaround_and_fleet(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        turnaround_cost = TurnaroundCost(turnarounds=turnaround_set)
        fleet_cost = FleetCost(available_fleet=0, lambda_fleet=1.0)

        ops_cost = OperationalCost(
            turnaround_cost=turnaround_cost,
            fleet_cost=fleet_cost,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 29}, period=1440)

        total = ops_cost(schedule, simple_network)

        # Should include both turnaround violation and fleet penalty
        turn_only = turnaround_cost(schedule, simple_network)
        fleet_only = fleet_cost(schedule, simple_network)

        assert abs(total - (turn_only + fleet_only)) < 1e-10

    def test_turnaround_only(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        turnaround_cost = TurnaroundCost(turnarounds=turnaround_set)
        ops_cost = OperationalCost(turnaround_cost=turnaround_cost)

        schedule = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)
        cost = ops_cost(schedule, simple_network)

        assert cost == turnaround_cost(schedule, simple_network)

    def test_delta_matches_full(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        turnaround_cost = TurnaroundCost(turnarounds=turnaround_set)
        fleet_cost = FleetCost(available_fleet=1)
        ops_cost = OperationalCost(
            turnaround_cost=turnaround_cost,
            fleet_cost=fleet_cost,
        )

        old_schedule = Schedule(offsets={"L1": 0, "L2": 40}, period=1440)
        new_schedule = Schedule(offsets={"L1": 5, "L2": 40}, period=1440)

        old_cost = ops_cost(old_schedule, simple_network)
        new_cost = ops_cost(new_schedule, simple_network)
        expected_delta = new_cost - old_cost

        delta = ops_cost.delta(
            old_schedule, new_schedule, {"L1"}, simple_network
        )

        assert abs(delta - expected_delta) < 1e-10

    def test_count_violations(
        self,
        simple_network: Network,
        turnaround_set: TurnaroundSet,
    ) -> None:
        turnaround_cost = TurnaroundCost(turnarounds=turnaround_set)
        ops_cost = OperationalCost(turnaround_cost=turnaround_cost)

        schedule_bad = Schedule(offsets={"L1": 0, "L2": 29}, period=1440)
        assert ops_cost.count_violations(schedule_bad, simple_network) == 1
