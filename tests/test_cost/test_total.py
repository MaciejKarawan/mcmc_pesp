"""Tests for total cost module."""

import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import Transfer, TransferParams, TransferSet
from src.cost.operational import (
    FleetCost,
    OperationalCost,
    TurnaroundConstraint,
    TurnaroundCost,
    TurnaroundSet,
)
from src.cost.robustness import DelayDistribution, RobustnessCost
from src.cost.total import TotalCost, TotalCostConfig, create_total_cost
from src.cost.transfer_cost import TransferCost


class TestTotalCost:
    """Tests for TotalCost class."""

    @pytest.fixture
    def simple_network(self) -> Network:
        """Create a simple network with two lines meeting at B."""
        network = Network()

        stations_l1 = [
            Station(id="A_L1", location="A", modality="bus"),
            Station(id="B_L1", location="B", modality="bus"),
        ]
        line1 = Line(
            id="L1",
            modality="bus",
            stations=stations_l1,
            travel_times=[15],
            dwell_times=[2, 2],
            frequency=60,
        )

        stations_l2 = [
            Station(id="B_L2", location="B", modality="bus"),
            Station(id="C_L2", location="C", modality="bus"),
        ]
        line2 = Line(
            id="L2",
            modality="bus",
            stations=stations_l2,
            travel_times=[20],
            dwell_times=[2, 2],
            frequency=60,
        )

        network.add_line(line1)
        network.add_line(line2)
        return network

    @pytest.fixture
    def transfer_set(self) -> TransferSet:
        """Create a TransferSet with one transfer."""
        params = TransferParams(
            w_min=3.0,
            w_ideal=8.0,
            w_max=30.0,
            alpha=2.0,
            beta=0.5,
            M=1e6,
        )
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
            weight=1.0,
        )
        return TransferSet(transfers={"T1": t})

    @pytest.fixture
    def transfer_cost(self, transfer_set: TransferSet) -> TransferCost:
        return TransferCost(transfers=transfer_set)

    @pytest.fixture
    def robustness_cost(self, transfer_set: TransferSet) -> RobustnessCost:
        dist = DelayDistribution(rate=1.0, alpha=0.95)
        return RobustnessCost(transfers=transfer_set, delay_distribution=dist)

    @pytest.fixture
    def operational_cost(self) -> OperationalCost:
        # Simple operational cost with no constraints
        return OperationalCost(turnaround_cost=None, fleet_cost=None)

    def test_pax_only(
        self,
        transfer_cost: TransferCost,
        simple_network: Network,
    ) -> None:
        """Test TotalCost with only PAX component."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            pax_weight=1.0,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        pax_only = transfer_cost(schedule, simple_network)
        total = total_cost(schedule, simple_network)

        assert total == pax_only

    def test_aggregates_components(
        self,
        transfer_cost: TransferCost,
        robustness_cost: RobustnessCost,
        operational_cost: OperationalCost,
        simple_network: Network,
    ) -> None:
        """Test that TotalCost correctly aggregates all components."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            operational_cost=operational_cost,
            robustness_cost=robustness_cost,
            pax_weight=2.0,
            ops_weight=1.5,
            rob_weight=0.5,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)

        pax = 2.0 * transfer_cost(schedule, simple_network)
        ops = 1.5 * operational_cost(schedule, simple_network)
        rob = 0.5 * robustness_cost(schedule, simple_network)

        expected = pax + ops + rob
        actual = total_cost(schedule, simple_network)

        assert abs(actual - expected) < 1e-10

    def test_delta_matches_full_difference(
        self,
        transfer_cost: TransferCost,
        robustness_cost: RobustnessCost,
        simple_network: Network,
    ) -> None:
        """CRITICAL: delta must exactly equal full difference."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            robustness_cost=robustness_cost,
            pax_weight=1.0,
            rob_weight=0.1,
        )

        old_schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = old_schedule.copy()
        new_schedule.shift_line("L1", 5)

        old_cost = total_cost(old_schedule, simple_network)
        new_cost = total_cost(new_schedule, simple_network)
        delta = total_cost.delta(old_schedule, new_schedule, {"L1"}, simple_network)

        assert abs(delta - (new_cost - old_cost)) < 1e-10

    def test_delta_with_multiple_changed_lines(
        self,
        transfer_cost: TransferCost,
        simple_network: Network,
    ) -> None:
        """Test delta when multiple lines change."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            pax_weight=1.0,
        )

        old_schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = old_schedule.copy()
        new_schedule.shift_line("L1", 5)
        new_schedule.shift_line("L2", -3)

        old_cost = total_cost(old_schedule, simple_network)
        new_cost = total_cost(new_schedule, simple_network)
        delta = total_cost.delta(
            old_schedule, new_schedule, {"L1", "L2"}, simple_network
        )

        assert abs(delta - (new_cost - old_cost)) < 1e-10

    def test_decompose(
        self,
        transfer_cost: TransferCost,
        robustness_cost: RobustnessCost,
        operational_cost: OperationalCost,
        simple_network: Network,
    ) -> None:
        """Test that decompose returns correct breakdown."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            operational_cost=operational_cost,
            robustness_cost=robustness_cost,
            pax_weight=2.0,
            ops_weight=1.5,
            rob_weight=0.5,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        breakdown = total_cost.decompose(schedule, simple_network)

        assert "pax" in breakdown
        assert "ops" in breakdown
        assert "rob" in breakdown
        assert "total" in breakdown

        # Verify total matches sum
        expected_total = breakdown["pax"] + breakdown["ops"] + breakdown["rob"]
        assert abs(breakdown["total"] - expected_total) < 1e-10

        # Verify total matches __call__
        assert abs(breakdown["total"] - total_cost(schedule, simple_network)) < 1e-10

    def test_get_component_weights(
        self,
        transfer_cost: TransferCost,
    ) -> None:
        """Test weight retrieval."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            pax_weight=2.0,
            ops_weight=1.5,
            rob_weight=0.5,
        )

        weights = total_cost.get_component_weights()
        assert weights["pax"] == 2.0
        assert weights["ops"] == 1.5
        assert weights["rob"] == 0.5

    def test_negative_weights_rejected(
        self,
        transfer_cost: TransferCost,
    ) -> None:
        """Test that negative weights raise error."""
        with pytest.raises(ValueError):
            TotalCost(
                transfer_cost=transfer_cost,
                pax_weight=-1.0,
            )

    def test_zero_weights_allowed(
        self,
        transfer_cost: TransferCost,
        robustness_cost: RobustnessCost,
        simple_network: Network,
    ) -> None:
        """Test that zero weights effectively disable components."""
        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            robustness_cost=robustness_cost,
            pax_weight=0.0,  # Disable PAX
            rob_weight=1.0,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        total = total_cost(schedule, simple_network)

        # Should only include ROB
        expected = robustness_cost(schedule, simple_network)
        assert abs(total - expected) < 1e-10


class TestCreateTotalCost:
    """Tests for factory functions."""

    @pytest.fixture
    def transfer_cost(self) -> TransferCost:
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0),
        )
        return TransferCost(transfers=TransferSet(transfers={"T1": t}))

    def test_create_with_default_config(self, transfer_cost: TransferCost) -> None:
        total_cost = create_total_cost(transfer_cost=transfer_cost)

        assert total_cost.pax_weight == 1.0
        assert total_cost.ops_weight == 1.0
        assert total_cost.rob_weight == 0.1

    def test_create_with_custom_config(self, transfer_cost: TransferCost) -> None:
        config = TotalCostConfig(
            pax_weight=2.0,
            ops_weight=0.5,
            rob_weight=0.2,
        )
        total_cost = create_total_cost(transfer_cost=transfer_cost, config=config)

        assert total_cost.pax_weight == 2.0
        assert total_cost.ops_weight == 0.5
        assert total_cost.rob_weight == 0.2


class TestTotalCostWithOperational:
    """Tests for TotalCost with operational constraints."""

    @pytest.fixture
    def simple_network(self) -> Network:
        """Network with turnaround possibility."""
        network = Network()

        stations_l1 = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        line1 = Line(
            id="L1",
            modality="bus",
            stations=stations_l1,
            travel_times=[20],
            dwell_times=[5, 5],
            frequency=60,
        )

        stations_l2 = [
            Station(id="B2", location="B", modality="bus"),
            Station(id="A2", location="A", modality="bus"),
        ]
        line2 = Line(
            id="L2",
            modality="bus",
            stations=stations_l2,
            travel_times=[20],
            dwell_times=[5, 5],
            frequency=60,
        )

        network.add_line(line1)
        network.add_line(line2)
        return network

    def test_with_turnaround_cost(
        self,
        simple_network: Network,
    ) -> None:
        """Test TotalCost with turnaround constraints."""
        # Create transfer
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0),
        )
        transfer_set = TransferSet(transfers={"T1": t})
        transfer_cost = TransferCost(transfers=transfer_set)

        # Create turnaround constraint
        turn = TurnaroundConstraint(
            id="TURN1",
            end_line="L1",
            end_station_idx=1,
            start_line="L2",
            start_station_idx=0,
            min_turnaround=5.0,
        )
        turnaround_set = TurnaroundSet(constraints={"TURN1": turn})
        turnaround_cost = TurnaroundCost(turnarounds=turnaround_set)
        ops_cost = OperationalCost(turnaround_cost=turnaround_cost)

        total_cost = TotalCost(
            transfer_cost=transfer_cost,
            operational_cost=ops_cost,
            pax_weight=1.0,
            ops_weight=1.0,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)

        # Verify decomposition
        breakdown = total_cost.decompose(schedule, simple_network)
        assert breakdown["ops"] >= 0
        assert breakdown["total"] == breakdown["pax"] + breakdown["ops"]
