"""Tests for transfer cost module."""

import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import (
    DEFAULT_TRANSFER_PARAMS,
    Transfer,
    TransferParams,
    TransferSet,
)
from src.cost.transfer_cost import (
    TransferCost,
    barrier_cost,
    soft_transfer_cost,
)


class TestBarrierCost:
    """Tests for barrier_cost function."""

    def test_no_violation(self) -> None:
        assert barrier_cost(-5.0, 1e6) == 0.0
        assert barrier_cost(0.0, 1e6) == 0.0

    def test_positive_violation(self) -> None:
        # violation = 2, M = 1e6 -> 4e6
        cost = barrier_cost(2.0, 1e6)
        assert cost == 4e6

    def test_small_violation(self) -> None:
        cost = barrier_cost(0.5, 1e6)
        assert cost == 0.25e6


class TestSoftTransferCost:
    """Tests for soft_transfer_cost function."""

    @pytest.fixture
    def params(self) -> TransferParams:
        return TransferParams(
            w_min=3.0,
            w_ideal=8.0,
            w_max=30.0,
            alpha=2.0,
            beta=0.5,
            M=1e6,
        )

    def test_below_w_min_barrier(self, params: TransferParams) -> None:
        # w=1 < w_min=3, violation=2
        cost = soft_transfer_cost(1.0, params)
        expected = 1e6 * (3.0 - 1.0) ** 2  # M * 2^2 = 4e6
        assert cost == expected

    def test_at_w_min(self, params: TransferParams) -> None:
        # w=3 = w_min, no barrier but quadratic to ideal
        cost = soft_transfer_cost(3.0, params)
        expected = 2.0 * (8.0 - 3.0) ** 2  # alpha * 5^2 = 50
        assert cost == expected

    def test_between_w_min_and_w_ideal(self, params: TransferParams) -> None:
        # w=5, alpha * (8-5)^2 = 2 * 9 = 18
        cost = soft_transfer_cost(5.0, params)
        assert cost == 18.0

    def test_at_w_ideal(self, params: TransferParams) -> None:
        # w = w_ideal = 8, cost should be 0
        cost = soft_transfer_cost(8.0, params)
        assert cost == 0.0

    def test_above_w_ideal(self, params: TransferParams) -> None:
        # w=12, beta * (12-8)^2 = 0.5 * 16 = 8
        cost = soft_transfer_cost(12.0, params)
        assert cost == 8.0

    def test_way_above_w_ideal(self, params: TransferParams) -> None:
        # w=28, beta * (28-8)^2 = 0.5 * 400 = 200
        cost = soft_transfer_cost(28.0, params)
        assert cost == 200.0


class TestTransferCost:
    """Tests for TransferCost class."""

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
    def cost_fn(self, transfer_set: TransferSet) -> TransferCost:
        return TransferCost(transfers=transfer_set)

    def test_compute_single_transfer_cost(
        self,
        cost_fn: TransferCost,
        simple_network: Network,
    ) -> None:
        # Set up schedule to get waiting_time = 8 (ideal)
        # L1 arrives at B at: offset_L1 + 2 + 15 = offset_L1 + 17
        # L2 departs from B at: offset_L2 + 2
        # waiting = (offset_L2 + 2) - (offset_L1 + 17) - 3 = offset_L2 - offset_L1 - 18
        # For waiting = 8: offset_L2 - offset_L1 = 26

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        transfer = list(cost_fn.transfers)[0]
        cost = cost_fn.compute_single_transfer_cost(transfer, schedule, simple_network)
        assert cost == 0.0  # Ideal waiting time

    def test_total_cost(
        self,
        cost_fn: TransferCost,
        simple_network: Network,
    ) -> None:
        # With ideal schedule
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        total = cost_fn(schedule, simple_network)
        assert total == 0.0

    def test_delta_cost_unchanged(
        self,
        cost_fn: TransferCost,
        simple_network: Network,
    ) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = schedule.copy()

        delta = cost_fn.delta(schedule, new_schedule, {"L1"}, simple_network)
        assert delta == 0.0

    def test_delta_cost_changed(
        self,
        cost_fn: TransferCost,
        simple_network: Network,
    ) -> None:
        old_schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = old_schedule.copy()
        new_schedule.shift_line("L1", 5)

        old_cost = cost_fn(old_schedule, simple_network)
        new_cost = cost_fn(new_schedule, simple_network)
        delta = cost_fn.delta(old_schedule, new_schedule, {"L1"}, simple_network)

        assert abs(delta - (new_cost - old_cost)) < 1e-10

    def test_delta_cost_only_considers_affected_transfers(
        self,
        simple_network: Network,
    ) -> None:
        # Create two transfers, only one involving L1
        params = DEFAULT_TRANSFER_PARAMS
        t1 = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        t2 = Transfer(
            id="T2",
            from_line="L2",
            from_station_idx=1,
            to_line="L2",  # Self-loop (artificial)
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t1, "T2": t2})
        cost_fn = TransferCost(transfers=ts)

        old_schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = old_schedule.copy()
        new_schedule.shift_line("L1", 5)

        # Delta should only account for T1 (involves L1), not T2
        delta = cost_fn.delta(old_schedule, new_schedule, {"L1"}, simple_network)

        # Manual calculation for just T1
        old_wait_t1 = t1.compute_waiting_time(old_schedule, simple_network)
        new_wait_t1 = t1.compute_waiting_time(new_schedule, simple_network)
        old_cost_t1 = soft_transfer_cost(old_wait_t1, params)
        new_cost_t1 = soft_transfer_cost(new_wait_t1, params)

        assert abs(delta - (new_cost_t1 - old_cost_t1)) < 1e-10

    def test_count_violations(
        self,
        cost_fn: TransferCost,
        simple_network: Network,
    ) -> None:
        # Ideal schedule - no violations
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        assert cost_fn.count_violations(schedule, simple_network) == 0

        # Bad schedule causing violation
        bad_schedule = Schedule(offsets={"L1": 0, "L2": 15}, period=1440)
        # waiting = 15 + 2 - 17 - 3 = -3, mod 60 = 57 (no violation actually)
        # Let's try another
        # waiting = (offset_L2 + 2) - (0 + 17) - 3 = offset_L2 - 18
        # For violation (w < 3): offset_L2 < 21
        # At offset_L2 = 19: waiting = 1 (violation)
        violation_schedule = Schedule(offsets={"L1": 0, "L2": 19}, period=1440)
        waiting = 19 + 2 - 17 - 3  # = 1 < w_min=3
        assert waiting == 1
        assert cost_fn.count_violations(violation_schedule, simple_network) == 1

    def test_get_worst_transfer(
        self,
        simple_network: Network,
    ) -> None:
        params = DEFAULT_TRANSFER_PARAMS
        t1 = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
            weight=1.0,
        )
        t2 = Transfer(
            id="T2",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
            weight=10.0,  # Higher weight
        )
        ts = TransferSet(transfers={"T1": t1, "T2": t2})
        cost_fn = TransferCost(transfers=ts)

        # With same waiting time, T2 has higher cost due to weight
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        # Both at ideal, so both have 0 cost
        worst, cost = cost_fn.get_worst_transfer(schedule, simple_network)
        assert cost == 0.0

        # Change to non-ideal
        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        worst, cost = cost_fn.get_worst_transfer(schedule, simple_network)
        assert worst is not None
        assert worst.id == "T2"  # Higher weight = higher cost

    def test_get_transfer_details(
        self,
        cost_fn: TransferCost,
        simple_network: Network,
    ) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        details = cost_fn.get_transfer_details(schedule, simple_network)

        assert len(details) == 1
        d = details[0]
        assert d["id"] == "T1"
        assert d["waiting_time"] == 8.0
        assert d["cost"] == 0.0
        assert d["violated"] is False

    def test_weighted_cost(self, simple_network: Network) -> None:
        params = DEFAULT_TRANSFER_PARAMS
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
            weight=5.0,
        )
        ts = TransferSet(transfers={"T1": t})
        cost_fn = TransferCost(transfers=ts)

        # Non-ideal schedule
        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        # waiting = 30 + 2 - 17 - 3 = 12, above ideal=8
        # cost = beta * (12-8)^2 = 0.5 * 16 = 8
        # weighted = 5 * 8 = 40

        total = cost_fn(schedule, simple_network)
        assert total == 40.0
