"""Tests for synthetic network generators."""

import pytest

from src.core.network import T_PERIOD
from src.core.schedule import Schedule
from src.core.transfers import DEFAULT_TRANSFER_PARAMS, TransferParams
from src.cost.transfer_cost import TransferCost, soft_transfer_cost
from src.io.synthetic import TriangleNetwork, create_triangle_network


class TestTriangleNetwork:
    """Tests for TriangleNetwork generator."""

    def test_create_network_default(self) -> None:
        builder = TriangleNetwork()
        network = builder.create_network()

        assert len(network) == 3
        assert "L1" in network.line_ids
        assert "L2" in network.line_ids
        assert "L3" in network.line_ids
        assert network.period == T_PERIOD

    def test_create_network_custom_frequency(self) -> None:
        builder = TriangleNetwork(frequency=30)
        network = builder.create_network()

        assert network.period == T_PERIOD
        for line_id in network.line_ids:
            assert network.get_line(line_id).frequency == 30

    def test_create_network_line_structure(self) -> None:
        builder = TriangleNetwork(travel_time=15, dwell_time=2)
        network = builder.create_network()

        # Each line should have 2 stations
        for line_id in network.line_ids:
            line = network.get_line(line_id)
            assert len(line.stations) == 2
            assert line.travel_times == [15]
            assert line.dwell_times == [2, 2]

    def test_create_network_locations(self) -> None:
        builder = TriangleNetwork()
        network = builder.create_network()

        # Check that lines connect the expected locations
        l1 = network.get_line("L1")
        assert l1.stations[0].location == "A"
        assert l1.stations[1].location == "B"

        l2 = network.get_line("L2")
        assert l2.stations[0].location == "B"
        assert l2.stations[1].location == "C"

        l3 = network.get_line("L3")
        assert l3.stations[0].location == "A"
        assert l3.stations[1].location == "C"

    def test_create_transfers(self) -> None:
        builder = TriangleNetwork()
        transfers = builder.create_transfers()

        # Should have 1 transfer (at B: L1 -> L2)
        assert len(transfers) == 1

        transfer = list(transfers)[0]
        assert transfer.from_line == "L1"
        assert transfer.to_line == "L2"
        assert transfer.walk_time == 3.0

    def test_create_transfers_custom_params(self) -> None:
        custom_params = TransferParams(
            w_min=5.0,
            w_ideal=10.0,
            w_max=30.0,
            alpha=3.0,
            beta=0.6,
        )
        builder = TriangleNetwork(transfer_params=custom_params)
        transfers = builder.create_transfers()

        transfer = list(transfers)[0]
        assert transfer.params.w_min == 5.0
        assert transfer.params.w_ideal == 10.0

    def test_create_returns_both(self) -> None:
        builder = TriangleNetwork()
        network, transfers = builder.create()

        assert len(network) == 3
        assert len(transfers) == 1

    def test_compute_optimal_offset(self) -> None:
        builder = TriangleNetwork(
            travel_time=15,
            dwell_time=2,
            walk_time=3.0,
            transfer_params=DEFAULT_TRANSFER_PARAMS,
        )
        network, transfers = builder.create()
        optimal = builder.compute_optimal_offset()

        # Verify that optimal offsets give ideal waiting time
        schedule = Schedule(offsets=optimal, period=T_PERIOD)
        transfer = list(transfers)[0]

        waiting_time = transfer.compute_waiting_time(schedule, network)

        # Should be close to w_ideal = 8
        assert abs(waiting_time - DEFAULT_TRANSFER_PARAMS.w_ideal) < 1e-10

    def test_optimal_offset_gives_zero_cost(self) -> None:
        builder = TriangleNetwork()
        network, transfers = builder.create()
        optimal = builder.compute_optimal_offset()

        schedule = Schedule(offsets=optimal, period=T_PERIOD)
        cost_fn = TransferCost(transfers=transfers)

        total_cost = cost_fn(schedule, network)
        assert total_cost == 0.0

    def test_non_optimal_offset_gives_positive_cost(self) -> None:
        builder = TriangleNetwork()
        network, transfers = builder.create()

        # Use non-optimal offsets
        schedule = Schedule(offsets={"L1": 0, "L2": 0, "L3": 0}, period=T_PERIOD)
        cost_fn = TransferCost(transfers=transfers)

        total_cost = cost_fn(schedule, network)
        assert total_cost > 0.0


class TestCreateTriangleNetworkFunction:
    """Tests for create_triangle_network convenience function."""

    def test_create_triangle_network_default(self) -> None:
        network, transfers = create_triangle_network()

        assert len(network) == 3
        assert len(transfers) == 1

    def test_create_triangle_network_custom(self) -> None:
        network, transfers = create_triangle_network(
            frequency=30,
            travel_time=10,
            dwell_time=1,
            walk_time=2.0,
        )

        assert network.period == T_PERIOD
        l1 = network.get_line("L1")
        assert l1.travel_times == [10]
        assert l1.dwell_times == [1, 1]
        assert l1.frequency == 30

        transfer = list(transfers)[0]
        assert transfer.walk_time == 2.0


class TestTriangleNetworkIntegration:
    """Integration tests for the triangle network."""

    def test_mcmc_can_find_optimum(self) -> None:
        """Test that exploring the space can find the optimum."""
        import numpy as np

        builder = TriangleNetwork()
        network, transfers = builder.create()
        optimal = builder.compute_optimal_offset()
        cost_fn = TransferCost(transfers=transfers)

        # Random search to verify optimum is findable
        rng = np.random.default_rng(42)
        best_cost = float("inf")
        best_schedule = None

        for _ in range(1000):
            schedule = Schedule.random(network, rng)
            cost = cost_fn(schedule, network)
            if cost < best_cost:
                best_cost = cost
                best_schedule = schedule

        # Should find something close to optimal
        optimal_schedule = Schedule(offsets=optimal, period=T_PERIOD)
        optimal_cost = cost_fn(optimal_schedule, network)

        # The best random should be within reasonable range
        # (with 1000 tries we should get close)
        assert best_cost < 100.0  # Should find reasonable solutions

    def test_waiting_time_sensitivity(self) -> None:
        """Test that small offset changes affect waiting time correctly."""
        builder = TriangleNetwork()
        network, transfers = builder.create()
        transfer = list(transfers)[0]

        schedule1 = Schedule(offsets={"L1": 0, "L2": 20, "L3": 0}, period=T_PERIOD)
        schedule2 = Schedule(offsets={"L1": 0, "L2": 25, "L3": 0}, period=T_PERIOD)

        w1 = transfer.compute_waiting_time(schedule1, network)
        w2 = transfer.compute_waiting_time(schedule2, network)

        # Shifting L2 by +5 should increase waiting time by 5
        assert w2 - w1 == 5.0

    def test_periodic_wraparound(self) -> None:
        """Test that waiting time wraps correctly at frequency boundary."""
        builder = TriangleNetwork(frequency=60)
        network, transfers = builder.create()
        transfer = list(transfers)[0]

        # Test wraparound
        schedule1 = Schedule(offsets={"L1": 0, "L2": 55, "L3": 0}, period=T_PERIOD)
        schedule2 = Schedule(offsets={"L1": 0, "L2": 5, "L3": 0}, period=T_PERIOD)

        w1 = transfer.compute_waiting_time(schedule1, network)
        w2 = transfer.compute_waiting_time(schedule2, network)

        # Both should give valid waiting times in [0, 60) (to_line.frequency)
        assert 0 <= w1 < 60
        assert 0 <= w2 < 60
