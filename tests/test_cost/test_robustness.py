"""Tests for robustness cost module."""

import math

import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import Transfer, TransferParams, TransferSet
from src.cost.robustness import (
    DEFAULT_DELAY_DISTRIBUTION,
    DelayDistribution,
    RobustnessCost,
)


class TestDelayDistribution:
    """Tests for DelayDistribution class."""

    def test_valid_construction(self) -> None:
        dist = DelayDistribution(rate=0.5, alpha=0.95)
        assert dist.rate == 0.5
        assert dist.alpha == 0.95

    def test_invalid_rate(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            DelayDistribution(rate=0, alpha=0.95)
        with pytest.raises(ValueError, match="rate must be positive"):
            DelayDistribution(rate=-1, alpha=0.95)

    def test_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be in"):
            DelayDistribution(rate=0.5, alpha=0)
        with pytest.raises(ValueError, match="alpha must be in"):
            DelayDistribution(rate=0.5, alpha=1)

    def test_mean(self) -> None:
        # rate = 0.5 -> mean = 2
        dist = DelayDistribution(rate=0.5, alpha=0.95)
        assert dist.mean == 2.0

        # rate = 1/3 -> mean = 3
        dist2 = DelayDistribution(rate=1.0 / 3.0, alpha=0.95)
        assert abs(dist2.mean - 3.0) < 1e-10

    def test_var_alpha(self) -> None:
        # VaR_α = -ln(1-α) / λ
        # For α=0.95, -ln(0.05) ≈ 2.996
        # For λ=1, VaR ≈ 2.996
        dist = DelayDistribution(rate=1.0, alpha=0.95)
        expected = -math.log(1 - 0.95)
        assert abs(dist.var_alpha - expected) < 1e-10

    def test_cvar_alpha(self) -> None:
        # CVaR_α = (-ln(1-α) + 1) / λ
        # For α=0.95, λ=1: (-ln(0.05) + 1) ≈ 3.996
        dist = DelayDistribution(rate=1.0, alpha=0.95)
        expected = -math.log(1 - 0.95) + 1
        assert abs(dist.cvar_alpha - expected) < 1e-10

    def test_cvar_greater_than_var(self) -> None:
        # CVaR is always greater than VaR
        dist = DelayDistribution(rate=0.5, alpha=0.95)
        assert dist.cvar_alpha > dist.var_alpha

    def test_default_distribution(self) -> None:
        # Default has mean=3
        assert abs(DEFAULT_DELAY_DISTRIBUTION.mean - 3.0) < 1e-10
        assert DEFAULT_DELAY_DISTRIBUTION.alpha == 0.95


class TestRobustnessCost:
    """Tests for RobustnessCost class."""

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
    def delay_dist(self) -> DelayDistribution:
        """Small delay distribution for testing."""
        return DelayDistribution(rate=1.0, alpha=0.95)  # CVaR ≈ 4

    @pytest.fixture
    def cost_fn(
        self, transfer_set: TransferSet, delay_dist: DelayDistribution
    ) -> RobustnessCost:
        return RobustnessCost(transfers=transfer_set, delay_distribution=delay_dist)

    def test_compute_buffer(
        self,
        cost_fn: RobustnessCost,
        simple_network: Network,
        transfer_set: TransferSet,
    ) -> None:
        # With ideal schedule: waiting = 8, walk = 3, buffer = 5
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        transfer = list(transfer_set)[0]

        buffer = cost_fn.compute_buffer(transfer, schedule, simple_network)
        # waiting - walk_time = 8 - 3 = 5
        assert abs(buffer - 5.0) < 1e-10

    def test_robustness_cost_with_sufficient_buffer(
        self,
        cost_fn: RobustnessCost,
        simple_network: Network,
    ) -> None:
        # CVaR ≈ 4, buffer = 5 (waiting=8, walk=3)
        # Buffer >= CVaR, so cost = 0
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        total = cost_fn(schedule, simple_network)
        assert total == 0.0

    def test_robustness_cost_with_insufficient_buffer(
        self,
        simple_network: Network,
        transfer_set: TransferSet,
    ) -> None:
        # Use larger CVaR so buffer is insufficient
        dist = DelayDistribution(rate=0.2, alpha=0.95)  # CVaR ≈ 20
        cost_fn = RobustnessCost(transfers=transfer_set, delay_distribution=dist)

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        # waiting = 8, walk = 3, buffer = 5
        # CVaR ≈ 20, shortfall = 20 - 5 = 15
        total = cost_fn(schedule, simple_network)
        expected_shortfall = dist.cvar_alpha - 5.0
        assert abs(total - expected_shortfall) < 1e-10

    def test_delta_matches_full_difference(
        self,
        cost_fn: RobustnessCost,
        simple_network: Network,
    ) -> None:
        """CRITICAL: delta must exactly equal full difference."""
        old_schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = old_schedule.copy()
        new_schedule.shift_line("L1", 5)

        old_cost = cost_fn(old_schedule, simple_network)
        new_cost = cost_fn(new_schedule, simple_network)
        delta = cost_fn.delta(old_schedule, new_schedule, {"L1"}, simple_network)

        assert abs(delta - (new_cost - old_cost)) < 1e-10

    def test_delta_with_no_change(
        self,
        cost_fn: RobustnessCost,
        simple_network: Network,
    ) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        new_schedule = schedule.copy()

        delta = cost_fn.delta(schedule, new_schedule, {"L1"}, simple_network)
        assert delta == 0.0

    def test_weighted_robustness_cost(
        self,
        simple_network: Network,
    ) -> None:
        """Test that weight is applied correctly."""
        dist = DelayDistribution(rate=0.2, alpha=0.95)  # CVaR ≈ 20
        params = TransferParams(
            w_min=3.0,
            w_ideal=8.0,
            w_max=30.0,
        )
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
            weight=5.0,  # Weight = 5
        )
        ts = TransferSet(transfers={"T1": t})
        cost_fn = RobustnessCost(transfers=ts, delay_distribution=dist)

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        # buffer = 5, shortfall = CVaR - 5
        # cost = weight * shortfall = 5 * shortfall
        shortfall = dist.cvar_alpha - 5.0
        expected = 5.0 * shortfall

        total = cost_fn(schedule, simple_network)
        assert abs(total - expected) < 1e-10

    def test_count_at_risk_transfers(
        self,
        simple_network: Network,
        transfer_set: TransferSet,
    ) -> None:
        # With small CVaR, no transfers at risk
        dist_small = DelayDistribution(rate=1.0, alpha=0.95)  # CVaR ≈ 4
        cost_fn = RobustnessCost(transfers=transfer_set, delay_distribution=dist_small)

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        # buffer = 5 > CVaR ≈ 4
        assert cost_fn.count_at_risk_transfers(schedule, simple_network) == 0

        # With large CVaR, transfer is at risk
        dist_large = DelayDistribution(rate=0.2, alpha=0.95)  # CVaR ≈ 20
        cost_fn2 = RobustnessCost(transfers=transfer_set, delay_distribution=dist_large)
        assert cost_fn2.count_at_risk_transfers(schedule, simple_network) == 1

    def test_robustness_details(
        self,
        cost_fn: RobustnessCost,
        simple_network: Network,
    ) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        details = cost_fn.get_robustness_details(schedule, simple_network)

        assert len(details) == 1
        d = details[0]
        assert d["id"] == "T1"
        assert d["buffer"] == 5.0
        assert d["at_risk"] is False  # buffer=5 > CVaR≈4
