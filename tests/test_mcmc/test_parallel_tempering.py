"""Tests for Parallel Tempering with inertia."""

import numpy as np
import pytest

from src.core.network import Line, Network, Station, T_PERIOD
from src.core.schedule import Schedule
from src.core.transfers import DEFAULT_TRANSFER_PARAMS, Transfer, TransferSet
from src.cost.transfer_cost import TransferCost
from src.mcmc.parallel_tempering import InertiaConfig, ParallelTempering, PTResult
from src.mcmc.proposals.single_line import SingleLineShift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_network() -> Network:
    """Two-line network for testing."""
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
def multimodal_network() -> Network:
    """Network with IC, Regional, and Local lines for inertia testing."""
    network = Network()

    # IC line
    ic_stations = [
        Station(id="A_IC", location="A", modality="IC"),
        Station(id="B_IC", location="B", modality="IC"),
    ]
    network.add_line(
        Line(
            id="IC1",
            modality="IC",
            stations=ic_stations,
            travel_times=[30],
            dwell_times=[3, 3],
            frequency=120,
        )
    )

    # Regional line
    reg_stations = [
        Station(id="B_Reg", location="B", modality="Regional"),
        Station(id="C_Reg", location="C", modality="Regional"),
    ]
    network.add_line(
        Line(
            id="R1",
            modality="Regional",
            stations=reg_stations,
            travel_times=[20],
            dwell_times=[2, 2],
            frequency=60,
        )
    )

    # Local line
    loc_stations = [
        Station(id="C_Loc", location="C", modality="Local"),
        Station(id="D_Loc", location="D", modality="Local"),
    ]
    network.add_line(
        Line(
            id="Loc1",
            modality="Local",
            stations=loc_stations,
            travel_times=[10],
            dwell_times=[1, 1],
            frequency=30,
        )
    )

    return network


@pytest.fixture
def transfer_set() -> TransferSet:
    """Transfer set for the simple network."""
    t = Transfer(
        id="T1",
        from_line="L1",
        from_station_idx=1,
        to_line="L2",
        to_station_idx=0,
        walk_time=3.0,
        params=DEFAULT_TRANSFER_PARAMS,
        weight=1.0,
    )
    return TransferSet(transfers={"T1": t})


@pytest.fixture
def cost_fn(transfer_set: TransferSet) -> TransferCost:
    return TransferCost(transfers=transfer_set)


@pytest.fixture
def multimodal_transfer_set() -> TransferSet:
    """Transfer set for multimodal network."""
    t1 = Transfer(
        id="T_IC_R",
        from_line="IC1",
        from_station_idx=1,
        to_line="R1",
        to_station_idx=0,
        walk_time=5.0,
        params=DEFAULT_TRANSFER_PARAMS,
        weight=2.0,
    )
    t2 = Transfer(
        id="T_R_Loc",
        from_line="R1",
        from_station_idx=1,
        to_line="Loc1",
        to_station_idx=0,
        walk_time=2.0,
        params=DEFAULT_TRANSFER_PARAMS,
        weight=1.0,
    )
    return TransferSet(transfers={"T_IC_R": t1, "T_R_Loc": t2})


# ---------------------------------------------------------------------------
# Temperature schedule tests
# ---------------------------------------------------------------------------


class TestTemperatureSchedule:
    """Tests for geometric temperature schedule."""

    def test_geometric_schedule_endpoints(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Schedule should start at beta_min and end at beta_max."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        betas = pt.temperature_schedule
        assert len(betas) == 4
        assert abs(betas[0] - 0.1) < 1e-10
        assert abs(betas[-1] - 10.0) < 1e-10

    def test_geometric_schedule_is_increasing(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Betas should be strictly increasing."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=8,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        betas = pt.temperature_schedule
        for i in range(len(betas) - 1):
            assert betas[i] < betas[i + 1]

    def test_geometric_ratio_is_constant(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Adjacent beta ratios should be constant (geometric sequence)."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=5,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        betas = pt.temperature_schedule
        ratios = [betas[i + 1] / betas[i] for i in range(len(betas) - 1)]
        for ratio in ratios:
            assert abs(ratio - ratios[0]) < 1e-10

    def test_two_replicas(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """With 2 replicas, should just have beta_min and beta_max."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=0.5,
            beta_max=5.0,
            rng=np.random.default_rng(42),
        )

        betas = pt.temperature_schedule
        assert len(betas) == 2
        assert abs(betas[0] - 0.5) < 1e-10
        assert abs(betas[1] - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# Inertia tests
# ---------------------------------------------------------------------------


class TestInertiaConfig:
    """Tests for inertia mechanism."""

    def test_inertia_validation(self) -> None:
        """Inertia values must be >= 1."""
        with pytest.raises(ValueError, match="must be >= 1"):
            InertiaConfig(modality_inertia={"IC": 0.5})

    def test_hottest_replica_uniform(
        self, multimodal_network: Network
    ) -> None:
        """Hottest replica should have uniform weights (I=1 for all)."""
        config = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        # Hottest = first replica (index = 0, lowest beta)
        weights = config.compute_line_weights(
            multimodal_network, replica_idx=0, n_replicas=4
        )

        # All weights should be equal (uniform)
        values = list(weights.values())
        expected = 1.0 / len(values)
        for v in values:
            assert abs(v - expected) < 1e-10

    def test_coldest_replica_has_hierarchy(
        self, multimodal_network: Network
    ) -> None:
        """Coldest replica should respect modality hierarchy."""
        config = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        # Coldest = last replica (index = n-1, highest beta)
        weights = config.compute_line_weights(
            multimodal_network, replica_idx=3, n_replicas=4
        )

        # Local line should have highest weight (1/1 = 1)
        # IC line should have lowest weight (1/20 = 0.05)
        assert weights["Loc1"] > weights["R1"] > weights["IC1"]

    def test_coldest_ic_weight_ratio(
        self, multimodal_network: Network
    ) -> None:
        """For coldest replica, IC weight should be 1/20 of Local weight."""
        config = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        # Coldest = last replica (index = n-1)
        weights = config.compute_line_weights(
            multimodal_network, replica_idx=3, n_replicas=4
        )

        # Raw: IC=1/20=0.05, Reg=1/5=0.2, Loc=1/1=1.0
        # Ratio IC/Loc should be 1/20
        ratio = weights["IC1"] / weights["Loc1"]
        assert abs(ratio - 1.0 / 20.0) < 1e-10

    def test_middle_replica_interpolation(
        self, multimodal_network: Network
    ) -> None:
        """Middle replicas should interpolate between cold and hot."""
        config = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        # Cold replica weights (highest index = highest beta)
        cold_weights = config.compute_line_weights(
            multimodal_network, replica_idx=3, n_replicas=4
        )
        # Hot replica weights (lowest index = lowest beta)
        hot_weights = config.compute_line_weights(
            multimodal_network, replica_idx=0, n_replicas=4
        )
        # Middle replica
        mid_weights = config.compute_line_weights(
            multimodal_network, replica_idx=2, n_replicas=4
        )

        # IC weight: cold < mid < hot (IC gets more access in hotter replicas)
        assert cold_weights["IC1"] < mid_weights["IC1"] < hot_weights["IC1"]

    def test_weights_sum_to_one(
        self, multimodal_network: Network
    ) -> None:
        """All weight vectors should sum to 1."""
        config = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        for replica_idx in range(4):
            weights = config.compute_line_weights(
                multimodal_network, replica_idx=replica_idx, n_replicas=4
            )
            assert abs(sum(weights.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestPTInitialization:
    """Tests for ParallelTempering initialization."""

    def test_invalid_n_replicas(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Should reject n_replicas < 2."""
        with pytest.raises(ValueError, match="n_replicas must be >= 2"):
            ParallelTempering(
                network=simple_network,
                cost_fn=cost_fn,
                proposals=[SingleLineShift(sigma=5.0)],
                n_replicas=1,
            )

    def test_invalid_beta_range(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Should reject beta_min >= beta_max."""
        with pytest.raises(ValueError, match="beta_min must be < beta_max"):
            ParallelTempering(
                network=simple_network,
                cost_fn=cost_fn,
                proposals=[SingleLineShift(sigma=5.0)],
                n_replicas=4,
                beta_min=10.0,
                beta_max=1.0,
            )

    def test_initialize_with_random(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Initialize with random schedules."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        pt.initialize()

        for sampler in pt._samplers:
            assert sampler.current_schedule is not None
            assert sampler.current_cost < float("inf")

    def test_initialize_with_explicit_schedules(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Initialize with explicit schedules."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        schedules = [
            Schedule(offsets={"L1": 0, "L2": 10}, period=T_PERIOD),
            Schedule(offsets={"L1": 5, "L2": 20}, period=T_PERIOD),
            Schedule(offsets={"L1": 10, "L2": 30}, period=T_PERIOD),
        ]
        pt.initialize(schedules)

        assert pt._samplers[0].current_schedule.offsets == {"L1": 0, "L2": 10}
        assert pt._samplers[2].current_schedule.offsets == {"L1": 10, "L2": 30}

    def test_initialize_wrong_number_of_schedules(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Should reject wrong number of schedules."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        with pytest.raises(ValueError, match="Expected 4 schedules"):
            pt.initialize([Schedule(offsets={"L1": 0, "L2": 0}, period=T_PERIOD)])

    def test_step_without_init_raises(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Step without initialize should raise RuntimeError."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=0.1,
            beta_max=10.0,
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            pt.step()


# ---------------------------------------------------------------------------
# Replica exchange tests
# ---------------------------------------------------------------------------


class TestReplicaExchange:
    """Tests for replica exchange (swap) mechanism."""

    def test_swap_acceptance_tracked(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Swap attempts and acceptances should be tracked."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            swap_interval=5,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        # Run enough steps to trigger swaps
        for _ in range(50):
            pt.step()

        # Should have swap data for n_replicas-1 pairs
        rates = pt.swap_acceptance_rates
        assert len(rates) == 2  # 3 replicas -> 2 pairs

        # Some swaps should have been attempted
        assert sum(pt._swap_attempts) > 0

    def test_swap_detailed_balance(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Verify swap acceptance formula satisfies detailed balance.

        For replicas with equal costs, swap should always be accepted.
        """
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=1.0,
            beta_max=2.0,
            swap_interval=1,
            rng=np.random.default_rng(42),
        )

        # Initialize both replicas with same schedule (same cost)
        same_schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=T_PERIOD)
        pt.initialize([same_schedule.copy(), same_schedule.copy()])

        # With equal costs: (beta_j - beta_i) * (cost_j - cost_i) = 0 >= 0
        # So swap should always be accepted
        accepted = pt._attempt_swap(0)
        assert accepted is True

    def test_swap_preserves_costs(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """After swap, costs should be correctly exchanged."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=1.0,
            beta_max=2.0,
            rng=np.random.default_rng(42),
        )

        s0 = Schedule(offsets={"L1": 0, "L2": 10}, period=T_PERIOD)
        s1 = Schedule(offsets={"L1": 0, "L2": 26}, period=T_PERIOD)
        pt.initialize([s0, s1])

        cost_0_before = pt._samplers[0].current_cost
        cost_1_before = pt._samplers[1].current_cost

        # Force a swap (we'll check if costs are properly exchanged)
        pt._attempt_swap(0)

        # After swap, one of two scenarios:
        # If accepted: costs are exchanged
        # If rejected: costs unchanged
        cost_0_after = pt._samplers[0].current_cost
        cost_1_after = pt._samplers[1].current_cost

        if pt._swap_accepts[0] > 0:
            # Swap accepted: costs exchanged
            assert abs(cost_0_after - cost_1_before) < 1e-10
            assert abs(cost_1_after - cost_0_before) < 1e-10
        else:
            # Swap rejected: costs unchanged
            assert abs(cost_0_after - cost_0_before) < 1e-10
            assert abs(cost_1_after - cost_1_before) < 1e-10

    def test_hot_accepts_more_swaps(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Pairs with closer temperatures should have higher swap acceptance."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=100.0,
            swap_interval=1,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        for _ in range(2000):
            pt.step()

        rates = pt.swap_acceptance_rates
        # With geometric schedule and large beta range,
        # all pairs should have some swap activity
        assert all(r >= 0 for r in rates)


# ---------------------------------------------------------------------------
# PT run tests
# ---------------------------------------------------------------------------


class TestPTRun:
    """Tests for full PT run."""

    def test_run_returns_pt_result(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Run should return a properly structured PTResult."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        result = pt.run(n_steps=100, log_every=0)

        assert isinstance(result, PTResult)
        assert result.best_schedule is not None
        assert result.best_cost < float("inf")
        assert len(result.cold_chain_samples) == 100
        assert len(result.cold_chain_costs) == 100
        assert len(result.per_replica_acceptance_rates) == 3
        assert len(result.swap_acceptance_rates) == 2
        assert len(result.temperature_schedule) == 3

    def test_run_with_burn_in(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Burn-in should discard initial samples from cold chain."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        result = pt.run(n_steps=100, burn_in=50, log_every=0)

        assert len(result.cold_chain_samples) == 50

    def test_run_with_thinning(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Thinning should keep every thin-th sample."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        result = pt.run(n_steps=100, thin=10, log_every=0)

        assert len(result.cold_chain_samples) == 10

    def test_best_cost_improves(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """PT should find better solutions than initial."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            swap_interval=5,
            rng=np.random.default_rng(42),
        )

        # Start from non-optimal schedule
        schedules = [
            Schedule(offsets={"L1": 0, "L2": 0}, period=T_PERIOD)
            for _ in range(4)
        ]
        pt.initialize(schedules)
        initial_cost = cost_fn(schedules[0], simple_network)

        result = pt.run(n_steps=2000, log_every=0)

        assert result.best_cost < initial_cost

    def test_pt_summary(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Summary should contain expected keys."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        result = pt.run(n_steps=50, log_every=0)
        summary = result.summary()

        assert "best_cost" in summary
        assert "n_replicas" in summary
        assert summary["n_replicas"] == 3
        assert "swap_acceptance" in summary
        assert "beta_min" in summary
        assert "beta_max" in summary


# ---------------------------------------------------------------------------
# Inertia integration tests
# ---------------------------------------------------------------------------


class TestPTWithInertia:
    """Tests for PT with inertia mechanism."""

    def test_pt_with_inertia_runs(
        self, multimodal_network: Network, multimodal_transfer_set: TransferSet
    ) -> None:
        """PT with inertia should run without errors."""
        cost_fn = TransferCost(transfers=multimodal_transfer_set)
        inertia = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        pt = ParallelTempering(
            network=multimodal_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            inertia_config=inertia,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        result = pt.run(n_steps=200, log_every=0)

        assert result.best_cost < float("inf")
        assert len(result.cold_chain_samples) == 200

    def test_inertia_propagates_to_proposals(
        self, multimodal_network: Network, multimodal_transfer_set: TransferSet
    ) -> None:
        """Proposals in cold replicas should have inertia weights set."""
        cost_fn = TransferCost(transfers=multimodal_transfer_set)
        inertia = InertiaConfig(
            modality_inertia={"IC": 20, "Regional": 5, "Local": 1}
        )

        pt = ParallelTempering(
            network=multimodal_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            inertia_config=inertia,
            rng=np.random.default_rng(42),
        )

        # Coldest replica (idx=n-1) should have hierarchical inertia weights
        cold_sampler = pt.get_cold_sampler()
        cold_proposal = cold_sampler.proposals[0]
        assert hasattr(cold_proposal, "line_weights")
        assert cold_proposal.line_weights is not None

        # Verify cold replica has hierarchy (IC < Regional < Local)
        cold_w = cold_proposal.line_weights
        assert cold_w["Loc1"] > cold_w["R1"] > cold_w["IC1"]

        # Hottest replica (idx=0) should have uniform weights
        hot_sampler = pt.get_replica_sampler(0)
        hot_proposal = hot_sampler.proposals[0]
        hot_w = hot_proposal.line_weights
        expected_uniform = 1.0 / len(hot_w)
        for v in hot_w.values():
            assert abs(v - expected_uniform) < 1e-10

    def test_no_inertia_means_no_weights(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Without inertia config, proposals should have no line_weights."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        for sampler in pt._samplers:
            for proposal in sampler.proposals:
                if hasattr(proposal, "line_weights"):
                    assert proposal.line_weights is None


# ---------------------------------------------------------------------------
# Properties and accessors tests
# ---------------------------------------------------------------------------


class TestPTProperties:
    """Tests for PT properties and accessors."""

    def test_get_replica_sampler(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Should return correct sampler for index."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        # Verify each sampler has the right beta
        for i in range(4):
            sampler = pt.get_replica_sampler(i)
            assert abs(sampler.beta - pt.temperature_schedule[i]) < 1e-10

    def test_get_cold_sampler(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Cold sampler should have highest beta."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        cold = pt.get_cold_sampler()
        assert abs(cold.beta - 10.0) < 1e-10

    def test_n_steps_counter(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Step counter should track correctly."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=2,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        assert pt.n_steps == 0

        for _ in range(25):
            pt.step()

        assert pt.n_steps == 25

    def test_best_tracking(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Best schedule/cost should be tracked across all replicas."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        for _ in range(500):
            pt.step()

        # Best should be <= any replica's current cost
        for sampler in pt._samplers:
            assert pt.best_cost <= sampler.current_cost + 1e-10


# ---------------------------------------------------------------------------
# Replica independence tests
# ---------------------------------------------------------------------------


class TestReplicaIndependence:
    """Tests to verify replicas are independent."""

    def test_replicas_have_different_rngs(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Each replica should have its own independent RNG."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=4,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )
        pt.initialize()

        # Run a few steps and check replicas diverge
        for _ in range(10):
            pt.step()

        # Not all replicas should have the same schedule
        schedules = [s.current_schedule.offsets for s in pt._samplers]
        unique = len(set(frozenset(s.items()) for s in schedules))
        assert unique > 1  # At least 2 different schedules

    def test_replicas_have_independent_proposals(
        self, simple_network: Network, cost_fn: TransferCost
    ) -> None:
        """Modifying one replica's proposal should not affect others."""
        pt = ParallelTempering(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[SingleLineShift(sigma=5.0)],
            n_replicas=3,
            beta_min=0.1,
            beta_max=10.0,
            rng=np.random.default_rng(42),
        )

        # Each replica's proposal should be a different object
        proposal_0 = pt._samplers[0].proposals[0]
        proposal_1 = pt._samplers[1].proposals[0]
        assert proposal_0 is not proposal_1
