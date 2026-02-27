"""Tests for MCMC sampler."""

import numpy as np
import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import DEFAULT_TRANSFER_PARAMS, Transfer, TransferSet
from src.cost.transfer_cost import TransferCost
from src.mcmc.proposals.single_line import SingleLineShift
from src.mcmc.sampler import MCMCSampler


@pytest.fixture
def simple_network() -> Network:
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
def transfer_set() -> TransferSet:
    """Create a TransferSet with one transfer."""
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


class TestMCMCSampler:
    """Tests for MCMCSampler."""

    def test_initialization(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            beta=1.0,
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        sampler.initialize(schedule)

        assert sampler.current_schedule is not None
        assert sampler.current_cost == cost_fn(schedule, simple_network)

    def test_step_without_init_raises(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        with pytest.raises(RuntimeError, match="not initialized"):
            sampler.step()

    def test_step_returns_schedule_and_acceptance(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 26}, period=1440)
        sampler.initialize(schedule)

        new_schedule, accepted = sampler.step()

        assert isinstance(new_schedule, Schedule)
        assert isinstance(accepted, (bool, np.bool_))

    def test_run_returns_result(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        sampler.initialize(schedule)

        result = sampler.run(n_steps=100, log_every=0)

        assert len(result.samples) == 100
        assert len(result.costs) == 100
        assert 0 <= result.acceptance_rate <= 1
        assert result.best_schedule is not None
        assert result.best_cost <= result.costs[0]

    def test_run_with_thinning(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        sampler.initialize(schedule)

        result = sampler.run(n_steps=100, thin=10, log_every=0)

        assert len(result.samples) == 10

    def test_run_with_burnin(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        sampler.initialize(schedule)

        result = sampler.run(n_steps=100, burn_in=50, log_every=0)

        assert len(result.samples) == 50

    def test_high_beta_is_greedy(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """High beta should mostly accept improvements and reject worse moves."""
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)

        # Start from non-optimal
        schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)

        # High beta = very greedy
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            beta=100.0,
            rng=rng,
        )
        sampler.initialize(schedule)

        result = sampler.run(n_steps=1000, log_every=0)

        # Should find good solution
        assert result.best_cost < cost_fn(schedule, simple_network)

    def test_tracks_best(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)
        sampler.initialize(schedule)

        result = sampler.run(n_steps=500, log_every=0)

        # Best should be <= minimum of samples
        assert result.best_cost <= min(result.costs)

    def test_acceptance_rate_property(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        sampler.initialize(schedule)

        assert sampler.acceptance_rate == 0.0  # No steps yet

        sampler.run(n_steps=100, log_every=0)

        assert 0 < sampler.acceptance_rate < 1

    def test_summary(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        sampler.initialize(schedule)

        result = sampler.run(n_steps=100, log_every=0)
        summary = result.summary()

        assert "n_samples" in summary
        assert "acceptance_rate" in summary
        assert "best_cost" in summary
        assert "mean_cost" in summary
        assert summary["n_samples"] == 100


class TestMCMCConvergence:
    """Integration tests for MCMC convergence."""

    def test_converges_to_optimum_on_simple_problem(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """MCMC should find optimal solution on simple problem."""
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            beta=1.0,
            rng=rng,
        )

        # Start from random
        schedule = Schedule.random(simple_network, rng)
        sampler.initialize(schedule)

        result = sampler.run(n_steps=5000, log_every=0)

        # Should find very low cost (close to 0)
        assert result.best_cost < 1.0


class TestProposalAdapterIntegration:
    """Tests for ProposalAdapter integration with MCMCSampler."""

    def test_adapter_created_in_sampling_mode(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """Adapter should be created in sampling mode."""
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            mode="sampling",
            beta=1.0,
            rng=rng,
        )

        schedule = Schedule.from_network(simple_network)
        sampler.initialize(schedule)

        assert sampler._adapter is not None
        assert sampler._adapter.mode == "sampling"

    def test_adapter_not_created_in_optimization_mode(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """Adapter should NOT be created in optimization mode."""
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            mode="optimization",
            beta=1.0,
            rng=rng,
        )

        schedule = Schedule.from_network(simple_network)
        sampler.initialize(schedule)

        assert sampler._adapter is None

    def test_adapter_weights_used_in_sampling_mode(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """Adapter weights should be used and updated in sampling mode."""
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            mode="sampling",
            beta=1.0,
            rng=rng,
        )

        schedule = Schedule.from_network(simple_network)
        sampler.initialize(schedule)

        # Run several steps
        for _ in range(100):
            sampler.step()

        # Verify adapter step count
        assert sampler._adapter._n_step == 100

    def test_static_weights_in_optimization_mode(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """Weights should remain static in optimization mode."""
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=[proposal],
            mode="optimization",
            beta=1.0,
            rng=rng,
        )

        schedule = Schedule.from_network(simple_network)
        sampler.initialize(schedule)

        initial_weights = sampler._normalized_weights.copy()

        for _ in range(100):
            sampler.step()

        # Verify weights unchanged
        assert sampler._normalized_weights == initial_weights

    def test_containment_maintained_in_sampling_mode(
        self,
        simple_network: Network,
        cost_fn: TransferCost,
    ) -> None:
        """Containment p_j >= epsilon should be maintained in sampling mode."""
        rng = np.random.default_rng(42)
        # Use multiple proposals to test containment
        proposals = [
            SingleLineShift(sigma=5.0, rng=rng),
            SingleLineShift(sigma=10.0, rng=rng),
        ]
        sampler = MCMCSampler(
            network=simple_network,
            cost_fn=cost_fn,
            proposals=proposals,
            mode="sampling",
            beta=1.0,
            rng=rng,
        )

        schedule = Schedule.from_network(simple_network)
        sampler.initialize(schedule)

        for _ in range(1000):
            sampler.step()

        # Check all weights >= epsilon
        weights = sampler._adapter.get_weights()
        assert all(w >= sampler._adapter.epsilon for w in weights), \
            f"Weights {weights} violate containment epsilon={sampler._adapter.epsilon}"
