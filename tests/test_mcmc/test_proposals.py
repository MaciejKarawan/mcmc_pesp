"""Tests for MCMC proposals."""

import numpy as np
import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.mcmc.proposals.single_line import SingleLineShift, UniformLineShift


@pytest.fixture
def simple_network() -> Network:
    """Create a simple network with two lines."""
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


class TestSingleLineShift:
    """Tests for SingleLineShift proposal."""

    def test_name(self) -> None:
        proposal = SingleLineShift()
        assert proposal.name == "single_line_shift"

    def test_propose_returns_new_schedule(
        self,
        simple_network: Network,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        schedule = Schedule(offsets={"L1": 10, "L2": 20}, period=1440)

        new_schedule, hastings_ratio, changed_lines = proposal.propose(
            schedule, simple_network
        )

        assert new_schedule is not schedule
        assert hastings_ratio == 1.0  # Symmetric proposal
        assert len(changed_lines) == 1
        assert list(changed_lines)[0] in ["L1", "L2"]

    def test_propose_modifies_one_line(
        self,
        simple_network: Network,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=5.0, rng=rng)
        schedule = Schedule(offsets={"L1": 10, "L2": 20}, period=1440)

        new_schedule, _, changed_lines = proposal.propose(schedule, simple_network)

        changed_line = list(changed_lines)[0]
        unchanged_line = "L2" if changed_line == "L1" else "L1"

        # One line should be unchanged
        assert new_schedule.get_offset(unchanged_line) == schedule.get_offset(
            unchanged_line
        )

    def test_propose_respects_period(
        self,
        simple_network: Network,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = SingleLineShift(sigma=30.0, rng=rng)  # Large sigma
        schedule = Schedule(offsets={"L1": 55, "L2": 55}, period=1440)

        for _ in range(100):
            new_schedule, _, _ = proposal.propose(schedule, simple_network)
            for line_id in new_schedule.offsets:
                offset = new_schedule.get_offset(line_id)
                assert 0 <= offset < 1440

    def test_propose_reproducible(
        self,
        simple_network: Network,
    ) -> None:
        schedule = Schedule(offsets={"L1": 10, "L2": 20}, period=1440)

        rng1 = np.random.default_rng(42)
        proposal1 = SingleLineShift(sigma=5.0, rng=rng1)
        result1 = proposal1.propose(schedule, simple_network)

        rng2 = np.random.default_rng(42)
        proposal2 = SingleLineShift(sigma=5.0, rng=rng2)
        result2 = proposal2.propose(schedule, simple_network)

        assert result1[0] == result2[0]
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]


class TestUniformLineShift:
    """Tests for UniformLineShift proposal."""

    def test_name(self) -> None:
        proposal = UniformLineShift()
        assert proposal.name == "uniform_line_shift"

    def test_propose_within_max_shift(
        self,
        simple_network: Network,
    ) -> None:
        rng = np.random.default_rng(42)
        max_shift = 5
        proposal = UniformLineShift(max_shift=max_shift, rng=rng)
        schedule = Schedule(offsets={"L1": 30, "L2": 30}, period=1440)

        for _ in range(100):
            new_schedule, _, changed_lines = proposal.propose(
                schedule, simple_network
            )
            changed_line = list(changed_lines)[0]
            original = schedule.get_offset(changed_line)
            new = new_schedule.get_offset(changed_line)

            # Compute actual shift (handling wraparound)
            diff = (new - original) % 1440
            if diff > 720:
                diff = diff - 1440

            assert -max_shift <= diff <= max_shift

    def test_symmetric_proposal(
        self,
        simple_network: Network,
    ) -> None:
        rng = np.random.default_rng(42)
        proposal = UniformLineShift(max_shift=10, rng=rng)
        schedule = Schedule(offsets={"L1": 30, "L2": 30}, period=1440)

        _, hastings_ratio, _ = proposal.propose(schedule, simple_network)
        assert hastings_ratio == 1.0
