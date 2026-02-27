"""Tests for schedule module."""

import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule


class TestSchedule:
    """Tests for Schedule class."""

    @pytest.fixture
    def sample_network(self) -> Network:
        """Create a sample network with two lines."""
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

    def test_schedule_creation(self) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        assert schedule.get_offset("L1") == 0
        assert schedule.get_offset("L2") == 30
        assert schedule.period == 1440

    def test_schedule_copy(self) -> None:
        original = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        copy = original.copy()

        assert copy == original
        assert copy is not original
        assert copy.offsets is not original.offsets

        # Modifying copy should not affect original
        copy.shift_line("L1", 10)
        assert original.get_offset("L1") == 0
        assert copy.get_offset("L1") == 10

    def test_set_offset_with_modulo(self) -> None:
        schedule = Schedule(offsets={"L1": 0}, period=1440)
        schedule.set_offset("L1", 1500)
        assert schedule.get_offset("L1") == 60  # 1500 % 1440 = 60

    def test_set_offset_negative_wraps(self) -> None:
        schedule = Schedule(offsets={"L1": 0}, period=1440)
        schedule.set_offset("L1", -10)
        assert schedule.get_offset("L1") == 1430  # -10 % 1440 = 1430

    def test_shift_line(self) -> None:
        schedule = Schedule(offsets={"L1": 10}, period=1440)
        schedule.shift_line("L1", 5)
        assert schedule.get_offset("L1") == 15

    def test_shift_line_with_wrap(self) -> None:
        schedule = Schedule(offsets={"L1": 1430}, period=1440)
        schedule.shift_line("L1", 20)
        assert schedule.get_offset("L1") == 10  # (1430 + 20) % 1440 = 10

    def test_shift_line_negative(self) -> None:
        schedule = Schedule(offsets={"L1": 10}, period=1440)
        schedule.shift_line("L1", -15)
        assert schedule.get_offset("L1") == 1435  # (10 - 15) % 1440 = 1435

    def test_shift_lines_multiple(self) -> None:
        schedule = Schedule(offsets={"L1": 10, "L2": 20, "L3": 30}, period=1440)
        schedule.shift_lines({"L1", "L2"}, 5)
        assert schedule.get_offset("L1") == 15
        assert schedule.get_offset("L2") == 25
        assert schedule.get_offset("L3") == 30  # Unchanged

    def test_from_network(self, sample_network: Network) -> None:
        schedule = Schedule.from_network(sample_network, default_offset=10)
        assert len(schedule) == 2
        assert schedule.get_offset("L1") == 10
        assert schedule.get_offset("L2") == 10
        assert schedule.period == 1440

    def test_random_schedule(self, sample_network: Network) -> None:
        import numpy as np

        rng = np.random.default_rng(42)
        schedule = Schedule.random(sample_network, rng)

        assert len(schedule) == 2
        assert 0 <= schedule.get_offset("L1") < 1440
        assert 0 <= schedule.get_offset("L2") < 1440

    def test_random_schedule_reproducible(self, sample_network: Network) -> None:
        import numpy as np

        rng1 = np.random.default_rng(42)
        schedule1 = Schedule.random(sample_network, rng1)

        rng2 = np.random.default_rng(42)
        schedule2 = Schedule.random(sample_network, rng2)

        assert schedule1 == schedule2

    def test_equality(self) -> None:
        s1 = Schedule(offsets={"L1": 10, "L2": 20}, period=1440)
        s2 = Schedule(offsets={"L1": 10, "L2": 20}, period=1440)
        s3 = Schedule(offsets={"L1": 10, "L2": 25}, period=1440)
        s4 = Schedule(offsets={"L1": 10, "L2": 20}, period=720)

        assert s1 == s2
        assert s1 != s3
        assert s1 != s4

    def test_len(self) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 30, "L3": 45}, period=1440)
        assert len(schedule) == 3

    def test_get_nonexistent_line_raises(self) -> None:
        schedule = Schedule(offsets={"L1": 0}, period=1440)
        with pytest.raises(KeyError):
            schedule.get_offset("L999")

    def test_shift_nonexistent_line_raises(self) -> None:
        schedule = Schedule(offsets={"L1": 0}, period=1440)
        with pytest.raises(KeyError):
            schedule.shift_line("L999", 10)
