"""Tests for network module."""

import pytest

from src.core.network import Line, Network, Station


class TestStation:
    """Tests for Station dataclass."""

    def test_station_creation(self) -> None:
        station = Station(id="S1", location="Warsaw", modality="IC")
        assert station.id == "S1"
        assert station.location == "Warsaw"
        assert station.modality == "IC"

    def test_station_is_hashable(self) -> None:
        station = Station(id="S1", location="Warsaw", modality="IC")
        # Should be usable in sets and dicts
        s = {station}
        assert station in s

    def test_station_equality(self) -> None:
        s1 = Station(id="S1", location="Warsaw", modality="IC")
        s2 = Station(id="S1", location="Warsaw", modality="IC")
        s3 = Station(id="S2", location="Warsaw", modality="IC")
        assert s1 == s2
        assert s1 != s3


class TestLine:
    """Tests for Line dataclass."""

    @pytest.fixture
    def simple_line(self) -> Line:
        """Create a simple two-station line."""
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        return Line(
            id="L1",
            modality="bus",
            stations=stations,
            travel_times=[15],
            dwell_times=[2, 2],
            frequency=60,
        )

    @pytest.fixture
    def three_station_line(self) -> Line:
        """Create a three-station line."""
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
            Station(id="C", location="C", modality="bus"),
        ]
        return Line(
            id="L1",
            modality="bus",
            stations=stations,
            travel_times=[10, 15],
            dwell_times=[2, 3, 2],
            frequency=60,
        )

    def test_line_creation(self, simple_line: Line) -> None:
        assert simple_line.id == "L1"
        assert simple_line.modality == "bus"
        assert len(simple_line.stations) == 2
        assert simple_line.frequency == 60

    def test_line_invalid_travel_times_length(self) -> None:
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        with pytest.raises(ValueError, match="travel_times must have length 1"):
            Line(
                id="L1",
                modality="bus",
                stations=stations,
                travel_times=[10, 15],  # Wrong length
                dwell_times=[2, 2],
                frequency=60,
            )

    def test_line_invalid_dwell_times_length(self) -> None:
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        with pytest.raises(ValueError, match="dwell_times must have length 2"):
            Line(
                id="L1",
                modality="bus",
                stations=stations,
                travel_times=[15],
                dwell_times=[2],  # Wrong length
                frequency=60,
            )

    def test_line_invalid_frequency(self) -> None:
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        with pytest.raises(ValueError, match="frequency must be positive"):
            Line(
                id="L1",
                modality="bus",
                stations=stations,
                travel_times=[15],
                dwell_times=[2, 2],
                frequency=0,
            )

    def test_total_travel_time(self, three_station_line: Line) -> None:
        assert three_station_line.total_travel_time() == 25  # 10 + 15

    def test_total_run_time(self, three_station_line: Line) -> None:
        # travel: 10 + 15 = 25, dwell: 2 + 3 + 2 = 7
        assert three_station_line.total_run_time() == 32

    def test_arrival_offset_first_station(self, three_station_line: Line) -> None:
        # Arrival at first station is at offset 0
        assert three_station_line.arrival_offset(0) == 0

    def test_arrival_offset_second_station(self, three_station_line: Line) -> None:
        # Arrival at B: travel[0] + dwell[0] = 10 + 2 = 12
        assert three_station_line.arrival_offset(1) == 12

    def test_arrival_offset_third_station(self, three_station_line: Line) -> None:
        # Arrival at C: travel[0] + travel[1] + dwell[0] + dwell[1] = 10 + 15 + 2 + 3 = 30
        assert three_station_line.arrival_offset(2) == 30

    def test_departure_offset_first_station(self, three_station_line: Line) -> None:
        # Departure from A: arrival(0) + dwell[0] = 0 + 2 = 2
        assert three_station_line.departure_offset(0) == 2

    def test_departure_offset_second_station(self, three_station_line: Line) -> None:
        # Departure from B: arrival(1) + dwell[1] = 12 + 3 = 15
        assert three_station_line.departure_offset(1) == 15

    def test_arrival_offset_invalid_index(self, simple_line: Line) -> None:
        with pytest.raises(IndexError):
            simple_line.arrival_offset(5)


class TestNetwork:
    """Tests for Network class."""

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

    def test_network_creation(self) -> None:
        network = Network()
        assert network.period == 1440
        assert len(network) == 0

    def test_add_line(self, sample_network: Network) -> None:
        assert len(sample_network) == 2
        assert "L1" in sample_network.line_ids
        assert "L2" in sample_network.line_ids

    def test_add_duplicate_line_raises(self, sample_network: Network) -> None:
        stations = [
            Station(id="X", location="X", modality="bus"),
            Station(id="Y", location="Y", modality="bus"),
        ]
        duplicate = Line(
            id="L1",  # Same ID as existing
            modality="bus",
            stations=stations,
            travel_times=[10],
            dwell_times=[2, 2],
            frequency=60,
        )
        with pytest.raises(ValueError, match="already exists"):
            sample_network.add_line(duplicate)

    def test_get_line(self, sample_network: Network) -> None:
        line = sample_network.get_line("L1")
        assert line.id == "L1"

    def test_get_nonexistent_line_raises(self, sample_network: Network) -> None:
        with pytest.raises(KeyError):
            sample_network.get_line("L999")

    def test_get_lines_at_location(self, sample_network: Network) -> None:
        lines_at_b = sample_network.get_lines_at_location("B")
        assert len(lines_at_b) == 2

        lines_at_a = sample_network.get_lines_at_location("A")
        assert len(lines_at_a) == 1

        lines_at_x = sample_network.get_lines_at_location("X")
        assert len(lines_at_x) == 0

    def test_get_station_index(self, sample_network: Network) -> None:
        line = sample_network.get_line("L1")
        idx = sample_network.get_station_index(line, "B")
        assert idx == 1

        idx = sample_network.get_station_index(line, "A")
        assert idx == 0

        idx = sample_network.get_station_index(line, "X")
        assert idx is None

    def test_line_ids(self, sample_network: Network) -> None:
        ids = sample_network.line_ids
        assert set(ids) == {"L1", "L2"}
