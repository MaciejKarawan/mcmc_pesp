"""Tests for peak concurrency diagnostic metric."""

from src.core.network import Line, Network, Station, T_PERIOD
from src.core.schedule import Schedule
from src.metrics.peak_concurrency import compute_peak_concurrency


class TestPeakConcurrency:
    """Tests for compute_peak_concurrency."""

    def test_peak_concurrency_single_line(self) -> None:
        """Single line with short run time needs 1 vehicle."""
        network = Network()
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        line = Line(
            id="L1",
            modality="bus",
            stations=stations,
            travel_times=[15],
            dwell_times=[2, 2],
            frequency=60,
        )
        network.add_line(line)

        schedule = Schedule(offsets={"L1": 0})

        peak = compute_peak_concurrency(schedule, network)
        # run_time = 15 + 2 + 2 = 19 min, frequency = 60
        # Only 1 vehicle at a time (19 < 60)
        assert peak == 1

    def test_peak_concurrency_multiple_lines(self) -> None:
        """Multiple lines should sum concurrency."""
        network = Network()

        for lid in ["L1", "L2"]:
            stations = [
                Station(id=f"A_{lid}", location="A", modality="bus"),
                Station(id=f"B_{lid}", location="B", modality="bus"),
            ]
            line = Line(
                id=lid,
                modality="bus",
                stations=stations,
                travel_times=[15],
                dwell_times=[2, 2],
                frequency=60,
            )
            network.add_line(line)

        schedule = Schedule(offsets={"L1": 0, "L2": 0})

        peak = compute_peak_concurrency(schedule, network)
        # Both lines have same timing, so 2 vehicles concurrent
        assert peak >= 2

    def test_peak_concurrency_high_frequency(self) -> None:
        """High frequency line may need multiple concurrent vehicles."""
        network = Network()
        stations = [
            Station(id="A", location="A", modality="bus"),
            Station(id="B", location="B", modality="bus"),
        ]
        line = Line(
            id="L1",
            modality="bus",
            stations=stations,
            travel_times=[50],
            dwell_times=[5, 5],
            frequency=30,  # Every 30 min
        )
        network.add_line(line)

        schedule = Schedule(offsets={"L1": 0})

        peak = compute_peak_concurrency(schedule, network)
        # run_time = 50 + 5 + 5 = 60, frequency = 30
        # Need ceil(60/30) = 2 vehicles concurrent
        assert peak == 2
