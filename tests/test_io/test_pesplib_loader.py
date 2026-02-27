"""Tests for PESPlib format loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.io.pesplib_loader import (
    EventMapping,
    PESPActivity,
    PESPEvent,
    PESPLibLoader,
    export_to_tim,
    load_pesplib_instance,
    parse_tim_file,
)


class TestPESPEvent:
    """Tests for PESPEvent dataclass."""

    def test_creation(self) -> None:
        event = PESPEvent(
            id=1,
            type="departure",
            stop_id=101,
            line_id=1,
            passengers=500,
            direction=">",
        )
        assert event.id == 1
        assert event.type == "departure"
        assert event.stop_id == 101
        assert event.line_id == 1
        assert event.passengers == 500
        assert event.direction == ">"

    def test_is_frozen(self) -> None:
        event = PESPEvent(1, "departure", 101, 1, 500, ">")
        with pytest.raises(AttributeError):
            event.id = 2  # type: ignore


class TestPESPActivity:
    """Tests for PESPActivity dataclass."""

    def test_creation(self) -> None:
        activity = PESPActivity(
            id=1,
            type="drive",
            from_event=1,
            to_event=2,
            lower_bound=10,
            upper_bound=10,
            passengers=500,
        )
        assert activity.id == 1
        assert activity.type == "drive"
        assert activity.from_event == 1
        assert activity.to_event == 2
        assert activity.lower_bound == 10
        assert activity.upper_bound == 10
        assert activity.passengers == 500


class TestEventMapping:
    """Tests for EventMapping."""

    def test_add_event(self) -> None:
        mapping = EventMapping()
        mapping.add_event(1, "L1_>", 0, "departure")

        assert mapping.event_to_line[1] == "L1_>"
        assert mapping.event_to_station_idx[1] == 0
        assert mapping.event_to_type[1] == "departure"

    def test_multiple_events(self) -> None:
        mapping = EventMapping()
        mapping.add_event(1, "L1_>", 0, "departure")
        mapping.add_event(2, "L1_>", 0, "arrival")
        mapping.add_event(3, "L1_>", 1, "departure")

        assert len(mapping.event_to_line) == 3
        assert mapping.event_to_station_idx[3] == 1


@pytest.fixture
def simple_pesplib_instance(tmp_path: Path) -> Path:
    """Create a simple PESPlib instance for testing."""
    instance_dir = tmp_path / "test_instance"
    instance_dir.mkdir()

    # Events.giv - simple two-station line
    events_content = """# event-id; type; stop-id; line-id; passengers; direction
1; "departure"; 101; 1; 500; ">"
2; "arrival"; 102; 1; 500; ">"
3; "departure"; 102; 1; 480; ">"
4; "arrival"; 103; 1; 480; ">"
"""
    (instance_dir / "Events.giv").write_text(events_content, encoding="utf-8")

    # Activities.giv
    activities_content = """# activity-id; type; from-event; to-event; lower-bound; upper-bound; passengers
1; "drive"; 1; 2; 10; 10; 500
2; "dwell"; 2; 3; 2; 5; 480
3; "drive"; 3; 4; 15; 15; 480
"""
    (instance_dir / "Activities.giv").write_text(activities_content, encoding="utf-8")

    # Config.cnf
    config_content = """period; 60
"""
    (instance_dir / "Config.cnf").write_text(config_content, encoding="utf-8")

    return instance_dir


@pytest.fixture
def two_line_instance_with_transfer(tmp_path: Path) -> Path:
    """Create a PESPlib instance with two lines and a transfer."""
    instance_dir = tmp_path / "transfer_instance"
    instance_dir.mkdir()

    # Events.giv - two lines meeting at stop 102
    events_content = """# event-id; type; stop-id; line-id; passengers; direction
1; "departure"; 101; 1; 500; ">"
2; "arrival"; 102; 1; 500; ">"
3; "departure"; 102; 1; 450; ">"
4; "arrival"; 103; 1; 450; ">"
5; "departure"; 201; 2; 300; ">"
6; "arrival"; 102; 2; 300; ">"
7; "departure"; 102; 2; 350; ">"
8; "arrival"; 202; 2; 350; ">"
"""
    (instance_dir / "Events.giv").write_text(events_content, encoding="utf-8")

    # Activities.giv - including a transfer at stop 102
    activities_content = """# activity-id; type; from-event; to-event; lower-bound; upper-bound; passengers
1; "drive"; 1; 2; 10; 10; 500
2; "dwell"; 2; 3; 2; 5; 450
3; "drive"; 3; 4; 12; 12; 450
4; "drive"; 5; 6; 8; 8; 300
5; "dwell"; 6; 7; 2; 5; 350
6; "drive"; 7; 8; 15; 15; 350
7; "transfer"; 2; 7; 3; 30; 100
"""
    (instance_dir / "Activities.giv").write_text(activities_content, encoding="utf-8")

    # Config.cnf
    config_content = """period; 60
"""
    (instance_dir / "Config.cnf").write_text(config_content, encoding="utf-8")

    return instance_dir


class TestPESPLibLoader:
    """Tests for PESPLibLoader."""

    def test_parse_events_basic(self, simple_pesplib_instance: Path) -> None:
        loader = PESPLibLoader(simple_pesplib_instance)
        events_path = simple_pesplib_instance / "Events.giv"
        events = loader._parse_events(events_path)

        assert len(events) == 4
        assert events[1].type == "departure"
        assert events[2].type == "arrival"
        assert events[1].stop_id == 101
        assert events[2].stop_id == 102

    def test_parse_events_handles_quotes(self, tmp_path: Path) -> None:
        events_file = tmp_path / "Events.giv"
        events_file.write_text(
            '1; "departure"; 101; 1; 500; ">"\n'
            "2; 'arrival'; 102; 1; 500; '>'\n",
            encoding="utf-8",
        )

        loader = PESPLibLoader(tmp_path)
        events = loader._parse_events(events_file)

        assert events[1].type == "departure"
        assert events[2].type == "arrival"
        assert events[1].direction == ">"

    def test_parse_activities_all_types(self, tmp_path: Path) -> None:
        activities_file = tmp_path / "Activities.giv"
        activities_file.write_text(
            '1; "drive"; 1; 2; 10; 10; 500\n'
            '2; "dwell"; 2; 3; 2; 5; 480\n'
            '3; "transfer"; 4; 5; 3; 60; 100\n'
            '4; "headway"; 1; 6; 5; 60; 0\n'
            '5; "sync"; 7; 8; 0; 0; 0\n',
            encoding="utf-8",
        )

        loader = PESPLibLoader(tmp_path)
        activities = loader._parse_activities(activities_file)

        assert len(activities) == 5
        assert activities[1].type == "drive"
        assert activities[2].type == "dwell"
        assert activities[3].type == "transfer"
        assert activities[4].type == "headway"
        assert activities[5].type == "sync"

    def test_parse_config_period(self, simple_pesplib_instance: Path) -> None:
        loader = PESPLibLoader(simple_pesplib_instance)
        config_path = simple_pesplib_instance / "Config.cnf"
        config = loader._parse_config(config_path)

        assert config["period"] == 60

    def test_full_load_simple_instance(self, simple_pesplib_instance: Path) -> None:
        network, transfers, event_mapping = load_pesplib_instance(simple_pesplib_instance)

        assert network.period == 60
        assert len(network.lines) >= 1

        # Should have created event mappings
        assert len(event_mapping.event_to_line) > 0

    def test_load_with_transfer(self, two_line_instance_with_transfer: Path) -> None:
        network, transfers, event_mapping = load_pesplib_instance(
            two_line_instance_with_transfer
        )

        assert network.period == 60
        assert len(network.lines) == 2
        assert len(transfers) == 1

        # Check transfer was extracted correctly
        transfer = list(transfers)[0]
        assert transfer.walk_time == 3.0  # lower_bound of transfer activity
        assert transfer.weight == 100.0

    def test_invalid_file_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pesplib_instance(tmp_path / "nonexistent")

    def test_skip_malformed_lines(self, tmp_path: Path) -> None:
        instance_dir = tmp_path / "malformed"
        instance_dir.mkdir()

        # Events with some malformed lines
        events_content = """# comment
1; "departure"; 101; 1; 500; ">"
malformed line without semicolons
2; "arrival"; 102; 1; 500; ">"
"""
        (instance_dir / "Events.giv").write_text(events_content, encoding="utf-8")
        (instance_dir / "Activities.giv").write_text(
            '1; "drive"; 1; 2; 10; 10; 500\n', encoding="utf-8"
        )
        (instance_dir / "Config.cnf").write_text("period; 60\n", encoding="utf-8")

        loader = PESPLibLoader(instance_dir)
        events = loader._parse_events(instance_dir / "Events.giv")

        # Should parse valid lines and skip malformed
        assert len(events) == 2
        assert 1 in events
        assert 2 in events


class TestTimFileParsing:
    """Tests for .tim file parsing."""

    def test_parse_tim_file(self, tmp_path: Path) -> None:
        tim_file = tmp_path / "solution.tim"
        tim_file.write_text(
            "# event-id; time\n1; 0\n2; 10\n3; 12\n4; 27\n",
            encoding="utf-8",
        )

        times = parse_tim_file(tim_file)

        assert times[1] == 0
        assert times[2] == 10
        assert times[3] == 12
        assert times[4] == 27

    def test_parse_tim_skips_comments(self, tmp_path: Path) -> None:
        tim_file = tmp_path / "solution.tim"
        tim_file.write_text(
            "# comment\n1; 0\n# another comment\n2; 10\n",
            encoding="utf-8",
        )

        times = parse_tim_file(tim_file)
        assert len(times) == 2


class TestExportToTim:
    """Tests for exporting to .tim format."""

    def test_export_basic(self, two_line_instance_with_transfer: Path) -> None:
        from src.core.schedule import Schedule

        network, transfers, event_mapping = load_pesplib_instance(
            two_line_instance_with_transfer
        )

        # Create a schedule
        schedule = Schedule.from_network(network, default_offset=0)

        # Export
        tim_content = export_to_tim(schedule, network, event_mapping)

        # Should contain header and event lines
        assert "# event-id; time" in tim_content
        lines = tim_content.strip().split("\n")
        assert len(lines) > 1  # At least header + some events


class TestLineReconstruction:
    """Tests for line reconstruction algorithm."""

    def test_reconstructs_correct_stations(
        self, simple_pesplib_instance: Path
    ) -> None:
        network, _, _ = load_pesplib_instance(simple_pesplib_instance)

        # Should have reconstructed at least one line
        assert len(network.lines) >= 1

        # Get the first line
        line = list(network.lines.values())[0]

        # Should have multiple stations
        assert len(line.stations) >= 2

    def test_reconstructs_travel_times(
        self, simple_pesplib_instance: Path
    ) -> None:
        network, _, _ = load_pesplib_instance(simple_pesplib_instance)

        line = list(network.lines.values())[0]

        # travel_times length should be stations - 1
        assert len(line.travel_times) == len(line.stations) - 1

        # Travel times should be positive (from the test instance)
        assert all(t >= 0 for t in line.travel_times)

    def test_reconstructs_dwell_times(
        self, simple_pesplib_instance: Path
    ) -> None:
        network, _, _ = load_pesplib_instance(simple_pesplib_instance)

        line = list(network.lines.values())[0]

        # dwell_times length should equal stations
        assert len(line.dwell_times) == len(line.stations)


class TestTransferExtraction:
    """Tests for transfer extraction."""

    def test_extracts_transfer_params(
        self, two_line_instance_with_transfer: Path
    ) -> None:
        _, transfers, _ = load_pesplib_instance(two_line_instance_with_transfer)

        assert len(transfers) == 1
        transfer = list(transfers)[0]

        # w_min should be the lower bound
        assert transfer.params.w_min == 3.0

        # walk_time should also be lower bound
        assert transfer.walk_time == 3.0

    def test_transfer_references_correct_lines(
        self, two_line_instance_with_transfer: Path
    ) -> None:
        network, transfers, _ = load_pesplib_instance(two_line_instance_with_transfer)

        transfer = list(transfers)[0]

        # from_line and to_line should exist in network
        assert transfer.from_line in network.lines
        assert transfer.to_line in network.lines
