"""PESPlib format parser and converter.

This module provides functionality to load PESPlib benchmark instances
and convert them to our Network + TransferSet representation.

PESPlib format:
- Events.giv: event definitions (vertices)
- Activities.giv: activity definitions (edges with constraints)
- Config.cnf: instance parameters (period, etc.)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.network import Line, Network, Station
from src.core.transfers import Transfer, TransferParams, TransferSet

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PESPEvent:
    """Parsed event from Events.giv.

    Args:
        id: Unique identifier (1-based in PESPlib).
        type: Event type ("departure" or "arrival").
        stop_id: Station ID.
        line_id: Line ID.
        passengers: Number of passengers (weight).
        direction: Direction indicator (">" or "<").
    """

    id: int
    type: str
    stop_id: int
    line_id: int
    passengers: int
    direction: str


@dataclass(frozen=True)
class PESPActivity:
    """Parsed activity from Activities.giv.

    Args:
        id: Unique identifier.
        type: Activity type (drive, dwell, transfer, headway, sync).
        from_event: Source event ID.
        to_event: Target event ID.
        lower_bound: Minimum duration in minutes.
        upper_bound: Maximum duration in minutes.
        passengers: Number of passengers (weight).
    """

    id: int
    type: str
    from_event: int
    to_event: int
    lower_bound: int
    upper_bound: int
    passengers: int


@dataclass
class EventMapping:
    """Maps PESPlib event IDs to our line-based representation.

    Used for validation and export back to PESPlib format.
    """

    event_to_line: dict[int, str] = field(default_factory=dict)
    event_to_station_idx: dict[int, int] = field(default_factory=dict)
    event_to_type: dict[int, str] = field(default_factory=dict)

    def add_event(
        self,
        event_id: int,
        line_id: str,
        station_idx: int,
        event_type: str,
    ) -> None:
        """Add an event mapping."""
        self.event_to_line[event_id] = line_id
        self.event_to_station_idx[event_id] = station_idx
        self.event_to_type[event_id] = event_type


@dataclass
class PESPLibLoader:
    """Parser for PESPlib format instances.

    Handles loading Events.giv, Activities.giv, and Config.cnf files
    and converting them to our Network + TransferSet representation.

    Args:
        instance_path: Path to instance directory or base name.
    """

    instance_path: Path

    def __post_init__(self) -> None:
        self.instance_path = Path(self.instance_path)

    def load(self) -> tuple[Network, TransferSet, EventMapping]:
        """Load and convert the PESPlib instance.

        Returns:
            Tuple of (Network, TransferSet, EventMapping for validation).

        Raises:
            FileNotFoundError: If required files are missing.
            ValueError: If files contain invalid data.
        """
        logger.info(f"Loading PESPlib instance from {self.instance_path}")

        # Determine file paths
        events_path = self._find_file("Events.giv")
        activities_path = self._find_file("Activities.giv")
        config_path = self._find_file("Config.cnf")

        # Parse files
        events = self._parse_events(events_path)
        activities = self._parse_activities(activities_path)
        config = self._parse_config(config_path)

        period = config.get("period", 60)
        logger.info(f"Loaded {len(events)} events, {len(activities)} activities, period={period}")

        # Reconstruct lines from drive/dwell activities
        lines, event_mapping = self._reconstruct_lines(events, activities, period)
        logger.info(f"Reconstructed {len(lines)} lines")

        # Build network
        network = Network(period=period)
        for line in lines.values():
            network.add_line(line)

        # Extract transfers
        transfers = self._extract_transfers(activities, events, event_mapping, period)
        logger.info(f"Extracted {len(transfers)} transfers")

        return network, transfers, event_mapping

    def _find_file(self, filename: str) -> Path:
        """Find a file in the instance directory.

        Handles both directory-based and file-based instance paths.
        """
        # If instance_path is a directory
        if self.instance_path.is_dir():
            path = self.instance_path / filename
            if path.exists():
                return path

        # Try parent directory
        parent = self.instance_path.parent
        path = parent / filename
        if path.exists():
            return path

        # Try as-is with instance name prefix
        path = self.instance_path.with_suffix("") / filename
        if path.exists():
            return path

        raise FileNotFoundError(
            f"Could not find {filename} in {self.instance_path} or its parent"
        )

    def _parse_events(self, events_file: Path) -> dict[int, PESPEvent]:
        """Parse Events.giv file.

        Format: event-id; type; stop-id; line-id; passengers; direction
        """
        events = {}

        with open(events_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                try:
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) < 6:
                        logger.warning(f"Skipping malformed line {line_num}: {line}")
                        continue

                    event = PESPEvent(
                        id=int(parts[0]),
                        type=self._strip_quotes(parts[1]),
                        stop_id=int(parts[2]),
                        line_id=int(parts[3]),
                        passengers=int(parts[4]),
                        direction=self._strip_quotes(parts[5]),
                    )
                    events[event.id] = event

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

        return events

    def _parse_activities(self, activities_file: Path) -> dict[int, PESPActivity]:
        """Parse Activities.giv file.

        Format: activity-id; type; from-event; to-event; lower-bound; upper-bound; passengers
        """
        activities = {}

        with open(activities_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                try:
                    parts = [p.strip() for p in line.split(";")]
                    if len(parts) < 7:
                        logger.warning(f"Skipping malformed line {line_num}: {line}")
                        continue

                    activity = PESPActivity(
                        id=int(parts[0]),
                        type=self._strip_quotes(parts[1]),
                        from_event=int(parts[2]),
                        to_event=int(parts[3]),
                        lower_bound=int(parts[4]),
                        upper_bound=int(parts[5]),
                        passengers=int(parts[6]),
                    )
                    activities[activity.id] = activity

                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

        return activities

    def _parse_config(self, config_file: Path) -> dict[str, Any]:
        """Parse Config.cnf file.

        Format: key; value
        """
        config = {}

        with open(config_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split(";")]
                if len(parts) >= 2:
                    key = parts[0].lower()
                    try:
                        # Try to parse as int
                        config[key] = int(parts[1])
                    except ValueError:
                        config[key] = parts[1]

        return config

    def _strip_quotes(self, s: str) -> str:
        """Remove surrounding quotes from a string."""
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1]
        return s

    def _reconstruct_lines(
        self,
        events: dict[int, PESPEvent],
        activities: dict[int, PESPActivity],
        period: int,
    ) -> tuple[dict[str, Line], EventMapping]:
        """Reconstruct lines from sequences of drive/dwell activities.

        Algorithm:
        1. Group events by (line_id, direction)
        2. Build graph from drive/dwell activities
        3. Find start events (departures with no incoming drive)
        4. Trace each line via BFS
        """
        # Build activity graph: from_event -> [(to_event, activity)]
        graph: dict[int, list[tuple[int, PESPActivity]]] = defaultdict(list)
        incoming_drive: set[int] = set()

        for act in activities.values():
            if act.type in ("drive", "dwell"):
                graph[act.from_event].append((act.to_event, act))
                if act.type == "drive":
                    incoming_drive.add(act.to_event)

        # Find start events (departures with no incoming drive)
        start_events = [
            e for e in events.values()
            if e.type == "departure" and e.id not in incoming_drive
        ]

        logger.debug(f"Found {len(start_events)} line start events")

        lines: dict[str, Line] = {}
        event_mapping = EventMapping()
        used_events: set[int] = set()

        for start_event in start_events:
            if start_event.id in used_events:
                continue

            line, traced_events = self._trace_line(
                start_event, graph, events, period
            )

            if line is not None:
                lines[line.id] = line
                used_events.update(traced_events)

                # Build event mapping for this line
                for event_id, station_idx, event_type in traced_events:
                    event_mapping.add_event(event_id, line.id, station_idx, event_type)

        return lines, event_mapping

    def _trace_line(
        self,
        start_event: PESPEvent,
        graph: dict[int, list[tuple[int, PESPActivity]]],
        events: dict[int, PESPEvent],
        period: int,
    ) -> tuple[Line | None, list[tuple[int, int, str]]]:
        """Trace a single line from a start event.

        Returns:
            Tuple of (Line, list of (event_id, station_idx, event_type)).
        """
        # Line ID from PESPlib line_id and direction
        line_id = f"L{start_event.line_id}_{start_event.direction}"

        stations: list[Station] = []
        travel_times: list[int] = []
        dwell_times: list[int] = []
        traced_events: list[tuple[int, int, str]] = []

        current_event = start_event
        station_idx = 0

        while current_event is not None:
            # Add station for departure event
            if current_event.type == "departure":
                station = Station(
                    id=f"S{current_event.stop_id}_{line_id}_{station_idx}",
                    location=f"Stop_{current_event.stop_id}",
                    modality="transit",
                )
                stations.append(station)
                traced_events.append((current_event.id, station_idx, "departure"))

                # Find dwell to arrival or drive to next station
                next_activities = graph.get(current_event.id, [])

                drive_act = None
                for to_event, act in next_activities:
                    if act.type == "drive":
                        drive_act = act
                        break

                if drive_act is not None:
                    travel_times.append(drive_act.lower_bound)
                    arrival_event = events.get(drive_act.to_event)

                    if arrival_event and arrival_event.type == "arrival":
                        traced_events.append((arrival_event.id, station_idx, "arrival"))

                        # Find dwell from arrival
                        arrival_activities = graph.get(arrival_event.id, [])
                        dwell_act = None
                        for to_event, act in arrival_activities:
                            if act.type == "dwell":
                                dwell_act = act
                                break

                        if dwell_act is not None:
                            dwell_times.append(dwell_act.lower_bound)
                            next_departure = events.get(dwell_act.to_event)

                            if next_departure and next_departure.type == "departure":
                                station_idx += 1
                                current_event = next_departure
                            else:
                                # End of line
                                current_event = None
                        else:
                            # No dwell, end of line - add minimal dwell
                            dwell_times.append(0)
                            current_event = None
                    else:
                        # No arrival event, end of line
                        current_event = None
                else:
                    # No drive, this is the last station
                    dwell_times.append(0)  # Terminal dwell
                    current_event = None

            else:
                # Unexpected event type at start
                logger.warning(f"Unexpected event type {current_event.type} at line start")
                break

        # Validate line structure
        if len(stations) < 2:
            logger.debug(f"Line {line_id} has fewer than 2 stations, skipping")
            return None, []

        # Ensure arrays are correct length
        # travel_times should have len(stations) - 1 elements
        # dwell_times should have len(stations) elements
        while len(travel_times) < len(stations) - 1:
            travel_times.append(0)
        while len(dwell_times) < len(stations):
            dwell_times.append(0)

        travel_times = travel_times[: len(stations) - 1]
        dwell_times = dwell_times[: len(stations)]

        line = Line(
            id=line_id,
            modality="transit",
            stations=stations,
            travel_times=travel_times,
            dwell_times=dwell_times,
            frequency=period,
        )

        return line, traced_events

    def _extract_transfers(
        self,
        activities: dict[int, PESPActivity],
        events: dict[int, PESPEvent],
        event_mapping: EventMapping,
        period: int,
    ) -> TransferSet:
        """Extract transfers from 'transfer' type activities."""
        transfers: dict[str, Transfer] = {}

        for act in activities.values():
            if act.type != "transfer":
                continue

            from_event = events.get(act.from_event)
            to_event = events.get(act.to_event)

            if from_event is None or to_event is None:
                logger.warning(
                    f"Transfer {act.id} references unknown events "
                    f"({act.from_event}, {act.to_event})"
                )
                continue

            from_line = event_mapping.event_to_line.get(act.from_event)
            to_line = event_mapping.event_to_line.get(act.to_event)

            if from_line is None or to_line is None:
                logger.warning(
                    f"Transfer {act.id} references events not mapped to lines"
                )
                continue

            from_station_idx = event_mapping.event_to_station_idx.get(act.from_event, 0)
            to_station_idx = event_mapping.event_to_station_idx.get(act.to_event, 0)

            # Create transfer parameters from PESPlib bounds
            # w_min = lower_bound (minimum transfer time)
            # w_ideal = midpoint or slightly above minimum
            w_min = float(act.lower_bound)
            w_max = float(act.upper_bound) if act.upper_bound < period else float(period)
            w_ideal = min(w_min + 5.0, (w_min + w_max) / 2)

            params = TransferParams(
                w_min=w_min,
                w_ideal=w_ideal,
                w_max=w_max,
                alpha=2.0,
                beta=0.5,
                M=1e6,
            )

            transfer = Transfer(
                id=f"T{act.id}",
                from_line=from_line,
                from_station_idx=from_station_idx,
                to_line=to_line,
                to_station_idx=to_station_idx,
                walk_time=float(act.lower_bound),
                params=params,
                weight=float(act.passengers) if act.passengers > 0 else 1.0,
            )
            transfers[transfer.id] = transfer

        return TransferSet(transfers=transfers)


def load_pesplib_instance(
    instance_path: str | Path,
) -> tuple[Network, TransferSet, EventMapping]:
    """Convenience function to load a PESPlib instance.

    Args:
        instance_path: Path to instance directory.

    Returns:
        Tuple of (Network, TransferSet, EventMapping).
    """
    loader = PESPLibLoader(Path(instance_path))
    return loader.load()


def parse_tim_file(tim_path: Path) -> dict[int, int]:
    """Parse a PESPlib .tim solution file.

    Args:
        tim_path: Path to .tim file.

    Returns:
        Dictionary mapping event_id to time.
    """
    times = {}

    with open(tim_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(";")]
            if len(parts) >= 2:
                try:
                    event_id = int(parts[0])
                    time = int(parts[1])
                    times[event_id] = time
                except ValueError:
                    continue

    return times


def export_to_tim(
    schedule: "Schedule",
    network: Network,
    event_mapping: EventMapping,
) -> str:
    """Export our Schedule to PESPlib .tim format.

    Args:
        schedule: Our schedule with line offsets.
        network: Network definition.
        event_mapping: Mapping from event IDs to our representation.

    Returns:
        String content of .tim file.
    """
    from src.core.schedule import Schedule

    lines = ["# event-id; time"]

    for event_id, line_id in event_mapping.event_to_line.items():
        station_idx = event_mapping.event_to_station_idx[event_id]
        event_type = event_mapping.event_to_type[event_id]

        line = network.get_line(line_id)
        offset = schedule.get_offset(line_id)

        # Compute time for this event
        if event_type == "departure":
            time = offset + line.departure_offset(station_idx)
        else:  # arrival
            time = offset + line.arrival_offset(station_idx)

        time = time % network.period
        lines.append(f"{event_id}; {time}")

    return "\n".join(lines)


def compute_pesplib_objective(
    schedule: "Schedule",
    network: Network,
    activities: dict[int, PESPActivity],
    event_mapping: EventMapping,
) -> float:
    """Compute the PESPlib objective value (weighted sum of durations).

    Args:
        schedule: Our schedule.
        network: Network definition.
        activities: Parsed activities.
        event_mapping: Event mapping.

    Returns:
        PESPlib objective value.
    """
    # First compute all event times
    event_times: dict[int, int] = {}

    for event_id, line_id in event_mapping.event_to_line.items():
        station_idx = event_mapping.event_to_station_idx[event_id]
        event_type = event_mapping.event_to_type[event_id]

        line = network.get_line(line_id)
        offset = schedule.get_offset(line_id)

        if event_type == "departure":
            time = offset + line.departure_offset(station_idx)
        else:
            time = offset + line.arrival_offset(station_idx)

        event_times[event_id] = time % network.period

    # Compute objective
    objective = 0.0
    period = network.period

    for act in activities.values():
        if act.from_event in event_times and act.to_event in event_times:
            duration = (event_times[act.to_event] - event_times[act.from_event]) % period
            objective += act.passengers * duration

    return objective
