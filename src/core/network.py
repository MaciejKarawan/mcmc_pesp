"""Core network structures: Station, Line, Network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

T_PERIOD = 1440  # Common scheduling period: minutes in a day


@dataclass(frozen=True)
class Station:
    """A station in the transport network.

    Args:
        id: Unique identifier for the station.
        location: Physical location identifier (e.g., "Warsaw Central").
        modality: Transport mode serving this station (e.g., "IC", "bus").
    """

    id: str
    location: str
    modality: str

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class Line:
    """A transport line with periodic service.

    A line represents a sequence of stations served by a particular mode
    of transport with fixed travel times and dwell times.

    Args:
        id: Unique identifier for the line.
        modality: Transport mode (e.g., "IC", "regional", "bus").
        stations: Ordered sequence of stations served by this line.
        travel_times: Travel times in minutes between consecutive stations.
            Length must be len(stations) - 1.
        dwell_times: Minimum dwell times in minutes at each station.
            Length must be len(stations).
        frequency: Service frequency in minutes (e.g., 60 for hourly).
    """

    id: str
    modality: str
    stations: list[Station]
    travel_times: list[int]
    dwell_times: list[int]
    frequency: int

    def __post_init__(self) -> None:
        if len(self.travel_times) != len(self.stations) - 1:
            raise ValueError(
                f"travel_times must have length {len(self.stations) - 1}, "
                f"got {len(self.travel_times)}"
            )
        if len(self.dwell_times) != len(self.stations):
            raise ValueError(
                f"dwell_times must have length {len(self.stations)}, "
                f"got {len(self.dwell_times)}"
            )
        if self.frequency <= 0:
            raise ValueError(f"frequency must be positive, got {self.frequency}")

    @property
    def n_trips(self) -> int:
        """Number of trips per period T."""
        return T_PERIOD // self.frequency

    def __hash__(self) -> int:
        return hash(self.id)

    def total_travel_time(self) -> int:
        """Total travel time from first to last station (excluding dwell times)."""
        return sum(self.travel_times)

    def total_run_time(self) -> int:
        """Total run time including all dwell times."""
        return sum(self.travel_times) + sum(self.dwell_times)

    def arrival_offset(self, station_index: int) -> int:
        """Compute arrival time offset at a station relative to line offset.

        Args:
            station_index: Index of the station (0-based).

        Returns:
            Minutes from line offset to arrival at station.
        """
        if station_index < 0 or station_index >= len(self.stations):
            raise IndexError(f"station_index {station_index} out of range")
        if station_index == 0:
            return 0
        return (
            sum(self.travel_times[:station_index])
            + sum(self.dwell_times[:station_index])
        )

    def departure_offset(self, station_index: int) -> int:
        """Compute departure time offset at a station relative to line offset.

        Args:
            station_index: Index of the station (0-based).

        Returns:
            Minutes from line offset to departure from station.
        """
        return self.arrival_offset(station_index) + self.dwell_times[station_index]


@dataclass
class Network:
    """A transport network consisting of lines and their relationships.

    Args:
        lines: Dictionary mapping line IDs to Line objects.
        period: Common scheduling period T (default T_PERIOD=1440).
    """

    lines: dict[str, Line] = field(default_factory=dict)
    period: int = T_PERIOD

    def add_line(self, line: Line) -> None:
        """Add a line to the network.

        Args:
            line: Line to add.

        Raises:
            ValueError: If line with same ID already exists.
        """
        if line.id in self.lines:
            raise ValueError(f"Line with id '{line.id}' already exists")
        self.lines[line.id] = line

    def get_line(self, line_id: str) -> Line:
        """Get a line by its ID.

        Args:
            line_id: ID of the line to retrieve.

        Returns:
            The requested Line object.

        Raises:
            KeyError: If line does not exist.
        """
        return self.lines[line_id]

    def get_lines_at_location(self, location: str) -> list[Line]:
        """Get all lines that serve a given location.

        Args:
            location: Physical location identifier.

        Returns:
            List of lines serving this location.
        """
        result = []
        for line in self.lines.values():
            for station in line.stations:
                if station.location == location:
                    result.append(line)
                    break
        return result

    def get_station_index(self, line: Line, location: str) -> int | None:
        """Find the index of a station in a line by location.

        Args:
            line: The line to search.
            location: Physical location to find.

        Returns:
            Station index if found, None otherwise.
        """
        for i, station in enumerate(line.stations):
            if station.location == location:
                return i
        return None

    @property
    def line_ids(self) -> list[str]:
        """List of all line IDs in the network."""
        return list(self.lines.keys())

    def __len__(self) -> int:
        return len(self.lines)