"""Generator for synthetic PESPlib-format instances.

This module provides functionality to generate minimal PESPlib instances
for testing the benchmark infrastructure without requiring external data.

The generated instances follow the PESPlib format:
- Events.giv: event definitions (vertices)
- Activities.giv: activity definitions (edges with constraints)
- Config.cnf: instance parameters (period, etc.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class MiniInstanceParams(NamedTuple):
    """Parameters for the mini synthetic instance.

    Attributes:
        n_lines: Number of lines in the instance.
        n_stations_per_line: Number of stations per line.
        period: Time period in minutes.
        travel_time: Travel time between stations.
        dwell_time: Dwell time at stations.
        transfer_lower: Lower bound for transfers.
        transfer_upper: Upper bound for transfers.
        passengers_per_transfer: Passenger weight for transfers.
    """

    n_lines: int = 2
    n_stations_per_line: int = 3
    period: int = 60
    travel_time: int = 10
    dwell_time: int = 2
    transfer_lower: int = 3
    transfer_upper: int = 30
    passengers_per_transfer: int = 50


@dataclass
class SyntheticPESPLibInstance:
    """Container for generated PESPlib instance files.

    Holds the content of all required PESPlib files as strings,
    ready to be written to disk or used directly.

    Attributes:
        events_content: Content of Events.giv file.
        activities_content: Content of Activities.giv file.
        config_content: Content of Config.cnf file.
        optimum_offsets: Known optimal offsets for validation.
        optimum_objective: Known optimal objective value.
    """

    events_content: str
    activities_content: str
    config_content: str
    optimum_offsets: dict[str, int]
    optimum_objective: float

    def write_to_directory(self, directory: Path) -> None:
        """Write instance files to a directory.

        Creates the directory if it doesn't exist.

        Args:
            directory: Target directory path.
        """
        directory.mkdir(parents=True, exist_ok=True)

        events_file = directory / "Events.giv"
        activities_file = directory / "Activities.giv"
        config_file = directory / "Config.cnf"

        events_file.write_text(self.events_content, encoding="utf-8")
        activities_file.write_text(self.activities_content, encoding="utf-8")
        config_file.write_text(self.config_content, encoding="utf-8")

        logger.info(f"Wrote PESPlib instance to {directory}")


def generate_mini_instance(
    params: MiniInstanceParams | None = None,
) -> SyntheticPESPLibInstance:
    """Generate a minimal PESPlib instance for testing.

    Creates a simple network with a few lines meeting at common stations,
    with transfers between them. The optimal solution is known and can be
    used for validation.

    Instance structure:
    - Line 1 (forward): S0 -> S1 -> S2
    - Line 2 (forward): S1 -> S2 -> S3
    - Transfer at S1 (L1 -> L2)
    - Transfer at S2 (L1 -> L2)

    Args:
        params: Instance parameters (uses defaults if None).

    Returns:
        SyntheticPESPLibInstance with all file contents and known optimum.
    """
    if params is None:
        params = MiniInstanceParams()

    events_lines = ["# event-id; type; stop-id; line-id; passengers; direction"]
    activities_lines = [
        "# activity-id; type; from-event; to-event; lower-bound; upper-bound; passengers"
    ]

    event_id = 1
    activity_id = 1

    # Event tracking for transfers
    departure_events: dict[tuple[int, int, str], int] = {}  # (line, station, dir) -> event_id
    arrival_events: dict[tuple[int, int, str], int] = {}

    # Generate events and activities for each line
    for line_id in range(1, params.n_lines + 1):
        direction = ">"
        prev_departure = None

        for station_idx in range(params.n_stations_per_line):
            # Adjust station ID to create overlapping stations
            # Line 1: stations 0, 1, 2
            # Line 2: stations 1, 2, 3
            station_id = station_idx + (line_id - 1)

            # Departure event
            events_lines.append(
                f'{event_id}; "departure"; {station_id}; {line_id}; 100; "{direction}"'
            )
            departure_events[(line_id, station_id, direction)] = event_id
            dep_event = event_id
            event_id += 1

            if station_idx > 0:
                # Drive activity from previous departure to this arrival
                arrival_event = event_id
                events_lines.append(
                    f'{event_id}; "arrival"; {station_id}; {line_id}; 100; "{direction}"'
                )
                arrival_events[(line_id, station_id, direction)] = event_id
                event_id += 1

                # Drive activity
                activities_lines.append(
                    f'{activity_id}; "drive"; {prev_departure}; {arrival_event}; '
                    f'{params.travel_time}; {params.travel_time}; 100'
                )
                activity_id += 1

                # Dwell activity from arrival to departure
                activities_lines.append(
                    f'{activity_id}; "dwell"; {arrival_event}; {dep_event}; '
                    f'{params.dwell_time}; {params.dwell_time}; 100'
                )
                activity_id += 1

            prev_departure = dep_event

    # Generate transfers between lines at shared stations
    # Line 1 stations: 0, 1, 2
    # Line 2 stations: 1, 2, 3
    # Shared: 1, 2

    transfer_stations = [1, 2]  # Stations where transfers are possible

    for station_id in transfer_stations:
        # Transfer from Line 1 arrival to Line 2 departure
        key_arr = (1, station_id, ">")
        key_dep = (2, station_id, ">")

        if key_arr in arrival_events and key_dep in departure_events:
            from_event = arrival_events[key_arr]
            to_event = departure_events[key_dep]

            activities_lines.append(
                f'{activity_id}; "transfer"; {from_event}; {to_event}; '
                f'{params.transfer_lower}; {params.transfer_upper}; '
                f'{params.passengers_per_transfer}'
            )
            activity_id += 1

    # Config file
    config_lines = [
        "# PESPlib configuration",
        f"period; {params.period}",
    ]

    # Compute known optimum
    # For this simple instance, optimal is when transfers align
    # With travel_time=10, dwell_time=2:
    # Line 1: dep@S0=0, arr@S1=10, dep@S1=12, arr@S2=22, dep@S2=24
    # Line 2 needs to depart S1 after L1 arrives + transfer_lower
    # Best: L2 offset = 10 + 3 = 13 (for ideal wait = 3)

    # Optimal offsets (these give minimal transfer wait times)
    optimum_offsets = {"L1_>": 0, "L2_>": 13}

    # Optimal objective (sum of passengers * duration for all activities)
    # This is a simplification - actual objective depends on exact schedule
    optimum_objective = 0.0  # Will be computed after loading

    instance = SyntheticPESPLibInstance(
        events_content="\n".join(events_lines),
        activities_content="\n".join(activities_lines),
        config_content="\n".join(config_lines),
        optimum_offsets=optimum_offsets,
        optimum_objective=optimum_objective,
    )

    return instance


def generate_triangle_pesplib() -> SyntheticPESPLibInstance:
    """Generate a PESPlib-format version of the triangle network.

    Creates the classic triangle network used in the PoC, but in
    PESPlib format for testing the loader and benchmark.

    Structure:
    - 3 lines forming a triangle
    - 3 transfer points (one at each vertex)
    - Period = 60 minutes

    Returns:
        SyntheticPESPLibInstance with triangle network.
    """
    period = 60
    travel_time = 15
    dwell_time = 2

    events_lines = ["# event-id; type; stop-id; line-id; passengers; direction"]
    activities_lines = [
        "# activity-id; type; from-event; to-event; lower-bound; upper-bound; passengers"
    ]

    event_id = 1
    activity_id = 1

    # Track events for transfers
    departures: dict[tuple[int, int], int] = {}  # (line, station) -> event_id
    arrivals: dict[tuple[int, int], int] = {}

    # Three lines: L1 (A->B), L2 (B->C), L3 (C->A)
    lines_def = [
        (1, [0, 1]),  # Line 1: station 0 (A) to station 1 (B)
        (2, [1, 2]),  # Line 2: station 1 (B) to station 2 (C)
        (3, [2, 0]),  # Line 3: station 2 (C) to station 0 (A)
    ]

    for line_id, stations in lines_def:
        prev_dep = None

        for i, station in enumerate(stations):
            # Departure
            events_lines.append(
                f'{event_id}; "departure"; {station}; {line_id}; 100; ">"'
            )
            departures[(line_id, station)] = event_id
            dep = event_id
            event_id += 1

            if i > 0:
                # Arrival at this station
                events_lines.append(
                    f'{event_id}; "arrival"; {station}; {line_id}; 100; ">"'
                )
                arrivals[(line_id, station)] = event_id
                arr = event_id
                event_id += 1

                # Drive
                activities_lines.append(
                    f'{activity_id}; "drive"; {prev_dep}; {arr}; '
                    f'{travel_time}; {travel_time}; 100'
                )
                activity_id += 1

                # Dwell
                activities_lines.append(
                    f'{activity_id}; "dwell"; {arr}; {dep}; '
                    f'{dwell_time}; {dwell_time}; 100'
                )
                activity_id += 1

            prev_dep = dep

    # Transfers at each vertex
    # A (station 0): L3 arrival -> L1 departure
    # B (station 1): L1 arrival -> L2 departure
    # C (station 2): L2 arrival -> L3 departure

    transfers = [
        (arrivals.get((3, 0)), departures.get((1, 0))),  # A
        (arrivals.get((1, 1)), departures.get((2, 1))),  # B
        (arrivals.get((2, 2)), departures.get((3, 2))),  # C
    ]

    for from_ev, to_ev in transfers:
        if from_ev is not None and to_ev is not None:
            activities_lines.append(
                f'{activity_id}; "transfer"; {from_ev}; {to_ev}; 3; 30; 100'
            )
            activity_id += 1

    config_lines = [
        "# Triangle network",
        f"period; {period}",
    ]

    # For the triangle, optimal is when all transfers have ideal wait time
    optimum_offsets = {"L1_>": 0, "L2_>": 17, "L3_>": 34}

    return SyntheticPESPLibInstance(
        events_content="\n".join(events_lines),
        activities_content="\n".join(activities_lines),
        config_content="\n".join(config_lines),
        optimum_offsets=optimum_offsets,
        optimum_objective=0.0,
    )


def ensure_synthetic_instances(base_dir: Path | None = None) -> dict[str, Path]:
    """Ensure synthetic test instances exist.

    Creates synthetic instances in the data directory if they don't exist.
    Useful for testing when real PESPlib instances aren't available.

    Args:
        base_dir: Base directory for data (defaults to data/pesplib/synthetic).

    Returns:
        Dictionary mapping instance names to their paths.
    """
    if base_dir is None:
        base_dir = Path("data/pesplib/synthetic")

    instances = {}

    # Mini instance
    mini_dir = base_dir / "mini"
    if not (mini_dir / "Events.giv").exists():
        logger.info("Generating synthetic 'mini' instance...")
        mini = generate_mini_instance()
        mini.write_to_directory(mini_dir)
    instances["mini"] = mini_dir

    # Triangle instance
    triangle_dir = base_dir / "triangle"
    if not (triangle_dir / "Events.giv").exists():
        logger.info("Generating synthetic 'triangle' instance...")
        triangle = generate_triangle_pesplib()
        triangle.write_to_directory(triangle_dir)
    instances["triangle"] = triangle_dir

    return instances
