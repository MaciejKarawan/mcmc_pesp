"""Peak vehicle concurrency diagnostic metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.network import T_PERIOD

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule


def compute_peak_concurrency(schedule: Schedule, network: Network) -> int:
    """Sweep-line algorithm for peak vehicle concurrency.

    Enumerates all trips for all lines, creates start/end events,
    and sweeps to find maximum concurrent vehicles.

    This is a DIAGNOSTIC metric, not part of the cost function.

    Args:
        schedule: Current schedule with line offsets.
        network: Network containing line definitions.

    Returns:
        Maximum number of vehicles in service at any point in time.
    """
    # Collect all trip start/end events
    events: list[tuple[int, int]] = []  # (time, +1 for start / -1 for end)

    for line in network.lines.values():
        offset = schedule.get_offset(line.id)
        run_time = line.total_run_time()
        n_trips = line.n_trips

        for trip_idx in range(n_trips):
            start_time = (offset + trip_idx * line.frequency) % T_PERIOD
            end_time = (start_time + run_time) % T_PERIOD

            if end_time > start_time:
                # Trip doesn't wrap around
                events.append((start_time, 1))
                events.append((end_time, -1))
            else:
                # Trip wraps around midnight — split into two intervals
                events.append((start_time, 1))
                events.append((T_PERIOD, -1))
                events.append((0, 1))
                events.append((end_time, -1))

    # Sort events by time, with ends (-1) before starts (+1) at same time
    events.sort(key=lambda e: (e[0], e[1]))

    # Sweep
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        if current > peak:
            peak = current

    return peak
