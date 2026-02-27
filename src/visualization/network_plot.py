"""Network and schedule visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import TransferSet


# Color schemes
LINE_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
]


@dataclass
class NetworkPlotter:
    """Visualize transport network schedules.

    Creates time-space diagrams showing line departures and transfers.
    """

    network: Network
    transfers: TransferSet
    figsize: tuple[int, int] = (14, 8)

    def _get_line_color(self, line_idx: int) -> str:
        """Get color for a line by index."""
        return LINE_COLORS[line_idx % len(LINE_COLORS)]

    def _get_location_positions(self) -> dict[str, float]:
        """Assign y-positions to locations for plotting."""
        locations = set()
        for line in self.network.lines.values():
            for station in line.stations:
                locations.add(station.location)

        sorted_locs = sorted(locations)
        return {loc: i for i, loc in enumerate(sorted_locs)}

    def plot_schedule(
        self,
        schedule: Schedule,
        ax: Axes | None = None,
        title: str | None = None,
        show_transfers: bool = True,
        highlight_transfer: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot a time-space diagram of the schedule.

        Args:
            schedule: Schedule to visualize.
            ax: Matplotlib axes to use. If None, creates new figure.
            title: Plot title.
            show_transfers: Whether to show transfer connections.
            highlight_transfer: ID of transfer to highlight.

        Returns:
            Tuple of (figure, axes).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        loc_positions = self._get_location_positions()
        period = self.network.period

        # Plot each line
        legend_handles = []
        for idx, (line_id, line) in enumerate(self.network.lines.items()):
            color = self._get_line_color(idx)
            offset = schedule.get_offset(line_id)

            times = []
            positions = []

            for i, station in enumerate(line.stations):
                # Arrival time
                arr_time = (offset + line.arrival_offset(i)) % period
                # Departure time
                dep_time = (offset + line.departure_offset(i)) % period

                y_pos = loc_positions[station.location]

                # Add arrival point
                times.append(arr_time)
                positions.append(y_pos)

                # Add departure point (if different)
                if dep_time != arr_time:
                    times.append(dep_time)
                    positions.append(y_pos)

            # Plot line trajectory
            ax.plot(
                times,
                positions,
                "o-",
                color=color,
                linewidth=2,
                markersize=8,
                label=line_id,
            )

            # Add line label at midpoint
            mid_idx = len(times) // 2
            ax.annotate(
                line_id,
                (times[mid_idx], positions[mid_idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

            legend_handles.append(
                mpatches.Patch(color=color, label=f"{line_id} (offset={offset})")
            )

        # Plot transfers
        if show_transfers:
            for transfer in self.transfers:
                from_line = self.network.get_line(transfer.from_line)
                to_line = self.network.get_line(transfer.to_line)

                from_offset = schedule.get_offset(transfer.from_line)
                to_offset = schedule.get_offset(transfer.to_line)

                # Arrival time at transfer point
                arr_time = (
                    from_offset + from_line.arrival_offset(transfer.from_station_idx)
                ) % period

                # Departure time from transfer point
                dep_time = (
                    to_offset + to_line.departure_offset(transfer.to_station_idx)
                ) % period

                # Waiting time
                wait_time = transfer.compute_waiting_time(schedule, self.network)

                # Get y positions
                from_station = from_line.stations[transfer.from_station_idx]
                to_station = to_line.stations[transfer.to_station_idx]
                from_y = loc_positions[from_station.location]
                to_y = loc_positions[to_station.location]

                # Determine color based on waiting time quality
                w_min = transfer.params.w_min
                w_ideal = transfer.params.w_ideal

                if wait_time < w_min:
                    transfer_color = "red"
                    linestyle = "--"
                elif wait_time <= w_ideal:
                    transfer_color = "green"
                    linestyle = "-"
                else:
                    transfer_color = "orange"
                    linestyle = "-"

                # Highlight specific transfer
                linewidth = 3 if transfer.id == highlight_transfer else 1.5
                alpha = 1.0 if transfer.id == highlight_transfer else 0.6

                # Draw transfer arc
                ax.annotate(
                    "",
                    xy=(dep_time, to_y),
                    xytext=(arr_time, from_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=transfer_color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                    ),
                )

                # Add waiting time label
                mid_x = (arr_time + dep_time) / 2
                mid_y = (from_y + to_y) / 2
                ax.annotate(
                    f"{wait_time:.0f}m",
                    (mid_x, mid_y),
                    fontsize=9,
                    color=transfer_color,
                    fontweight="bold",
                    ha="center",
                )

        # Configure axes
        ax.set_xlabel("Time (minutes)", fontsize=12)
        ax.set_ylabel("Location", fontsize=12)
        ax.set_xlim(-2, period + 2)
        ax.set_ylim(-0.5, len(loc_positions) - 0.5)

        # Set y-ticks to location names
        ax.set_yticks(list(loc_positions.values()))
        ax.set_yticklabels(list(loc_positions.keys()))

        # Add grid
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=period, color="gray", linestyle=":", alpha=0.5)

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        # Legend
        ax.legend(handles=legend_handles, loc="upper right")

        return fig, ax

    def plot_cost_landscape(
        self,
        schedule: Schedule,
        line_id: str,
        cost_fn,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot cost as function of one line's offset.

        Args:
            schedule: Base schedule.
            line_id: Line to vary.
            cost_fn: Cost function.
            ax: Matplotlib axes.

        Returns:
            Tuple of (figure, axes).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.get_figure()

        period = self.network.period
        offsets = np.arange(period)
        costs = []

        for offset in offsets:
            test_schedule = schedule.copy()
            test_schedule.set_offset(line_id, offset)
            cost = cost_fn(test_schedule, self.network)
            costs.append(cost)

        current_offset = schedule.get_offset(line_id)
        current_cost = cost_fn(schedule, self.network)

        ax.plot(offsets, costs, "b-", linewidth=2, label="Cost")
        ax.axvline(x=current_offset, color="red", linestyle="--", label="Current")
        ax.scatter([current_offset], [current_cost], color="red", s=100, zorder=5)

        # Mark minimum
        min_idx = np.argmin(costs)
        ax.scatter([min_idx], [costs[min_idx]], color="green", s=100, zorder=5,
                   marker="*", label=f"Minimum (offset={min_idx})")

        ax.set_xlabel(f"Offset of {line_id} (minutes)")
        ax.set_ylabel("Cost")
        ax.set_title(f"Cost Landscape for {line_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax
