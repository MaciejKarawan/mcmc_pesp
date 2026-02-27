"""Schedule representation for MCMC state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.network import T_PERIOD

if TYPE_CHECKING:
    from src.core.network import Network


@dataclass
class Schedule:
    """A schedule representing line offsets - the MCMC state.

    The schedule is a mapping from line IDs to their time offsets within
    the period. An offset of 0 means the line departs from its first
    station at time 0, offset of 5 means it departs at minute 5, etc.

    All arithmetic is performed modulo the period to maintain periodicity.

    Args:
        offsets: Dictionary mapping line_id to offset in minutes.
        period: Common scheduling period T for modular arithmetic (default T_PERIOD=1440).
    """

    offsets: dict[str, int] = field(default_factory=dict)
    period: int = T_PERIOD

    def copy(self) -> Schedule:
        """Create a deep copy of this schedule.

        Returns:
            A new Schedule with copied offsets.
        """
        return Schedule(offsets=dict(self.offsets), period=self.period)

    def get_offset(self, line_id: str) -> int:
        """Get the offset for a line.

        Args:
            line_id: ID of the line.

        Returns:
            The offset in minutes.

        Raises:
            KeyError: If line_id is not in schedule.
        """
        return self.offsets[line_id]

    def set_offset(self, line_id: str, offset: int) -> None:
        """Set the offset for a line (with modular arithmetic).

        Args:
            line_id: ID of the line.
            offset: New offset value (will be taken mod period).
        """
        self.offsets[line_id] = offset % self.period

    def shift_line(self, line_id: str, delta: int) -> None:
        """Shift a line's offset by delta minutes.

        Args:
            line_id: ID of the line to shift.
            delta: Amount to shift (can be negative).

        Raises:
            KeyError: If line_id is not in schedule.
        """
        current = self.offsets[line_id]
        self.offsets[line_id] = (current + delta) % self.period

    def shift_lines(self, line_ids: set[str], delta: int) -> None:
        """Shift multiple lines by the same delta.

        Args:
            line_ids: Set of line IDs to shift.
            delta: Amount to shift (can be negative).

        Raises:
            KeyError: If any line_id is not in schedule.
        """
        for line_id in line_ids:
            self.shift_line(line_id, delta)

    @classmethod
    def from_network(cls, network: Network, default_offset: int = 0) -> Schedule:
        """Create a schedule from a network with all lines at the same offset.

        Args:
            network: The network to create schedule for.
            default_offset: Initial offset for all lines.

        Returns:
            A new Schedule with all lines at default_offset.
        """
        offsets = {line_id: default_offset % network.period for line_id in network.lines}
        return cls(offsets=offsets, period=network.period)

    @classmethod
    def random(cls, network: Network, rng: object | None = None) -> Schedule:
        """Create a random schedule for a network.

        Args:
            network: The network to create schedule for.
            rng: NumPy random generator (optional). If None, uses default.

        Returns:
            A new Schedule with random offsets.
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        offsets = {
            line_id: int(rng.integers(0, network.period))
            for line_id in network.lines
        }
        return cls(offsets=offsets, period=network.period)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schedule):
            return NotImplemented
        return self.offsets == other.offsets and self.period == other.period

    def __len__(self) -> int:
        return len(self.offsets)
