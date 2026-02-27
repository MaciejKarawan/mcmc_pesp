"""Transfer definitions and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule


@dataclass(frozen=True)
class TransferParams:
    """Parameters for transfer cost calculation.

    Defines the soft cost function shape for a transfer connection.
    The cost function is:
    - Barrier (quadratic penalty) for w < w_min
    - Quadratic approach to ideal for w_min <= w < w_ideal
    - Quadratic departure from ideal for w >= w_ideal

    Args:
        w_min: Minimum acceptable waiting time in minutes.
        w_ideal: Ideal waiting time in minutes.
        w_max: Maximum reasonable waiting time in minutes.
        alpha: Penalty coefficient for being below ideal (early side).
        beta: Penalty coefficient for being above ideal (late side).
        M: Barrier coefficient for constraint violations.
    """

    w_min: float
    w_ideal: float
    w_max: float = 30.0
    alpha: float = 2.0
    beta: float = 0.5
    M: float = 1e6

    def __post_init__(self) -> None:
        if self.w_min < 0:
            raise ValueError(f"w_min must be non-negative, got {self.w_min}")
        if self.w_ideal < self.w_min:
            raise ValueError(
                f"w_ideal ({self.w_ideal}) must be >= w_min ({self.w_min})"
            )
        if self.w_max < self.w_ideal:
            raise ValueError(
                f"w_max ({self.w_max}) must be >= w_ideal ({self.w_ideal})"
            )
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta must be non-negative")
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")


# Default parameters for common transfer types
DEFAULT_TRANSFER_PARAMS = TransferParams(
    w_min=3.0,
    w_ideal=8.0,
    w_max=30.0,
    alpha=2.0,
    beta=0.5,
    M=1e6,
)

IC_TO_IC_PARAMS = TransferParams(
    w_min=5.0,
    w_ideal=12.0,
    w_max=30.0,
    alpha=3.0,
    beta=0.5,
    M=1e6,
)

IC_TO_REGIONAL_PARAMS = TransferParams(
    w_min=8.0,
    w_ideal=15.0,
    w_max=45.0,
    alpha=2.5,
    beta=0.4,
    M=1e6,
)

REGIONAL_TO_BUS_PARAMS = TransferParams(
    w_min=5.0,
    w_ideal=10.0,
    w_max=30.0,
    alpha=2.0,
    beta=0.5,
    M=1e6,
)

TRAM_TO_TRAM_PARAMS = TransferParams(
    w_min=2.0,
    w_ideal=5.0,
    w_max=15.0,
    alpha=2.0,
    beta=0.8,
    M=1e6,
)


@dataclass
class Transfer:
    """A transfer connection between two lines at a location.

    Represents a passenger transfer opportunity from one line to another.

    Args:
        id: Unique identifier for the transfer.
        from_line: ID of the line the passenger arrives on.
        from_station_idx: Index of the arrival station in from_line.
        to_line: ID of the line the passenger departs on.
        to_station_idx: Index of the departure station in to_line.
        walk_time: Walking time between platforms in minutes.
        params: Cost function parameters for this transfer.
        weight: Importance weight (e.g., passenger count estimate).
    """

    id: str
    from_line: str
    from_station_idx: int
    to_line: str
    to_station_idx: int
    walk_time: float
    params: TransferParams
    weight: float = 1.0

    def __hash__(self) -> int:
        return hash(self.id)

    def compute_waiting_time(
        self,
        schedule: Schedule,
        network: Network,
    ) -> float:
        """Compute the waiting time for this transfer given a schedule.

        Waiting time = departure_time(to_line) - arrival_time(from_line) - walk_time

        All times are computed modulo the period.

        Args:
            schedule: Current schedule with line offsets.
            network: Network containing line definitions.

        Returns:
            Waiting time in minutes, always in [0, to_line.frequency). Always >= 0.
        """
        from_line = network.get_line(self.from_line)
        to_line = network.get_line(self.to_line)

        # Arrival time at transfer point from origin line
        from_offset = schedule.get_offset(self.from_line)
        arrival_time = from_offset + from_line.arrival_offset(self.from_station_idx)

        # Departure time from transfer point on destination line
        to_offset = schedule.get_offset(self.to_line)
        departure_time = to_offset + to_line.departure_offset(self.to_station_idx)

        # w_gamma(x) = (-Delta_gamma(x)) mod f_{L_2}
        # Result is always in [0, f_{L_2}) — the time until next departure
        frequency = to_line.frequency
        raw_wait = departure_time - arrival_time - self.walk_time

        waiting_time = raw_wait % frequency

        return waiting_time


@dataclass
class TransferSet:
    """A collection of transfers in the network.

    Provides efficient lookup of transfers by line.
    """

    transfers: dict[str, Transfer]
    _by_from_line: dict[str, list[Transfer]] | None = None
    _by_to_line: dict[str, list[Transfer]] | None = None

    def __post_init__(self) -> None:
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices for transfers by line."""
        self._by_from_line = {}
        self._by_to_line = {}

        for transfer in self.transfers.values():
            if transfer.from_line not in self._by_from_line:
                self._by_from_line[transfer.from_line] = []
            self._by_from_line[transfer.from_line].append(transfer)

            if transfer.to_line not in self._by_to_line:
                self._by_to_line[transfer.to_line] = []
            self._by_to_line[transfer.to_line].append(transfer)

    def add_transfer(self, transfer: Transfer) -> None:
        """Add a transfer to the set.

        Args:
            transfer: Transfer to add.

        Raises:
            ValueError: If transfer with same ID exists.
        """
        if transfer.id in self.transfers:
            raise ValueError(f"Transfer with id '{transfer.id}' already exists")
        self.transfers[transfer.id] = transfer
        self._build_indices()

    def get_transfers_from_line(self, line_id: str) -> list[Transfer]:
        """Get all transfers originating from a line.

        Args:
            line_id: ID of the origin line.

        Returns:
            List of transfers from this line.
        """
        if self._by_from_line is None:
            self._build_indices()
        return self._by_from_line.get(line_id, [])

    def get_transfers_to_line(self, line_id: str) -> list[Transfer]:
        """Get all transfers arriving at a line.

        Args:
            line_id: ID of the destination line.

        Returns:
            List of transfers to this line.
        """
        if self._by_to_line is None:
            self._build_indices()
        return self._by_to_line.get(line_id, [])

    def get_transfers_involving_line(self, line_id: str) -> list[Transfer]:
        """Get all transfers involving a line (either direction).

        Args:
            line_id: ID of the line.

        Returns:
            List of transfers involving this line.
        """
        from_transfers = set(self.get_transfers_from_line(line_id))
        to_transfers = set(self.get_transfers_to_line(line_id))
        return list(from_transfers | to_transfers)

    def get_transfers_involving_lines(self, line_ids: set[str]) -> list[Transfer]:
        """Get all transfers involving any of the given lines.

        Args:
            line_ids: Set of line IDs.

        Returns:
            List of transfers involving any of these lines.
        """
        result = set()
        for line_id in line_ids:
            result.update(self.get_transfers_involving_line(line_id))
        return list(result)

    def __iter__(self):
        return iter(self.transfers.values())

    def __len__(self) -> int:
        return len(self.transfers)

    @classmethod
    def empty(cls) -> TransferSet:
        """Create an empty TransferSet."""
        return cls(transfers={})
