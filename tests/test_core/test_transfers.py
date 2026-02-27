"""Tests for transfers module."""

import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import (
    DEFAULT_TRANSFER_PARAMS,
    Transfer,
    TransferParams,
    TransferSet,
)


class TestTransferParams:
    """Tests for TransferParams dataclass."""

    def test_creation_default(self) -> None:
        params = TransferParams(w_min=3.0, w_ideal=8.0)
        assert params.w_min == 3.0
        assert params.w_ideal == 8.0
        assert params.w_max == 30.0
        assert params.alpha == 2.0
        assert params.beta == 0.5
        assert params.M == 1e6

    def test_creation_custom(self) -> None:
        params = TransferParams(
            w_min=5.0,
            w_ideal=12.0,
            w_max=45.0,
            alpha=3.0,
            beta=0.4,
            M=1e5,
        )
        assert params.w_min == 5.0
        assert params.w_ideal == 12.0
        assert params.w_max == 45.0
        assert params.alpha == 3.0
        assert params.beta == 0.4
        assert params.M == 1e5

    def test_invalid_w_min_negative(self) -> None:
        with pytest.raises(ValueError, match="w_min must be non-negative"):
            TransferParams(w_min=-1.0, w_ideal=8.0)

    def test_invalid_w_ideal_less_than_w_min(self) -> None:
        with pytest.raises(ValueError, match="w_ideal .* must be >= w_min"):
            TransferParams(w_min=10.0, w_ideal=5.0)

    def test_invalid_w_max_less_than_w_ideal(self) -> None:
        with pytest.raises(ValueError, match="w_max .* must be >= w_ideal"):
            TransferParams(w_min=3.0, w_ideal=10.0, w_max=5.0)

    def test_invalid_negative_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha and beta must be non-negative"):
            TransferParams(w_min=3.0, w_ideal=8.0, alpha=-1.0)

    def test_invalid_zero_M(self) -> None:
        with pytest.raises(ValueError, match="M must be positive"):
            TransferParams(w_min=3.0, w_ideal=8.0, M=0)


class TestTransfer:
    """Tests for Transfer dataclass."""

    @pytest.fixture
    def simple_network(self) -> Network:
        """Create a simple network with two lines meeting at B."""
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

    @pytest.fixture
    def transfer_at_b(self) -> Transfer:
        """Create a transfer from L1 to L2 at location B."""
        return Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,  # Arrival at B
            to_line="L2",
            to_station_idx=0,  # Departure from B
            walk_time=3.0,
            params=DEFAULT_TRANSFER_PARAMS,
            weight=1.0,
        )

    def test_transfer_creation(self, transfer_at_b: Transfer) -> None:
        assert transfer_at_b.id == "T1"
        assert transfer_at_b.from_line == "L1"
        assert transfer_at_b.to_line == "L2"
        assert transfer_at_b.walk_time == 3.0

    def test_compute_waiting_time_aligned(
        self,
        simple_network: Network,
        transfer_at_b: Transfer,
    ) -> None:
        # L1 arrives at B at: offset_L1 + dwell[0] + travel[0] = 0 + 2 + 15 = 17
        # L2 departs from B at: offset_L2 + dwell[0] = 30 + 2 = 32
        # Waiting time = 32 - 17 - 3 = 12
        schedule = Schedule(offsets={"L1": 0, "L2": 30}, period=1440)
        waiting_time = transfer_at_b.compute_waiting_time(schedule, simple_network)
        assert waiting_time == 12.0

    def test_compute_waiting_time_with_wrap(
        self,
        simple_network: Network,
        transfer_at_b: Transfer,
    ) -> None:
        # L1 arrives at B at: 50 + 2 + 15 = 67 -> 7 (mod 60)
        # L2 departs from B at: 0 + 2 = 2
        # Raw: 2 - 7 - 3 = -8, mod 60 = 52
        schedule = Schedule(offsets={"L1": 50, "L2": 0}, period=1440)
        waiting_time = transfer_at_b.compute_waiting_time(schedule, simple_network)
        assert waiting_time == 52.0

    def test_compute_waiting_time_zero_offset(
        self,
        simple_network: Network,
        transfer_at_b: Transfer,
    ) -> None:
        # L1 arrives at B at: 0 + 2 + 15 = 17
        # L2 departs from B at: 0 + 2 = 2
        # Raw: 2 - 17 - 3 = -18, mod 60 = 42
        schedule = Schedule(offsets={"L1": 0, "L2": 0}, period=1440)
        waiting_time = transfer_at_b.compute_waiting_time(schedule, simple_network)
        assert waiting_time == 42.0


class TestTransferSet:
    """Tests for TransferSet class."""

    @pytest.fixture
    def transfers(self) -> TransferSet:
        """Create a TransferSet with multiple transfers."""
        t1 = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=DEFAULT_TRANSFER_PARAMS,
        )
        t2 = Transfer(
            id="T2",
            from_line="L2",
            from_station_idx=1,
            to_line="L3",
            to_station_idx=0,
            walk_time=3.0,
            params=DEFAULT_TRANSFER_PARAMS,
        )
        t3 = Transfer(
            id="T3",
            from_line="L1",
            from_station_idx=0,
            to_line="L3",
            to_station_idx=1,
            walk_time=3.0,
            params=DEFAULT_TRANSFER_PARAMS,
        )
        return TransferSet(transfers={"T1": t1, "T2": t2, "T3": t3})

    def test_len(self, transfers: TransferSet) -> None:
        assert len(transfers) == 3

    def test_iteration(self, transfers: TransferSet) -> None:
        ids = {t.id for t in transfers}
        assert ids == {"T1", "T2", "T3"}

    def test_get_transfers_from_line(self, transfers: TransferSet) -> None:
        from_l1 = transfers.get_transfers_from_line("L1")
        assert len(from_l1) == 2
        assert all(t.from_line == "L1" for t in from_l1)

        from_l2 = transfers.get_transfers_from_line("L2")
        assert len(from_l2) == 1

        from_l3 = transfers.get_transfers_from_line("L3")
        assert len(from_l3) == 0

    def test_get_transfers_to_line(self, transfers: TransferSet) -> None:
        to_l2 = transfers.get_transfers_to_line("L2")
        assert len(to_l2) == 1

        to_l3 = transfers.get_transfers_to_line("L3")
        assert len(to_l3) == 2

    def test_get_transfers_involving_line(self, transfers: TransferSet) -> None:
        involving_l1 = transfers.get_transfers_involving_line("L1")
        assert len(involving_l1) == 2

        involving_l2 = transfers.get_transfers_involving_line("L2")
        assert len(involving_l2) == 2  # T1 (to L2), T2 (from L2)

        involving_l3 = transfers.get_transfers_involving_line("L3")
        assert len(involving_l3) == 2

    def test_get_transfers_involving_lines_multiple(
        self,
        transfers: TransferSet,
    ) -> None:
        involving = transfers.get_transfers_involving_lines({"L1", "L2"})
        # All 3 transfers involve either L1 or L2
        assert len(involving) == 3

    def test_add_transfer(self) -> None:
        ts = TransferSet.empty()
        assert len(ts) == 0

        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=DEFAULT_TRANSFER_PARAMS,
        )
        ts.add_transfer(t)
        assert len(ts) == 1

    def test_add_duplicate_raises(self, transfers: TransferSet) -> None:
        duplicate = Transfer(
            id="T1",  # Same ID
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=DEFAULT_TRANSFER_PARAMS,
        )
        with pytest.raises(ValueError, match="already exists"):
            transfers.add_transfer(duplicate)

    def test_empty(self) -> None:
        ts = TransferSet.empty()
        assert len(ts) == 0
        assert list(ts) == []
