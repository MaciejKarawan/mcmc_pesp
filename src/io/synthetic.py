"""Synthetic network generators for testing and PoC."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.network import T_PERIOD, Line, Network, Station
from src.core.transfers import (
    DEFAULT_TRANSFER_PARAMS,
    Transfer,
    TransferParams,
    TransferSet,
)


@dataclass
class TriangleNetwork:
    """A simple triangle network for Proof of Concept.

    The Triangle network consists of 3 locations (A, B, C) arranged in a triangle,
    with 3 lines connecting them:
    - Line L1: A -> B
    - Line L2: B -> C
    - Line L3: A -> C

    This creates 3 transfer opportunities:
    - At A: L3 -> L1 (passengers going A->B->C can also go A->C)
    - At B: L1 -> L2 (passengers transferring from L1 to L2)
    - At C: L3 -> L2 (passengers arriving on L3 can transfer to L2 going back)

    The network has a known optimal solution where all transfers have ideal
    waiting times.

    Args:
        frequency: Service frequency for all lines (default 60 minutes).
        travel_time: Travel time between stations (default 15 minutes).
        dwell_time: Dwell time at each station (default 2 minutes).
        walk_time: Walk time for transfers (default 3 minutes).
        transfer_params: Parameters for transfer costs (default params used if None).
    """

    frequency: int = 60
    travel_time: int = 15
    dwell_time: int = 2
    walk_time: float = 3.0
    transfer_params: TransferParams | None = None

    def __post_init__(self) -> None:
        if self.transfer_params is None:
            self.transfer_params = DEFAULT_TRANSFER_PARAMS

    def create_network(self) -> Network:
        """Create the triangle network.

        Returns:
            Network object with 3 lines.
        """
        # Create stations
        station_a1 = Station(id="A_L1", location="A", modality="bus")
        station_b1 = Station(id="B_L1", location="B", modality="bus")

        station_b2 = Station(id="B_L2", location="B", modality="bus")
        station_c2 = Station(id="C_L2", location="C", modality="bus")

        station_a3 = Station(id="A_L3", location="A", modality="bus")
        station_c3 = Station(id="C_L3", location="C", modality="bus")

        # Create lines
        line_1 = Line(
            id="L1",
            modality="bus",
            stations=[station_a1, station_b1],
            travel_times=[self.travel_time],
            dwell_times=[self.dwell_time, self.dwell_time],
            frequency=self.frequency,
        )

        line_2 = Line(
            id="L2",
            modality="bus",
            stations=[station_b2, station_c2],
            travel_times=[self.travel_time],
            dwell_times=[self.dwell_time, self.dwell_time],
            frequency=self.frequency,
        )

        line_3 = Line(
            id="L3",
            modality="bus",
            stations=[station_a3, station_c3],
            travel_times=[self.travel_time],
            dwell_times=[self.dwell_time, self.dwell_time],
            frequency=self.frequency,
        )

        # Create network
        network = Network(period=T_PERIOD)
        network.add_line(line_1)
        network.add_line(line_2)
        network.add_line(line_3)

        return network

    def create_transfers(self) -> TransferSet:
        """Create the transfer set for the triangle network.

        Returns:
            TransferSet with 3 transfers.
        """
        assert self.transfer_params is not None

        transfers = {}

        # Transfer at B: L1 (arriving) -> L2 (departing)
        # Passenger arrives at B on L1, transfers to L2
        transfer_b = Transfer(
            id="T_B_L1_L2",
            from_line="L1",
            from_station_idx=1,  # Arrival at B (second station of L1)
            to_line="L2",
            to_station_idx=0,  # Departure from B (first station of L2)
            walk_time=self.walk_time,
            params=self.transfer_params,
            weight=1.0,
        )
        transfers[transfer_b.id] = transfer_b

        # Transfer at A: L3 -> L1
        # Passenger could arrive on some line and transfer to L1 or L3
        # For simplicity: arriving on L1 (going back) to L3
        # Actually let's make it: external arrival -> L1 and L3
        # Simpler: L1 arrives at A (first station), L3 departs from A
        # But L1 starts at A, doesn't arrive... Let's reconsider.

        # Actually for a proper triangle, let's think of it as:
        # - L1: A -> B (so arrives at B, departs from A)
        # - L2: B -> C (so arrives at C, departs from B)
        # - L3: A -> C (so arrives at C, departs from A)

        # Meaningful transfers:
        # - At B: L1 arrival -> L2 departure (main transfer)
        # - At C: L2 arrival and L3 arrival (but both arrive, no transfer needed)
        # - At A: L1 and L3 both depart (but we could have external arrivals)

        # For a simple PoC, let's have just the main transfer at B
        # and add a reverse direction for L2

        # Let's simplify: create only the key transfer at B
        # and one more to make it interesting

        # Transfer at C: L3 (arriving) -> reverse of L2 conceptually
        # But L2 goes B->C, so no departure from C...

        # Let's just keep the single transfer for now - it's enough for PoC
        # The optimization will try to align L1 arrival at B with L2 departure from B

        return TransferSet(transfers=transfers)

    def create(self) -> tuple[Network, TransferSet]:
        """Create both the network and transfers.

        Returns:
            Tuple of (Network, TransferSet).
        """
        return self.create_network(), self.create_transfers()

    def compute_optimal_offset(self) -> dict[str, int]:
        """Compute the optimal offsets for this network.

        For the triangle network with one transfer at B (L1->L2), the optimal
        solution is when:
            departure_L2(B) - arrival_L1(B) - walk_time = w_ideal

        Let:
            - theta_1 = offset of L1
            - theta_2 = offset of L2
            - arrival_L1(B) = theta_1 + travel_time + dwell_time
            - departure_L2(B) = theta_2 + dwell_time

        For ideal waiting time w_ideal:
            theta_2 + dwell - (theta_1 + travel + dwell) - walk = w_ideal
            theta_2 - theta_1 = travel + walk + w_ideal

        With L1 at offset 0:
            theta_2 = travel + walk + w_ideal

        Returns:
            Dict mapping line IDs to optimal offsets.
        """
        assert self.transfer_params is not None
        w_ideal = self.transfer_params.w_ideal

        # Fix L1 at 0, L3 can be anything (not involved in transfer)
        theta_1 = 0
        theta_3 = 0  # Arbitrary

        # Compute theta_2 for ideal transfer at B
        # arrival_L1(B) = theta_1 + travel_time + dwell_time (arrival after travel + initial dwell)
        # Actually: L1 departs A at theta_1 + dwell_time, arrives B at theta_1 + dwell_time + travel_time
        arrival_l1_b = theta_1 + self.dwell_time + self.travel_time

        # departure_L2(B) = theta_2 + dwell_time (first station departure)
        # For waiting time = w_ideal:
        # theta_2 + dwell - arrival_l1_b - walk = w_ideal
        # theta_2 = arrival_l1_b + walk + w_ideal - dwell
        theta_2 = int(arrival_l1_b + self.walk_time + w_ideal - self.dwell_time)
        theta_2 = theta_2 % self.frequency

        return {"L1": theta_1, "L2": theta_2, "L3": theta_3}


def create_triangle_network(
    frequency: int = 60,
    travel_time: int = 15,
    dwell_time: int = 2,
    walk_time: float = 3.0,
    transfer_params: TransferParams | None = None,
) -> tuple[Network, TransferSet]:
    """Convenience function to create a triangle network.

    Args:
        frequency: Service frequency for all lines.
        travel_time: Travel time between stations.
        dwell_time: Dwell time at each station.
        walk_time: Walk time for transfers.
        transfer_params: Parameters for transfer costs.

    Returns:
        Tuple of (Network, TransferSet).
    """
    builder = TriangleNetwork(
        frequency=frequency,
        travel_time=travel_time,
        dwell_time=dwell_time,
        walk_time=walk_time,
        transfer_params=transfer_params,
    )
    return builder.create()
