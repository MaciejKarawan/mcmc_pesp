"""Tests for component move proposal."""

import numpy as np
import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import Transfer, TransferParams, TransferSet
from src.mcmc.proposals.component import (
    ComponentMove,
    DependencyGraph,
    create_component_move,
)


class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    @pytest.fixture
    def simple_transfers(self) -> TransferSet:
        """Create transfers forming L1 <-> L2 <-> L3 chain."""
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)

        t1 = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        t2 = Transfer(
            id="T2",
            from_line="L2",
            from_station_idx=1,
            to_line="L3",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )

        return TransferSet(transfers={"T1": t1, "T2": t2})

    def test_from_transfer_set(self, simple_transfers: TransferSet) -> None:
        graph = DependencyGraph.from_transfer_set(simple_transfers)

        assert "L1" in graph.adjacency
        assert "L2" in graph.adjacency
        assert "L3" in graph.adjacency

        # L1 connected to L2
        assert "L2" in graph.get_neighbors("L1")
        assert "L1" in graph.get_neighbors("L2")

        # L2 connected to L3
        assert "L3" in graph.get_neighbors("L2")
        assert "L2" in graph.get_neighbors("L3")

        # L1 not directly connected to L3
        assert "L3" not in graph.get_neighbors("L1")

    def test_empty_transfer_set(self) -> None:
        ts = TransferSet(transfers={})
        graph = DependencyGraph.from_transfer_set(ts)

        assert len(graph.adjacency) == 0
        assert graph.average_degree() == 0

    def test_average_degree(self, simple_transfers: TransferSet) -> None:
        graph = DependencyGraph.from_transfer_set(simple_transfers)

        # L1: degree 1 (connected to L2)
        # L2: degree 2 (connected to L1 and L3)
        # L3: degree 1 (connected to L2)
        # Average = (1 + 2 + 1) / 3 = 1.33...
        expected = 4 / 3
        assert abs(graph.average_degree() - expected) < 1e-10

    def test_get_connected_component(self, simple_transfers: TransferSet) -> None:
        graph = DependencyGraph.from_transfer_set(simple_transfers)

        # Starting from L1, should reach L2 and L3
        component = graph.get_connected_component("L1")
        assert component == {"L1", "L2", "L3"}

        # Same for starting from L3
        component = graph.get_connected_component("L3")
        assert component == {"L1", "L2", "L3"}

    def test_get_neighbors_unknown_line(self) -> None:
        graph = DependencyGraph(adjacency={"L1": {"L2"}})
        assert graph.get_neighbors("L99") == set()

    def test_self_loop_ignored(self) -> None:
        """Self-loop transfers shouldn't create edges."""
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L1",  # Same line
            to_station_idx=1,
            walk_time=0.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})
        graph = DependencyGraph.from_transfer_set(ts)

        # L1 should have no neighbors (self-loop doesn't count)
        assert len(graph.get_neighbors("L1")) == 0


class TestComponentMove:
    """Tests for ComponentMove class."""

    @pytest.fixture
    def simple_network(self) -> Network:
        """Create a simple network with three lines."""
        network = Network()

        for line_id in ["L1", "L2", "L3"]:
            stations = [
                Station(id=f"A_{line_id}", location="A", modality="bus"),
                Station(id=f"B_{line_id}", location="B", modality="bus"),
            ]
            line = Line(
                id=line_id,
                modality="bus",
                stations=stations,
                travel_times=[15],
                dwell_times=[2, 2],
                frequency=60,
            )
            network.add_line(line)

        return network

    @pytest.fixture
    def chain_transfers(self) -> TransferSet:
        """L1 <-> L2 <-> L3 chain."""
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)

        t1 = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        t2 = Transfer(
            id="T2",
            from_line="L2",
            from_station_idx=1,
            to_line="L3",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )

        return TransferSet(transfers={"T1": t1, "T2": t2})

    @pytest.fixture
    def component_move(self, chain_transfers: TransferSet) -> ComponentMove:
        graph = DependencyGraph.from_transfer_set(chain_transfers)
        rng = np.random.default_rng(42)
        return ComponentMove(
            dependency_graph=graph,
            sigma=5.0,
            p_decay=0.3,
            rng=rng,
        )

    def test_sample_component_includes_seed(
        self,
        component_move: ComponentMove,
    ) -> None:
        """Seed line is always included in component."""
        for _ in range(10):
            component = component_move._sample_component("L1")
            assert "L1" in component

    def test_sample_component_subset_of_connected(
        self,
        component_move: ComponentMove,
    ) -> None:
        """Sampled component is subset of full connected component."""
        full = component_move.dependency_graph.get_connected_component("L1")

        for _ in range(10):
            sampled = component_move._sample_component("L1")
            assert sampled.issubset(full)

    def test_propose_returns_valid_structure(
        self,
        component_move: ComponentMove,
        simple_network: Network,
    ) -> None:
        schedule = Schedule.from_network(simple_network)

        new_schedule, hastings, changed = component_move.propose(schedule, simple_network)

        assert isinstance(new_schedule, Schedule)
        assert isinstance(hastings, float)
        assert isinstance(changed, set)
        assert len(changed) > 0

    def test_hastings_ratio_is_one(
        self,
        component_move: ComponentMove,
        simple_network: Network,
    ) -> None:
        """Symmetric proposal has Hastings ratio = 1.0."""
        schedule = Schedule.from_network(simple_network)

        for _ in range(10):
            _, hastings, _ = component_move.propose(schedule, simple_network)
            assert hastings == 1.0

    def test_shifts_all_lines_in_component(
        self,
        chain_transfers: TransferSet,
        simple_network: Network,
    ) -> None:
        """All lines in component are shifted by same amount."""
        graph = DependencyGraph.from_transfer_set(chain_transfers)
        rng = np.random.default_rng(123)

        # Use p_decay=0 to always include all neighbors
        move = ComponentMove(
            dependency_graph=graph,
            sigma=5.0,
            p_decay=0.0,  # Always include all neighbors
            rng=rng,
        )

        schedule = Schedule(offsets={"L1": 10, "L2": 20, "L3": 30}, period=1440)
        new_schedule, _, changed = move.propose(schedule, simple_network)

        # All lines should be shifted by same amount
        if len(changed) > 1:
            deltas = []
            for line_id in changed:
                old = schedule.get_offset(line_id)
                new = new_schedule.get_offset(line_id)
                delta = (new - old) % 1440
                if delta > 720:
                    delta -= 1440
                deltas.append(delta)

            # All deltas should be the same
            assert len(set(deltas)) == 1

    def test_compute_p_decay(self) -> None:
        """Test automatic p_decay computation."""
        # With d_avg = 2, p_decay = min(0.5, 0.8/2) = 0.4
        adj = {"L1": {"L2", "L3"}, "L2": {"L1"}, "L3": {"L1"}}
        graph = DependencyGraph(adjacency=adj)

        move = ComponentMove(
            dependency_graph=graph,
            sigma=5.0,
            p_decay=None,  # Auto-compute
        )

        expected = min(0.5, 0.8 / graph.average_degree())
        assert abs(move._compute_p_decay() - expected) < 1e-10

    def test_with_empty_graph(
        self,
        simple_network: Network,
    ) -> None:
        """Test behavior with empty dependency graph."""
        graph = DependencyGraph(adjacency={})
        rng = np.random.default_rng(42)
        move = ComponentMove(dependency_graph=graph, sigma=5.0, rng=rng)

        schedule = Schedule.from_network(simple_network)
        new_schedule, hastings, changed = move.propose(schedule, simple_network)

        # Should still work, moving single lines
        assert hastings == 1.0

    def test_name_property(self, component_move: ComponentMove) -> None:
        assert component_move.name == "component_move"


class TestCreateComponentMove:
    """Tests for factory function."""

    def test_create_from_transfers(self) -> None:
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})

        move = create_component_move(ts, sigma=10.0, p_decay=0.5)

        assert move.sigma == 10.0
        assert move.p_decay == 0.5
        assert "L1" in move.dependency_graph.adjacency
        assert "L2" in move.dependency_graph.adjacency
