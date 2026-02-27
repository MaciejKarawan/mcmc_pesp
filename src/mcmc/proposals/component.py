"""Component move proposal for shifting connected line groups."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from src.core.network import Network
    from src.core.schedule import Schedule
    from src.core.transfers import TransferSet


@dataclass
class DependencyGraph:
    """Graph of line dependencies based on transfers.

    Represents which lines are connected through transfer opportunities.
    Used for Component Move to identify correlated lines.

    Args:
        adjacency: Mapping from line_id to set of connected line_ids.
    """

    adjacency: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_transfer_set(cls, transfers: TransferSet) -> DependencyGraph:
        """Build dependency graph from transfer set.

        Creates an undirected graph where lines are connected if there
        exists a transfer between them.

        Args:
            transfers: Set of transfers defining connections.

        Returns:
            DependencyGraph with adjacency structure.
        """
        adj: dict[str, set[str]] = {}

        for transfer in transfers:
            from_line = transfer.from_line
            to_line = transfer.to_line

            # Add edges in both directions (undirected graph)
            if from_line not in adj:
                adj[from_line] = set()
            if to_line not in adj:
                adj[to_line] = set()

            if from_line != to_line:
                adj[from_line].add(to_line)
                adj[to_line].add(from_line)

        return cls(adjacency=adj)

    def get_neighbors(self, line_id: str) -> set[str]:
        """Get all lines connected to the given line.

        Args:
            line_id: ID of the line to query.

        Returns:
            Set of connected line IDs.
        """
        return self.adjacency.get(line_id, set())

    def get_all_lines(self) -> set[str]:
        """Get all lines in the graph.

        Returns:
            Set of all line IDs.
        """
        return set(self.adjacency.keys())

    def average_degree(self) -> float:
        """Compute average node degree.

        Returns:
            Average number of neighbors per line.
        """
        if not self.adjacency:
            return 0.0
        total_degree = sum(len(neighbors) for neighbors in self.adjacency.values())
        return total_degree / len(self.adjacency)

    def get_connected_component(self, start_line: str) -> set[str]:
        """Get the full connected component containing start_line.

        Uses BFS to find all reachable lines.

        Args:
            start_line: Starting line for BFS.

        Returns:
            Set of all lines in the component.
        """
        if start_line not in self.adjacency:
            return {start_line}

        component: set[str] = set()
        queue: deque[str] = deque([start_line])

        while queue:
            current = queue.popleft()
            if current in component:
                continue
            component.add(current)
            for neighbor in self.get_neighbors(current):
                if neighbor not in component:
                    queue.append(neighbor)

        return component


@dataclass
class ComponentMove:
    """Proposal that shifts a connected component of lines.

    Uses BFS with stochastic decay to sample a subset of connected
    lines, then shifts them all by the same amount.

    The decay probability controls how far the component spreads:
    - High decay (close to 1): Small components (mostly single line)
    - Low decay (close to 0): Large components (entire connected part)

    The proposal is symmetric because p_decay is constant and doesn't
    depend on the current state, so Hastings ratio = 1.0.
    When line_weights are provided (for inertia), seed selection is biased
    but weights are state-independent, preserving symmetry.

    Args:
        dependency_graph: Graph defining line connections.
        sigma: Standard deviation for Gaussian shift.
        p_decay: Decay probability per edge (None = auto-compute).
        rng: NumPy random generator.
        line_weights: Optional dict mapping line_id to selection weight.
    """

    dependency_graph: DependencyGraph
    sigma: float = 5.0
    p_decay: float | None = None
    rng: Generator = field(default_factory=lambda: np.random.default_rng())
    line_weights: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return "component_move"

    def _compute_p_decay(self) -> float:
        """Compute decay probability based on average degree.

        Formula: p_decay = min(0.5, 0.8 / d_avg)

        This ensures expected component size stays bounded.
        For d_avg * p_decay < 1, the expected component size is finite.

        Returns:
            Decay probability.
        """
        d_avg = self.dependency_graph.average_degree()
        if d_avg == 0:
            return 0.5
        return min(0.5, 0.8 / d_avg)

    def _sample_component(self, seed_line: str) -> set[str]:
        """Sample a random component using BFS with decay.

        At each step, edges are traversed with probability (1 - p_decay).
        This creates randomly sized components centered on the seed.

        Args:
            seed_line: Starting line for BFS.

        Returns:
            Set of lines in the sampled component.
        """
        p_decay = self.p_decay if self.p_decay is not None else self._compute_p_decay()

        component: set[str] = {seed_line}
        queue: deque[str] = deque([seed_line])

        while queue:
            current = queue.popleft()

            for neighbor in self.dependency_graph.get_neighbors(current):
                if neighbor not in component:
                    # Include neighbor with probability (1 - p_decay)
                    if self.rng.random() > p_decay:
                        component.add(neighbor)
                        queue.append(neighbor)

        return component

    def propose(
        self,
        schedule: Schedule,
        network: Network,
    ) -> tuple[Schedule, float, set[str]]:
        """Propose shifting a component of lines.

        Algorithm:
        1. Select seed line uniformly at random
        2. Sample component via BFS with decay
        3. Sample shift δ ~ N(0, σ²)
        4. Apply shift to all lines in component

        Args:
            schedule: Current schedule state.
            network: Network definition.

        Returns:
            Tuple of (new_schedule, hastings_ratio=1.0, changed_lines).
        """
        # Select seed line (with optional inertia weights)
        line_ids = network.line_ids
        if not line_ids:
            return schedule.copy(), 1.0, set()

        if self.line_weights is not None:
            weights = np.array([self.line_weights.get(lid, 1.0) for lid in line_ids])
            weights = weights / weights.sum()
            seed_line = str(self.rng.choice(line_ids, p=weights))
        else:
            seed_line = str(self.rng.choice(line_ids))

        # Sample component
        component = self._sample_component(seed_line)

        # Filter to lines that actually exist in network
        component = component & set(line_ids)

        if not component:
            return schedule.copy(), 1.0, set()

        # Sample shift
        delta = self.rng.normal(0, self.sigma)
        delta_int = int(round(delta))

        # Apply shift to all lines in component
        new_schedule = schedule.copy()
        for line_id in component:
            new_schedule.shift_line(line_id, delta_int)

        # Symmetric proposal -> Hastings ratio = 1.0
        return new_schedule, 1.0, component


def create_component_move(
    transfers: TransferSet,
    sigma: float = 5.0,
    p_decay: float | None = None,
    rng: Generator | None = None,
) -> ComponentMove:
    """Factory function to create ComponentMove from transfers.

    Args:
        transfers: Transfer set defining line dependencies.
        sigma: Standard deviation for Gaussian shift.
        p_decay: Decay probability (None = auto).
        rng: Random generator (None = new default).

    Returns:
        Configured ComponentMove instance.
    """
    graph = DependencyGraph.from_transfer_set(transfers)

    if rng is None:
        rng = np.random.default_rng()

    return ComponentMove(
        dependency_graph=graph,
        sigma=sigma,
        p_decay=p_decay,
        rng=rng,
    )
