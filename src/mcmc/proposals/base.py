"""Base interface for MCMC proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.core.network import Network
    from src.core.schedule import Schedule


class Proposal(Protocol):
    """Protocol for MCMC move proposals.

    A proposal generates candidate states for the Metropolis-Hastings algorithm.
    Each proposal must provide:
    - A method to generate a new state from the current state
    - The Hastings ratio q(x|x') / q(x'|x) for the move
    - A name for logging and diagnostics
    """

    @property
    def name(self) -> str:
        """Name of this proposal for logging."""
        ...

    def propose(
        self,
        schedule: Schedule,
        network: Network,
    ) -> tuple[Schedule, float, set[str]]:
        """Generate a proposal move.

        Args:
            schedule: Current schedule state.
            network: Network definition.

        Returns:
            Tuple of:
            - new_schedule: The proposed new state
            - hastings_ratio: q(x|x') / q(x'|x), equals 1.0 for symmetric proposals
            - changed_lines: Set of line IDs that were modified
        """
        ...


@dataclass
class ProposalResult:
    """Result of a proposal move."""

    new_schedule: Schedule
    hastings_ratio: float
    changed_lines: set[str]
    proposal_name: str
