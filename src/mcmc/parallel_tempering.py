"""Inertial Parallel Tempering for MCMC-PESP.

Implements Parallel Tempering (replica exchange) with:
- Geometric temperature schedule
- Inertia mechanism (modality-based line selection bias per replica)
- Metropolis-based replica exchange between adjacent temperatures
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from src.core.schedule import Schedule
from src.mcmc.sampler import MCMCResult, MCMCSampler

if TYPE_CHECKING:
    from numpy.random import Generator

    from src.core.network import Network
    from src.cost.transfer_cost import CostFunction
    from src.mcmc.proposals.base import Proposal

logger = logging.getLogger(__name__)


@dataclass
class InertiaConfig:
    """Configuration for the inertia mechanism.

    Maps transport modality to inertia value. Higher inertia means the line
    is modified less frequently in cold replicas, preserving good solutions
    for high-priority services.

    The inertia vector is interpolated per replica:
    - Cold replicas (high index, high beta): use full modality inertia
    - Hot replicas (low index, low beta): uniform (I_L = 1 for all L)

    Args:
        modality_inertia: Mapping from modality name to inertia value (>= 1).
            Example: {"IC": 20, "Regional": 5, "Local": 1}
    """

    modality_inertia: dict[str, float]

    def __post_init__(self) -> None:
        for modality, inertia in self.modality_inertia.items():
            if inertia < 1.0:
                raise ValueError(
                    f"Inertia for {modality} must be >= 1, got {inertia}"
                )

    def compute_line_weights(
        self,
        network: Network,
        replica_idx: int,
        n_replicas: int,
    ) -> dict[str, float]:
        """Compute line selection weights P(L) proportional to 1/I_L.

        Interpolates between hierarchical inertia (cold replicas) and
        uniform weights (hot replicas).

        Args:
            network: Network with line definitions.
            replica_idx: Index of the replica (0 = hottest, M-1 = coldest).
            n_replicas: Total number of replicas.

        Returns:
            Dict mapping line_id to normalized selection weight.
        """
        # Alpha: 1 for hottest (idx=0), 0 for coldest (idx=M-1)
        alpha = 1.0 - replica_idx / (n_replicas - 1) if n_replicas > 1 else 0.0

        weights: dict[str, float] = {}
        for line_id, line in network.lines.items():
            i_hierarchical = self.modality_inertia.get(line.modality, 1.0)
            # Interpolate: cold uses full inertia, hot uses I=1
            i_effective = 1.0 + (1.0 - alpha) * (i_hierarchical - 1.0)
            weights[line_id] = 1.0 / i_effective

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights


@dataclass
class PTResult:
    """Result of a Parallel Tempering run.

    Args:
        best_schedule: Best schedule found across all replicas.
        best_cost: Cost of the best schedule.
        cold_chain_samples: Samples from the coldest replica (target distribution).
        cold_chain_costs: Costs from the coldest replica.
        per_replica_acceptance_rates: MH acceptance rate for each replica.
        swap_acceptance_rates: Swap acceptance rate for each adjacent pair.
        temperature_schedule: List of beta values for each replica.
    """

    best_schedule: Schedule
    best_cost: float
    cold_chain_samples: list[Schedule]
    cold_chain_costs: list[float]
    per_replica_acceptance_rates: list[float]
    swap_acceptance_rates: list[float]
    temperature_schedule: list[float]

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "best_cost": self.best_cost,
            "n_replicas": len(self.temperature_schedule),
            "n_cold_samples": len(self.cold_chain_samples),
            "cold_mean_cost": float(np.mean(self.cold_chain_costs))
            if self.cold_chain_costs
            else float("inf"),
            "cold_std_cost": float(np.std(self.cold_chain_costs))
            if self.cold_chain_costs
            else 0.0,
            "per_replica_acceptance": self.per_replica_acceptance_rates,
            "swap_acceptance": self.swap_acceptance_rates,
            "beta_min": self.temperature_schedule[0],
            "beta_max": self.temperature_schedule[-1],
        }


@dataclass
class ParallelTempering:
    """Inertial Parallel Tempering for PESP optimization.

    Runs M replicas at different temperatures with periodic state exchanges.
    Cold replicas exploit good solutions while hot replicas explore broadly.
    The inertia mechanism biases cold replicas to preserve high-priority
    line schedules.

    Args:
        network: Transport network definition.
        cost_fn: Cost function to minimize.
        proposals: List of proposal move generators (will be copied per replica).
        proposal_weights: Weights for selecting proposals.
        n_replicas: Number of temperature replicas.
        beta_min: Inverse temperature for the hottest replica.
        beta_max: Inverse temperature for the coldest replica.
        swap_interval: Attempt replica swaps every N steps.
        mode: "sampling" (formal MCMC) or "optimization" (best-so-far).
        inertia_config: Inertia configuration (None = no inertia).
        rng: NumPy random generator.
    """

    network: Network
    cost_fn: CostFunction
    proposals: list[Proposal]
    proposal_weights: list[float] | None = None
    n_replicas: int = 8
    beta_min: float = 0.1
    beta_max: float = 10.0
    swap_interval: int = 10
    mode: Literal["sampling", "optimization"] = "optimization"
    inertia_config: InertiaConfig | None = None
    rng: Generator = field(default_factory=lambda: np.random.default_rng())

    # Internal state
    _samplers: list[MCMCSampler] = field(default_factory=list, init=False)
    _betas: list[float] = field(default_factory=list, init=False)
    _swap_attempts: list[int] = field(default_factory=list, init=False)
    _swap_accepts: list[int] = field(default_factory=list, init=False)
    _step_count: int = field(default=0, init=False)
    _best_schedule: Schedule | None = field(default=None, init=False)
    _best_cost: float = field(default=float("inf"), init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.n_replicas < 2:
            raise ValueError(f"n_replicas must be >= 2, got {self.n_replicas}")
        if self.beta_min <= 0 or self.beta_max <= 0:
            raise ValueError("beta_min and beta_max must be positive")
        if self.beta_min >= self.beta_max:
            raise ValueError("beta_min must be < beta_max")

        self._betas = self._compute_temperature_schedule()
        self._swap_attempts = [0] * (self.n_replicas - 1)
        self._swap_accepts = [0] * (self.n_replicas - 1)
        self._samplers = self._create_replica_samplers()

    def _compute_temperature_schedule(self) -> list[float]:
        """Compute geometric temperature schedule.

        beta_i = beta_min * r^i, where r = (beta_max/beta_min)^(1/(M-1))

        Returns:
            List of M beta values from beta_min to beta_max.
        """
        if self.n_replicas == 1:
            return [self.beta_max]

        r = (self.beta_max / self.beta_min) ** (1.0 / (self.n_replicas - 1))
        return [self.beta_min * r**i for i in range(self.n_replicas)]

    def _create_replica_proposals(
        self, replica_idx: int
    ) -> list[Proposal]:
        """Create proposal copies for a specific replica with inertia weights.

        Args:
            replica_idx: Index of the replica (0 = hottest, M-1 = coldest).

        Returns:
            List of proposals configured for this replica.
        """
        # Compute line weights for this replica
        line_weights: dict[str, float] | None = None
        if self.inertia_config is not None:
            line_weights = self.inertia_config.compute_line_weights(
                self.network, replica_idx, self.n_replicas
            )

        replica_rng = np.random.default_rng(
            self.rng.integers(0, 2**32)
        )

        replica_proposals: list[Proposal] = []
        for proposal in self.proposals:
            # Deep copy the proposal and set replica-specific parameters
            new_proposal = copy.deepcopy(proposal)
            new_proposal.rng = np.random.default_rng(
                replica_rng.integers(0, 2**32)
            )

            # Set line_weights if proposal supports it
            if hasattr(new_proposal, "line_weights"):
                new_proposal.line_weights = line_weights

            replica_proposals.append(new_proposal)

        return replica_proposals

    def _create_replica_samplers(self) -> list[MCMCSampler]:
        """Create one MCMCSampler per replica with appropriate beta and inertia.

        Returns:
            List of M samplers.
        """
        samplers = []
        for i, beta in enumerate(self._betas):
            replica_proposals = self._create_replica_proposals(i)

            sampler = MCMCSampler(
                network=self.network,
                cost_fn=self.cost_fn,
                proposals=replica_proposals,
                proposal_weights=list(self.proposal_weights)
                if self.proposal_weights
                else None,
                beta=beta,
                mode=self.mode,
                rng=np.random.default_rng(self.rng.integers(0, 2**32)),
            )
            samplers.append(sampler)

        return samplers

    def initialize(self, schedules: list[Schedule] | None = None) -> None:
        """Initialize all replicas.

        Args:
            schedules: Optional list of initial schedules (one per replica).
                If None, random schedules are generated.

        Raises:
            ValueError: If schedules list has wrong length.
        """
        if schedules is not None and len(schedules) != self.n_replicas:
            raise ValueError(
                f"Expected {self.n_replicas} schedules, got {len(schedules)}"
            )

        for i, sampler in enumerate(self._samplers):
            if schedules is not None:
                sampler.initialize(schedules[i])
            else:
                sampler.initialize(
                    Schedule.random(self.network, self.rng)
                )

        self._step_count = 0
        self._initialized = True

        # Initialize best tracking
        self._update_best()

        logger.info(
            f"Initialized PT with {self.n_replicas} replicas, "
            f"beta=[{self._betas[0]:.3f}, ..., {self._betas[-1]:.3f}]"
        )

    def _update_best(self) -> None:
        """Update global best from all replicas."""
        for sampler in self._samplers:
            if sampler.current_cost < self._best_cost:
                self._best_cost = sampler.current_cost
                self._best_schedule = sampler.current_schedule.copy()

            # Also check replica's own best (in optimization mode)
            if (
                sampler.best_schedule is not None
                and sampler.best_cost < self._best_cost
            ):
                self._best_cost = sampler.best_cost
                self._best_schedule = sampler.best_schedule.copy()

    def _attempt_swap(self, i: int) -> bool:
        """Attempt state swap between replicas i and i+1.

        Swap acceptance probability:
            alpha = min(1, exp((beta_{i+1} - beta_i) * (Cost(x_{i+1}) - Cost(x_i))))

        Args:
            i: Index of the lower-temperature replica in the pair.

        Returns:
            True if swap was accepted.
        """
        s_i = self._samplers[i]
        s_j = self._samplers[i + 1]

        beta_i = self._betas[i]
        beta_j = self._betas[i + 1]

        cost_i = s_i.current_cost
        cost_j = s_j.current_cost

        # Metropolis swap criterion
        log_alpha = (beta_j - beta_i) * (cost_j - cost_i)
        accept = log_alpha >= 0 or self.rng.random() < np.exp(log_alpha)

        self._swap_attempts[i] += 1

        if accept:
            self._swap_accepts[i] += 1

            # Swap schedules and costs between replicas
            sched_i = s_i._current_schedule
            cost_val_i = s_i._current_cost
            s_i._current_schedule = s_j._current_schedule
            s_i._current_cost = s_j._current_cost
            s_j._current_schedule = sched_i
            s_j._current_cost = cost_val_i

        return accept

    def step(self) -> None:
        """Perform one PT step: local MH steps for all replicas + swap attempts.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "ParallelTempering not initialized. Call initialize() first."
            )

        # Local MH step for each replica
        for sampler in self._samplers:
            sampler.step()

        self._step_count += 1

        # Attempt replica swaps at specified interval
        if self._step_count % self.swap_interval == 0:
            # Randomly order adjacent pairs to avoid systematic bias
            pairs = list(range(self.n_replicas - 1))
            self.rng.shuffle(pairs)
            for i in pairs:
                self._attempt_swap(i)

        # Update global best
        self._update_best()

    def run(
        self,
        n_steps: int,
        burn_in: int = 0,
        thin: int = 1,
        log_every: int = 1000,
    ) -> PTResult:
        """Run Parallel Tempering for a fixed number of steps.

        Collects samples from the coldest replica (highest beta = target
        distribution).

        Args:
            n_steps: Number of PT steps to run.
            burn_in: Number of initial steps to discard.
            thin: Keep every thin-th sample from cold chain.
            log_every: Log progress every N steps (0 = no logging).

        Returns:
            PTResult with samples, costs, and diagnostics.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "ParallelTempering not initialized. Call initialize() first."
            )

        cold_samples: list[Schedule] = []
        cold_costs: list[float] = []

        cold_idx = self.n_replicas - 1  # Coldest replica (highest beta)

        for i in range(n_steps):
            self.step()

            # Collect samples from coldest replica after burn-in
            if i >= burn_in and (i - burn_in) % thin == 0:
                cold_sampler = self._samplers[cold_idx]
                cold_samples.append(cold_sampler.current_schedule.copy())
                cold_costs.append(cold_sampler.current_cost)

            # Logging
            if log_every > 0 and (i + 1) % log_every == 0:
                swap_rates = self.swap_acceptance_rates
                avg_swap = (
                    np.mean(swap_rates) if swap_rates else 0.0
                )
                logger.info(
                    f"PT step {i + 1}/{n_steps}: "
                    f"best={self._best_cost:.2f}, "
                    f"cold_cost={self._samplers[cold_idx].current_cost:.2f}, "
                    f"avg_swap_rate={avg_swap:.3f}"
                )

        return PTResult(
            best_schedule=self._best_schedule.copy()
            if self._best_schedule is not None
            else self._samplers[cold_idx].current_schedule.copy(),
            best_cost=self._best_cost,
            cold_chain_samples=cold_samples,
            cold_chain_costs=cold_costs,
            per_replica_acceptance_rates=[
                s.acceptance_rate for s in self._samplers
            ],
            swap_acceptance_rates=self.swap_acceptance_rates,
            temperature_schedule=list(self._betas),
        )

    @property
    def swap_acceptance_rates(self) -> list[float]:
        """Swap acceptance rate for each adjacent pair."""
        rates = []
        for attempts, accepts in zip(
            self._swap_attempts, self._swap_accepts
        ):
            rates.append(accepts / attempts if attempts > 0 else 0.0)
        return rates

    @property
    def best_schedule(self) -> Schedule | None:
        """Best schedule found across all replicas."""
        return self._best_schedule

    @property
    def best_cost(self) -> float:
        """Best cost found across all replicas."""
        return self._best_cost

    @property
    def temperature_schedule(self) -> list[float]:
        """List of beta values for each replica."""
        return list(self._betas)

    @property
    def n_steps(self) -> int:
        """Total number of PT steps completed."""
        return self._step_count

    def get_replica_sampler(self, idx: int) -> MCMCSampler:
        """Get the MCMCSampler for a specific replica.

        Args:
            idx: Replica index (0 = hottest, M-1 = coldest).

        Returns:
            The MCMCSampler for the replica.
        """
        return self._samplers[idx]

    def get_cold_sampler(self) -> MCMCSampler:
        """Get the MCMCSampler for the coldest replica."""
        return self._samplers[self.n_replicas - 1]
