"""Simple loader for PESPlib raw format (single CSV file).

PESPlib raw format: activity_id; from_event; to_event; lower; upper; weight
Period is assumed to be 60 unless specified otherwise.

This loader treats the PESP problem as a general constraint satisfaction problem
where each event is a "pseudo-line" with a single offset to be determined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PESPActivity:
    """A single PESP activity (constraint).

    Args:
        id: Activity identifier.
        from_event: Source event ID.
        to_event: Target event ID.
        lower: Lower bound on duration.
        upper: Upper bound on duration.
        weight: Passenger weight.
    """

    id: int
    from_event: int
    to_event: int
    lower: int
    upper: int
    weight: int


@dataclass
class RawPESPInstance:
    """A raw PESP instance loaded from PESPlib format.

    Args:
        name: Instance name.
        activities: List of activities (constraints).
        n_events: Number of events (derived from activities).
        period: Time period (default 60).
    """

    name: str
    activities: list[PESPActivity]
    n_events: int
    period: int = 60

    def compute_objective(self, times: dict[int, int]) -> float:
        """Compute the PESPlib objective (weighted slack sum).

        Objective = sum over activities of weight * duration
        where duration = (time[to] - time[from]) mod T

        Args:
            times: Dictionary mapping event_id to assigned time.

        Returns:
            Objective value.
        """
        total = 0.0
        for act in self.activities:
            if act.from_event in times and act.to_event in times:
                duration = (times[act.to_event] - times[act.from_event]) % self.period
                total += act.weight * duration
        return total

    def check_feasibility(self, times: dict[int, int]) -> tuple[bool, list[int]]:
        """Check if a solution satisfies all constraints.

        Args:
            times: Dictionary mapping event_id to assigned time.

        Returns:
            Tuple of (is_feasible, list_of_violated_activity_ids).
        """
        violated = []
        for act in self.activities:
            if act.from_event in times and act.to_event in times:
                duration = (times[act.to_event] - times[act.from_event]) % self.period
                if duration < act.lower or duration > act.upper:
                    violated.append(act.id)
        return len(violated) == 0, violated

    def get_event_ids(self) -> set[int]:
        """Get all event IDs referenced in activities."""
        events = set()
        for act in self.activities:
            events.add(act.from_event)
            events.add(act.to_event)
        return events


def load_pesplib_raw(path: Path | str, period: int = 60) -> RawPESPInstance:
    """Load a PESPlib instance from raw format.

    Args:
        path: Path to the .txt file.
        period: Time period (default 60).

    Returns:
        RawPESPInstance.
    """
    path = Path(path)
    activities = []
    events = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 6:
                continue

            try:
                act = PESPActivity(
                    id=int(parts[0]),
                    from_event=int(parts[1]),
                    to_event=int(parts[2]),
                    lower=int(parts[3]),
                    upper=int(parts[4]),
                    weight=int(parts[5]),
                )
                activities.append(act)
                events.add(act.from_event)
                events.add(act.to_event)
            except ValueError:
                continue

    logger.info(f"Loaded {len(activities)} activities, {len(events)} events from {path.name}")

    return RawPESPInstance(
        name=path.stem,
        activities=activities,
        n_events=len(events),
        period=period,
    )


@dataclass
class PESPSchedule:
    """A schedule (time assignment) for a PESP instance.

    Args:
        times: Dictionary mapping event_id to time (0 to period-1).
        period: Time period.
    """

    times: dict[int, int]
    period: int

    def copy(self) -> PESPSchedule:
        """Create a copy of this schedule."""
        return PESPSchedule(times=self.times.copy(), period=self.period)

    def shift_event(self, event_id: int, delta: int) -> None:
        """Shift an event's time by delta (mod period)."""
        if event_id in self.times:
            self.times[event_id] = (self.times[event_id] + delta) % self.period

    @classmethod
    def random(cls, instance: RawPESPInstance, rng=None) -> PESPSchedule:
        """Create a random schedule for an instance."""
        if rng is None:
            rng = np.random.default_rng()

        event_ids = instance.get_event_ids()
        times = {eid: int(rng.integers(0, instance.period)) for eid in event_ids}
        return cls(times=times, period=instance.period)

    @classmethod
    def zeros(cls, instance: RawPESPInstance) -> PESPSchedule:
        """Create a schedule with all times set to 0."""
        event_ids = instance.get_event_ids()
        times = {eid: 0 for eid in event_ids}
        return cls(times=times, period=instance.period)


@dataclass
class PESPCostFunction:
    """Cost function for PESP (weighted slack + violation penalty).

    This cost function penalizes constraint violations with a large barrier,
    and otherwise minimizes the weighted slack.

    Args:
        instance: The PESP instance.
        violation_penalty: Penalty per violated constraint.
    """

    instance: RawPESPInstance
    violation_penalty: float = 1e6

    def __call__(self, schedule: PESPSchedule) -> float:
        """Compute cost for a schedule."""
        total_cost = 0.0

        for act in self.instance.activities:
            if act.from_event in schedule.times and act.to_event in schedule.times:
                duration = (schedule.times[act.to_event] - schedule.times[act.from_event]) % self.instance.period

                # Check feasibility
                if duration < act.lower or duration > act.upper:
                    # Compute slack (distance to feasible region)
                    if duration < act.lower:
                        slack = act.lower - duration
                    else:
                        slack = duration - act.upper
                    total_cost += self.violation_penalty * slack
                else:
                    # Feasible: add weighted duration
                    total_cost += act.weight * duration

        return total_cost

    def delta(
        self,
        old_schedule: PESPSchedule,
        new_schedule: PESPSchedule,
        changed_events: set[int],
    ) -> float:
        """Compute incremental cost change.

        Only recomputes cost for activities involving changed events.
        """
        affected_activities = [
            act for act in self.instance.activities
            if act.from_event in changed_events or act.to_event in changed_events
        ]

        old_cost = 0.0
        new_cost = 0.0

        for act in affected_activities:
            # Old cost
            if act.from_event in old_schedule.times and act.to_event in old_schedule.times:
                duration = (old_schedule.times[act.to_event] - old_schedule.times[act.from_event]) % self.instance.period
                if duration < act.lower or duration > act.upper:
                    if duration < act.lower:
                        slack = act.lower - duration
                    else:
                        slack = duration - act.upper
                    old_cost += self.violation_penalty * slack
                else:
                    old_cost += act.weight * duration

            # New cost
            if act.from_event in new_schedule.times and act.to_event in new_schedule.times:
                duration = (new_schedule.times[act.to_event] - new_schedule.times[act.from_event]) % self.instance.period
                if duration < act.lower or duration > act.upper:
                    if duration < act.lower:
                        slack = act.lower - duration
                    else:
                        slack = duration - act.upper
                    new_cost += self.violation_penalty * slack
                else:
                    new_cost += act.weight * duration

        return new_cost - old_cost


@dataclass
class PESPSingleEventShift:
    """Proposal that shifts a single event's time."""

    instance: RawPESPInstance
    sigma: float = 5.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    @property
    def name(self) -> str:
        return "single_event_shift"

    def propose(
        self,
        schedule: PESPSchedule,
    ) -> tuple[PESPSchedule, float, set[int]]:
        """Propose shifting a single event."""
        event_ids = list(schedule.times.keys())
        if not event_ids:
            return schedule.copy(), 1.0, set()

        event_id = int(self.rng.choice(event_ids))

        delta = int(round(self.rng.normal(0, self.sigma)))

        new_schedule = schedule.copy()
        new_schedule.shift_event(event_id, delta)

        return new_schedule, 1.0, {event_id}


def run_pesp_mcmc(
    instance: RawPESPInstance,
    n_steps: int = 10000,
    beta: float = 0.001,
    seed: int = 42,
    log_every: int = 2000,
) -> tuple[PESPSchedule, float, list[float]]:
    """Run MCMC on a PESP instance.

    Args:
        instance: The PESP instance.
        n_steps: Number of MCMC steps.
        beta: Inverse temperature.
        seed: Random seed.
        log_every: Log interval.

    Returns:
        Tuple of (best_schedule, best_cost, cost_history).
    """
    rng = np.random.default_rng(seed)

    cost_fn = PESPCostFunction(instance)
    proposal = PESPSingleEventShift(instance, sigma=5.0, rng=rng)

    # Initialize
    current = PESPSchedule.random(instance, rng)
    current_cost = cost_fn(current)

    best = current.copy()
    best_cost = current_cost

    costs = [current_cost]
    n_accepted = 0

    for step in range(n_steps):
        # Propose
        new_schedule, hastings, changed = proposal.propose(current)

        # Compute cost
        delta_cost = cost_fn.delta(current, new_schedule, changed)
        new_cost = current_cost + delta_cost

        # Accept/reject
        log_alpha = -beta * delta_cost + np.log(hastings)
        accept = log_alpha >= 0 or rng.random() < np.exp(log_alpha)

        if accept:
            current = new_schedule
            current_cost = new_cost
            n_accepted += 1

            if new_cost < best_cost:
                best = new_schedule.copy()
                best_cost = new_cost

        costs.append(current_cost)

        if log_every > 0 and (step + 1) % log_every == 0:
            accept_rate = n_accepted / (step + 1)
            logger.info(
                f"Step {step + 1}/{n_steps}: cost={current_cost:.2f}, "
                f"best={best_cost:.2f}, accept_rate={accept_rate:.3f}"
            )

    return best, best_cost, costs
