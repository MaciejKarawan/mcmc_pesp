"""Synthetic scenario generators for benchmarking MCMC-PESP.

Provides a factory pattern for creating 7 types of synthetic scenarios
with varying difficulty levels and full parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.random import Generator

from src.core.network import T_PERIOD, Line, Network, Station
from src.core.transfers import (
    DEFAULT_TRANSFER_PARAMS,
    IC_TO_IC_PARAMS,
    IC_TO_REGIONAL_PARAMS,
    REGIONAL_TO_BUS_PARAMS,
    Transfer,
    TransferParams,
    TransferSet,
)
from src.cost.operational import TurnaroundConstraint, TurnaroundSet


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ScenarioMetadata:
    """Metadata describing a synthetic scenario.

    Args:
        name: Human-readable name of the scenario.
        description: Brief description of the scenario.
        scenario_type: Type identifier (e.g. "hub", "chain").
        difficulty: Difficulty level.
        n_lines: Number of lines in the network.
        n_transfers: Number of transfers.
        n_turnarounds: Number of turnaround constraints.
        optimal_solution: Known optimal offsets (None if unknown).
        expected_cost_range: Approximate cost range (None if unknown).
        properties: Additional metadata properties.
    """

    name: str
    description: str
    scenario_type: str
    difficulty: Literal["easy", "medium", "hard"]
    n_lines: int
    n_transfers: int
    n_turnarounds: int = 0
    optimal_solution: dict[str, int] | None = None
    expected_cost_range: tuple[float, float] | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticScenario:
    """A complete synthetic scenario for benchmarking.

    Args:
        network: Transport network.
        transfers: Set of transfers.
        metadata: Scenario description and properties.
        turnarounds: Turnaround constraints (optional).
    """

    network: Network
    transfers: TransferSet
    metadata: ScenarioMetadata
    turnarounds: TurnaroundSet | None = None


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

_COMMON_DEFAULTS: dict[str, Any] = {
    "frequency": 60,
    "travel_time": 15,
    "dwell_time": 2,
    "walk_time": 3.0,
    "w_min": 3.0,
    "w_ideal": 8.0,
    "w_max": 30.0,
    "alpha": 2.0,
    "beta": 0.5,
    "M": 1e6,
}


def _get_param(kwargs: dict[str, Any], key: str) -> Any:
    """Return a kwarg value if present, otherwise common default."""
    return kwargs.get(key, _COMMON_DEFAULTS[key])


def _make_params(kwargs: dict[str, Any]) -> TransferParams:
    """Build TransferParams from kwargs with common defaults."""
    return TransferParams(
        w_min=_get_param(kwargs, "w_min"),
        w_ideal=_get_param(kwargs, "w_ideal"),
        w_max=_get_param(kwargs, "w_max"),
        alpha=_get_param(kwargs, "alpha"),
        beta=_get_param(kwargs, "beta"),
        M=_get_param(kwargs, "M"),
    )


def _compute_optimal_outbound_offset(
    travel_time: int,
    dwell_time: int,
    walk_time: float,
    w_ideal: float,
    frequency: int,
) -> int:
    """Compute optimal outbound offset for hub/chain scenarios.

    For a line [A, B] with travel_time, dwell_time, frequency:
      arrival_offset(1) = travel_time + dwell_time
      departure_offset(0) = dwell_time

    Waiting time = (to_offset + to.departure_offset(to_idx)
                    - from_offset - from.arrival_offset(from_idx)
                    - walk_time) % frequency

    For from_station_idx=1 (arrival at B), to_station_idx=0 (departure from B):
      w = (out_offset + dwell_time - 0 - (travel_time + dwell_time) - walk_time) % freq
      w = (out_offset - travel_time - walk_time) % freq

    For w == w_ideal:
      out_offset = (travel_time + walk_time + w_ideal) % freq
    """
    offset = int(travel_time + walk_time + w_ideal) % frequency
    return offset


# ---------------------------------------------------------------------------
# Hub scenario
# ---------------------------------------------------------------------------


def _build_hub_network(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a hub-and-spoke network.

    Central hub with N/2 inbound lines (endpoint -> hub) and N/2 outbound
    lines (hub -> endpoint). Every inbound->outbound pair is a transfer.

    Known optimum: all inbound offset=0, all outbound offset=optimal.
    Optimal cost = 0 (every transfer at w_ideal).
    """
    n_lines_map = {"easy": 6, "medium": 8, "hard": 12}
    n_lines = kwargs.get("n_lines", n_lines_map[difficulty])
    if n_lines % 2 != 0:
        raise ValueError(f"n_lines must be even for hub, got {n_lines}")

    n_in = n_lines // 2
    n_out = n_lines // 2

    frequency = _get_param(kwargs, "frequency")
    travel_time = _get_param(kwargs, "travel_time")
    dwell_time = _get_param(kwargs, "dwell_time")
    walk_time = _get_param(kwargs, "walk_time")
    params = _make_params(kwargs)

    network = Network(period=T_PERIOD)

    # Inbound lines: endpoint_i -> Hub
    for i in range(n_in):
        ep = Station(id=f"EP_IN{i}", location=f"Endpoint_IN{i}", modality="bus")
        hub = Station(id=f"HUB_IN{i}", location="Hub", modality="bus")
        line = Line(
            id=f"IN{i}",
            modality="bus",
            stations=[ep, hub],
            travel_times=[travel_time],
            dwell_times=[dwell_time, dwell_time],
            frequency=frequency,
        )
        network.add_line(line)

    # Outbound lines: Hub -> endpoint_j
    for j in range(n_out):
        hub = Station(id=f"HUB_OUT{j}", location="Hub", modality="bus")
        ep = Station(id=f"EP_OUT{j}", location=f"Endpoint_OUT{j}", modality="bus")
        line = Line(
            id=f"OUT{j}",
            modality="bus",
            stations=[hub, ep],
            travel_times=[travel_time],
            dwell_times=[dwell_time, dwell_time],
            frequency=frequency,
        )
        network.add_line(line)

    # Transfers: every inbound arrival (idx=1) -> every outbound departure (idx=0)
    transfers_dict: dict[str, Transfer] = {}
    for i in range(n_in):
        for j in range(n_out):
            tid = f"T_IN{i}_OUT{j}"
            t = Transfer(
                id=tid,
                from_line=f"IN{i}",
                from_station_idx=1,
                to_line=f"OUT{j}",
                to_station_idx=0,
                walk_time=walk_time,
                params=params,
                weight=1.0,
            )
            transfers_dict[tid] = t

    transfer_set = TransferSet(transfers=transfers_dict)

    # Optimal solution
    opt_out = _compute_optimal_outbound_offset(
        travel_time, dwell_time, walk_time, params.w_ideal, frequency
    )
    optimal = {}
    for i in range(n_in):
        optimal[f"IN{i}"] = 0
    for j in range(n_out):
        optimal[f"OUT{j}"] = opt_out

    metadata = ScenarioMetadata(
        name=f"hub_{difficulty}",
        description=f"Hub network with {n_in} inbound + {n_out} outbound lines",
        scenario_type="hub",
        difficulty=difficulty,
        n_lines=n_lines,
        n_transfers=len(transfers_dict),
        optimal_solution=optimal,
        expected_cost_range=(0.0, 0.0),
        properties={"n_inbound": n_in, "n_outbound": n_out},
    )

    return SyntheticScenario(
        network=network,
        transfers=transfer_set,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Chain scenario
# ---------------------------------------------------------------------------


def _build_chain_network(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a linear chain network.

    Lines L0(S0->S1), L1(S1->S2), ..., L_{N-1}(S_{N-1}->S_N).
    N-1 transfers, each from_station_idx=1 to to_station_idx=0.

    Known optimum: offset_0=0, offset_{i+1} = (offset_i + step) % T_PERIOD
    where step = travel_time + walk_time + w_ideal (mod frequency).
    Optimal cost = 0.
    """
    n_lines_map = {"easy": 3, "medium": 5, "hard": 8}
    n_lines = kwargs.get("n_lines", n_lines_map[difficulty])

    frequency = _get_param(kwargs, "frequency")
    travel_time = _get_param(kwargs, "travel_time")
    dwell_time = _get_param(kwargs, "dwell_time")
    walk_time = _get_param(kwargs, "walk_time")
    params = _make_params(kwargs)

    network = Network(period=T_PERIOD)

    # Create N+1 locations: S0, S1, ..., S_N
    # Line i goes from S_i to S_{i+1}
    for i in range(n_lines):
        s_from = Station(
            id=f"S{i}_L{i}", location=f"S{i}", modality="bus"
        )
        s_to = Station(
            id=f"S{i+1}_L{i}", location=f"S{i+1}", modality="bus"
        )
        line = Line(
            id=f"L{i}",
            modality="bus",
            stations=[s_from, s_to],
            travel_times=[travel_time],
            dwell_times=[dwell_time, dwell_time],
            frequency=frequency,
        )
        network.add_line(line)

    # Transfers: L_i arrival (idx=1) -> L_{i+1} departure (idx=0)
    transfers_dict: dict[str, Transfer] = {}
    for i in range(n_lines - 1):
        tid = f"T_L{i}_L{i+1}"
        t = Transfer(
            id=tid,
            from_line=f"L{i}",
            from_station_idx=1,
            to_line=f"L{i+1}",
            to_station_idx=0,
            walk_time=walk_time,
            params=params,
            weight=1.0,
        )
        transfers_dict[tid] = t

    transfer_set = TransferSet(transfers=transfers_dict)

    # Optimal offsets: propagate step by step
    step = _compute_optimal_outbound_offset(
        travel_time, dwell_time, walk_time, params.w_ideal, frequency
    )
    optimal: dict[str, int] = {}
    offset = 0
    for i in range(n_lines):
        optimal[f"L{i}"] = offset % T_PERIOD
        offset += step

    metadata = ScenarioMetadata(
        name=f"chain_{difficulty}",
        description=f"Chain of {n_lines} lines with {n_lines - 1} transfers",
        scenario_type="chain",
        difficulty=difficulty,
        n_lines=n_lines,
        n_transfers=len(transfers_dict),
        optimal_solution=optimal,
        expected_cost_range=(0.0, 0.0),
    )

    return SyntheticScenario(
        network=network,
        transfers=transfer_set,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Conflicting transfers (ring)
# ---------------------------------------------------------------------------


def _build_conflicting_transfers(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a ring network with conflicting transfer constraints.

    Lines form a ring: L0(S0->S1), L1(S1->S2), ..., L_{N-1}(S_{N-1}->S0).
    Each transfer has a DIFFERENT w_ideal so sum of constraints != 0 mod freq.

    No solution can achieve cost = 0 (conflicting).
    """
    n_lines_map = {"easy": 3, "medium": 4, "hard": 5}
    n_lines = kwargs.get("n_lines", n_lines_map[difficulty])

    frequency = _get_param(kwargs, "frequency")
    travel_time = _get_param(kwargs, "travel_time")
    dwell_time = _get_param(kwargs, "dwell_time")
    walk_time = _get_param(kwargs, "walk_time")

    # Different w_ideal for each transfer to create conflicts
    base_ideals = kwargs.get("w_ideals", None)
    if base_ideals is None:
        # Generate ideals that don't sum to 0 mod frequency
        # e.g. 8, 12, 15, 10, 7 — partial sums won't align
        candidate_ideals = [8.0, 12.0, 15.0, 10.0, 7.0, 11.0, 14.0, 9.0]
        base_ideals = candidate_ideals[:n_lines]

    network = Network(period=T_PERIOD)

    # Ring of stations: S0, S1, ..., S_{N-1}
    for i in range(n_lines):
        j = (i + 1) % n_lines
        s_from = Station(
            id=f"S{i}_L{i}", location=f"S{i}", modality="bus"
        )
        s_to = Station(
            id=f"S{j}_L{i}", location=f"S{j}", modality="bus"
        )
        line = Line(
            id=f"L{i}",
            modality="bus",
            stations=[s_from, s_to],
            travel_times=[travel_time],
            dwell_times=[dwell_time, dwell_time],
            frequency=frequency,
        )
        network.add_line(line)

    # Transfers: L_i arrival (idx=1) -> L_{i+1 mod N} departure (idx=0)
    transfers_dict: dict[str, Transfer] = {}
    for i in range(n_lines):
        j = (i + 1) % n_lines
        w_ideal_i = base_ideals[i]
        params_i = TransferParams(
            w_min=_get_param(kwargs, "w_min"),
            w_ideal=w_ideal_i,
            w_max=_get_param(kwargs, "w_max"),
            alpha=_get_param(kwargs, "alpha"),
            beta=_get_param(kwargs, "beta"),
            M=_get_param(kwargs, "M"),
        )
        tid = f"T_L{i}_L{j}"
        t = Transfer(
            id=tid,
            from_line=f"L{i}",
            from_station_idx=1,
            to_line=f"L{j}",
            to_station_idx=0,
            walk_time=walk_time,
            params=params_i,
            weight=1.0,
        )
        transfers_dict[tid] = t

    transfer_set = TransferSet(transfers=transfers_dict)

    # Verify that the constraints actually conflict
    # Sum of required offsets steps = sum(travel_time + walk_time + w_ideal_i)
    # Must NOT be 0 mod frequency for a conflict
    total_required = sum(
        travel_time + walk_time + ideal for ideal in base_ideals
    )
    has_conflict = (int(total_required) % frequency) != 0

    metadata = ScenarioMetadata(
        name=f"conflicting_{difficulty}",
        description=(
            f"Ring of {n_lines} lines with conflicting w_ideal values. "
            f"Constraint sum = {int(total_required)} % {frequency} = "
            f"{int(total_required) % frequency} (conflict={has_conflict})"
        ),
        scenario_type="conflicting",
        difficulty=difficulty,
        n_lines=n_lines,
        n_transfers=len(transfers_dict),
        optimal_solution=None,
        expected_cost_range=None,
        properties={
            "w_ideals": base_ideals,
            "has_conflict": has_conflict,
            "constraint_sum_mod_freq": int(total_required) % frequency,
        },
    )

    return SyntheticScenario(
        network=network,
        transfers=transfer_set,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Weighted transfers
# ---------------------------------------------------------------------------


def _build_weighted_transfers(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a hub network with heterogeneous transfer weights.

    Reuses hub topology but randomly assigns high/low weights.
    Some transfers have weight_ratio times higher weight than others.
    """
    weight_ratio = kwargs.get("weight_ratio", 100.0)
    seed = kwargs.get("seed", None)

    # Build base hub
    hub = _build_hub_network(difficulty, **kwargs)

    rng = np.random.default_rng(seed)

    # Determine number of high-weight transfers
    n_high_map = {"easy": 2, "medium": 4, "hard": 6}
    n_high = min(n_high_map[difficulty], len(hub.transfers))

    # Randomly select high-weight transfers
    transfer_ids = list(hub.transfers.transfers.keys())
    high_ids = set(rng.choice(transfer_ids, size=n_high, replace=False))

    for tid, transfer in hub.transfers.transfers.items():
        if tid in high_ids:
            # Modify weight in-place (Transfer is a mutable dataclass)
            transfer.weight = weight_ratio
        else:
            transfer.weight = 1.0

    metadata = ScenarioMetadata(
        name=f"weighted_{difficulty}",
        description=(
            f"Hub network with {n_high} high-weight (x{weight_ratio}) transfers"
        ),
        scenario_type="weighted",
        difficulty=difficulty,
        n_lines=hub.metadata.n_lines,
        n_transfers=hub.metadata.n_transfers,
        optimal_solution=None,  # No simple analytical optimum with weights
        expected_cost_range=None,
        properties={
            "weight_ratio": weight_ratio,
            "n_high_weight": n_high,
            "high_weight_ids": sorted(high_ids),
        },
    )

    return SyntheticScenario(
        network=hub.network,
        transfers=hub.transfers,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Full cost network (with turnarounds)
# ---------------------------------------------------------------------------


def _build_full_cost_network(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a hub with bidirectional lines and turnaround constraints.

    For each pair i: Out_i (hub -> endpoint) and Ret_i (endpoint -> hub).
    Transfers: Ret_i arrival at hub -> Out_j departure from hub.
    Turnarounds: Out_i arrival at endpoint -> Ret_i departure from endpoint.
    """
    n_pairs_map = {"easy": 2, "medium": 4, "hard": 6}
    n_pairs = kwargs.get("n_pairs", n_pairs_map[difficulty])

    frequency = _get_param(kwargs, "frequency")
    travel_time = _get_param(kwargs, "travel_time")
    dwell_time = _get_param(kwargs, "dwell_time")
    walk_time = _get_param(kwargs, "walk_time")
    params = _make_params(kwargs)
    min_turnaround = kwargs.get("min_turnaround", 10.0)

    network = Network(period=T_PERIOD)

    # Outbound lines: Hub -> Endpoint_i
    for i in range(n_pairs):
        hub_s = Station(id=f"HUB_O{i}", location="Hub", modality="bus")
        ep_s = Station(
            id=f"EP{i}_O{i}", location=f"Endpoint_{i}", modality="bus"
        )
        line = Line(
            id=f"OUT{i}",
            modality="bus",
            stations=[hub_s, ep_s],
            travel_times=[travel_time],
            dwell_times=[dwell_time, dwell_time],
            frequency=frequency,
        )
        network.add_line(line)

    # Return lines: Endpoint_i -> Hub
    for i in range(n_pairs):
        ep_s = Station(
            id=f"EP{i}_R{i}", location=f"Endpoint_{i}", modality="bus"
        )
        hub_s = Station(id=f"HUB_R{i}", location="Hub", modality="bus")
        line = Line(
            id=f"RET{i}",
            modality="bus",
            stations=[ep_s, hub_s],
            travel_times=[travel_time],
            dwell_times=[dwell_time, dwell_time],
            frequency=frequency,
        )
        network.add_line(line)

    # Transfers: Ret_i arrival at hub (idx=1) -> Out_j departure from hub (idx=0)
    transfers_dict: dict[str, Transfer] = {}
    for i in range(n_pairs):
        for j in range(n_pairs):
            if i == j:
                continue  # Skip same pair
            tid = f"T_RET{i}_OUT{j}"
            t = Transfer(
                id=tid,
                from_line=f"RET{i}",
                from_station_idx=1,
                to_line=f"OUT{j}",
                to_station_idx=0,
                walk_time=walk_time,
                params=params,
                weight=1.0,
            )
            transfers_dict[tid] = t

    transfer_set = TransferSet(transfers=transfers_dict)

    # Turnaround constraints: Out_i endpoint (idx=1) -> Ret_i endpoint (idx=0)
    turnaround_dict: dict[str, TurnaroundConstraint] = {}
    for i in range(n_pairs):
        tc = TurnaroundConstraint(
            id=f"TA_OUT{i}_RET{i}",
            end_line=f"OUT{i}",
            end_station_idx=1,
            start_line=f"RET{i}",
            start_station_idx=0,
            min_turnaround=min_turnaround,
        )
        turnaround_dict[tc.id] = tc

    turnaround_set = TurnaroundSet(constraints=turnaround_dict)

    metadata = ScenarioMetadata(
        name=f"full_cost_{difficulty}",
        description=(
            f"{n_pairs} bidirectional pairs with transfers and turnaround constraints"
        ),
        scenario_type="full_cost",
        difficulty=difficulty,
        n_lines=2 * n_pairs,
        n_transfers=len(transfers_dict),
        n_turnarounds=n_pairs,
        optimal_solution=None,
        expected_cost_range=None,
        properties={"n_pairs": n_pairs, "min_turnaround": min_turnaround},
    )

    return SyntheticScenario(
        network=network,
        transfers=transfer_set,
        metadata=metadata,
        turnarounds=turnaround_set,
    )


# ---------------------------------------------------------------------------
# Multi-hub mesh (wild scenario 1)
# ---------------------------------------------------------------------------


def _build_multi_hub_mesh(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a deterministic multi-hub mesh network.

    3 hubs (Central, North, South) connected by IC lines, with regional
    and local feeder lines. Combines multiple difficulties:
    - Conflicting constraints (many lines at each hub)
    - Heterogeneous weights (IC=50, regional=10, local=1)
    - Mixed frequencies (IC=120, regional=60, local=30)
    - Turnaround constraints at terminals
    - Multi-modal TransferParams
    """
    ic_freq = kwargs.get("ic_frequency", 120)
    reg_freq = kwargs.get("regional_frequency", _get_param(kwargs, "frequency"))
    local_freq = kwargs.get("local_frequency", 30)
    travel_time = _get_param(kwargs, "travel_time")
    dwell_time = _get_param(kwargs, "dwell_time")
    walk_time = _get_param(kwargs, "walk_time")

    # Difficulty scaling
    n_suburbs_map = {"easy": 2, "medium": 3, "hard": 3}
    n_local_map = {"easy": 1, "medium": 2, "hard": 2}
    n_suburbs = n_suburbs_map[difficulty]
    n_local_per_hub = n_local_map[difficulty]
    ic_min_turnaround = kwargs.get("ic_min_turnaround", 15.0)
    reg_min_turnaround = kwargs.get("reg_min_turnaround", 8.0)

    hub_names = ["Central", "North", "South"]
    network = Network(period=T_PERIOD)
    transfers_dict: dict[str, Transfer] = {}
    turnaround_dict: dict[str, TurnaroundConstraint] = {}

    line_counter = 0

    def _add_line(lid: str, modality: str, loc_from: str, loc_to: str,
                  freq: int, tt: int = travel_time) -> str:
        nonlocal line_counter
        s_from = Station(id=f"{loc_from}_{lid}", location=loc_from, modality=modality)
        s_to = Station(id=f"{loc_to}_{lid}", location=loc_to, modality=modality)
        line = Line(
            id=lid,
            modality=modality,
            stations=[s_from, s_to],
            travel_times=[tt],
            dwell_times=[dwell_time, dwell_time],
            frequency=freq,
        )
        network.add_line(line)
        line_counter += 1
        return lid

    # --- IC lines between hubs ---
    ic_pairs = [("Central", "North"), ("Central", "South"), ("North", "South")]
    ic_line_ids: list[str] = []
    for h1, h2 in ic_pairs:
        lid = f"IC_{h1}_{h2}"
        _add_line(lid, "IC", h1, h2, ic_freq, tt=travel_time * 2)
        ic_line_ids.append(lid)

    # IC-to-IC transfers at each hub where two IC lines meet
    ic_params = kwargs.get("ic_params", IC_TO_IC_PARAMS)
    for hub_name in hub_names:
        # Find IC lines serving this hub
        lines_at_hub: list[tuple[str, int]] = []  # (line_id, station_idx)
        for lid in ic_line_ids:
            line = network.get_line(lid)
            for idx, st in enumerate(line.stations):
                if st.location == hub_name:
                    lines_at_hub.append((lid, idx))

        # Create transfers between IC lines at this hub
        for a, (lid_a, idx_a) in enumerate(lines_at_hub):
            for b, (lid_b, idx_b) in enumerate(lines_at_hub):
                if a == b:
                    continue
                # Only transfer from arriving (idx=1) to departing (idx=0)
                if idx_a == 1 and idx_b == 0:
                    tid = f"T_IC_{lid_a}_{lid_b}_{hub_name}"
                    t = Transfer(
                        id=tid,
                        from_line=lid_a,
                        from_station_idx=idx_a,
                        to_line=lid_b,
                        to_station_idx=idx_b,
                        walk_time=walk_time,
                        params=ic_params,
                        weight=50.0,
                    )
                    transfers_dict[tid] = t

    # IC turnaround at terminal stations
    for lid in ic_line_ids:
        tc = TurnaroundConstraint(
            id=f"TA_{lid}",
            end_line=lid,
            end_station_idx=1,
            start_line=lid,
            start_station_idx=0,
            min_turnaround=ic_min_turnaround,
        )
        turnaround_dict[tc.id] = tc

    # --- Regional lines from each hub to suburbs ---
    ic_to_reg_params = kwargs.get("ic_to_regional_params", IC_TO_REGIONAL_PARAMS)
    reg_to_bus_params = kwargs.get("reg_to_bus_params", REGIONAL_TO_BUS_PARAMS)
    regional_line_ids: list[str] = []

    for hub_name in hub_names:
        for s in range(n_suburbs):
            suburb = f"{hub_name}_Sub{s}"
            lid = f"REG_{hub_name}_{s}"
            _add_line(lid, "regional", hub_name, suburb, reg_freq)
            regional_line_ids.append(lid)

            # Turnaround at suburb terminal
            tc = TurnaroundConstraint(
                id=f"TA_{lid}",
                end_line=lid,
                end_station_idx=1,
                start_line=lid,
                start_station_idx=0,
                min_turnaround=reg_min_turnaround,
            )
            turnaround_dict[tc.id] = tc

            # Transfer: IC arriving at hub -> regional departing from hub
            for ic_lid in ic_line_ids:
                ic_line = network.get_line(ic_lid)
                for idx, st in enumerate(ic_line.stations):
                    if st.location == hub_name and idx == 1:
                        tid = f"T_IC2REG_{ic_lid}_{lid}"
                        t = Transfer(
                            id=tid,
                            from_line=ic_lid,
                            from_station_idx=1,
                            to_line=lid,
                            to_station_idx=0,
                            walk_time=walk_time,
                            params=ic_to_reg_params,
                            weight=10.0,
                        )
                        transfers_dict[tid] = t

            # Transfer: regional arriving at hub -> IC departing from hub
            for ic_lid in ic_line_ids:
                ic_line = network.get_line(ic_lid)
                for idx, st in enumerate(ic_line.stations):
                    if st.location == hub_name and idx == 0:
                        tid = f"T_REG2IC_{lid}_{ic_lid}"
                        t = Transfer(
                            id=tid,
                            from_line=lid,
                            from_station_idx=1,  # regional arrives at hub
                            to_line=ic_lid,
                            to_station_idx=0,  # IC departs from hub
                            walk_time=walk_time,
                            params=ic_to_reg_params,
                            weight=10.0,
                        )
                        transfers_dict[tid] = t

    # --- Local lines at each hub ---
    local_line_ids: list[str] = []
    for hub_name in hub_names:
        for k in range(n_local_per_hub):
            stop = f"{hub_name}_Local{k}"
            lid = f"LOC_{hub_name}_{k}"
            _add_line(lid, "bus", hub_name, stop, local_freq, tt=travel_time // 2)
            local_line_ids.append(lid)

            # Transfer: regional arriving at hub -> local departing from hub
            for reg_lid in regional_line_ids:
                reg_line = network.get_line(reg_lid)
                # Regional arrives at hub (idx=1 means endpoint, but for regional
                # we need to check: reg goes hub->suburb, so it doesn't arrive at hub)
                # Actually regional: hub -> suburb. So arrival at hub is only
                # for return trips. For simplicity, let's create feeder transfers
                # from IC arriving at hub to local lines.
                pass

            # Transfer: IC arriving at hub -> local departing
            for ic_lid in ic_line_ids:
                ic_line = network.get_line(ic_lid)
                for idx, st in enumerate(ic_line.stations):
                    if st.location == hub_name and idx == 1:
                        tid = f"T_IC2LOC_{ic_lid}_{lid}"
                        t = Transfer(
                            id=tid,
                            from_line=ic_lid,
                            from_station_idx=1,
                            to_line=lid,
                            to_station_idx=0,
                            walk_time=walk_time,
                            params=reg_to_bus_params,
                            weight=1.0,
                        )
                        transfers_dict[tid] = t

            # Transfer: local arriving at hub stop -> regional departing
            # Local goes hub->stop, so local arrives at stop (idx=1), not hub.
            # To create hub transfer, we'd need a return local line.
            # For simplicity, create inter-regional transfers at the hub
            # between different regional lines.

    # Inter-regional transfers at each hub (regional arriving -> regional departing)
    # Regional lines go hub->suburb, so they depart from hub (idx=0)
    # We need lines arriving AT the hub. Since regional lines are one-way hub->suburb,
    # we reinterpret: at each hub, IC arrives and regional/local depart.
    # This already captures the cascade IC -> regional -> local.

    transfer_set = TransferSet(transfers=transfers_dict)
    turnaround_set = TurnaroundSet(constraints=turnaround_dict)

    n_lines_total = len(network.lines)

    metadata = ScenarioMetadata(
        name=f"multi_hub_mesh_{difficulty}",
        description=(
            f"Multi-hub mesh: {len(hub_names)} hubs, "
            f"{len(ic_line_ids)} IC + {len(regional_line_ids)} regional + "
            f"{len(local_line_ids)} local lines"
        ),
        scenario_type="multi_hub_mesh",
        difficulty=difficulty,
        n_lines=n_lines_total,
        n_transfers=len(transfers_dict),
        n_turnarounds=len(turnaround_dict),
        optimal_solution=None,
        expected_cost_range=None,
        properties={
            "n_hubs": len(hub_names),
            "n_ic_lines": len(ic_line_ids),
            "n_regional_lines": len(regional_line_ids),
            "n_local_lines": len(local_line_ids),
            "ic_frequency": ic_freq,
            "regional_frequency": reg_freq,
            "local_frequency": local_freq,
        },
    )

    return SyntheticScenario(
        network=network,
        transfers=transfer_set,
        metadata=metadata,
        turnarounds=turnaround_set,
    )


# ---------------------------------------------------------------------------
# Random network (wild scenario 2)
# ---------------------------------------------------------------------------


def _build_random_network(
    difficulty: Literal["easy", "medium", "hard"],
    **kwargs: Any,
) -> SyntheticScenario:
    """Build a random network with parameterized topology.

    1. Create n_stations locations.
    2. Create n_lines lines, each connecting 2-4 random stations.
    3. Assign modality (IC/regional/bus) and frequency.
    4. Create transfers where lines share locations.
    5. Filter by transfer_density.
    6. Assign random weights.
    7. Optionally add turnaround constraints.
    """
    defaults = {
        "easy": {"n_stations": 8, "n_lines": 6, "transfer_density": 0.5},
        "medium": {"n_stations": 15, "n_lines": 12, "transfer_density": 0.6},
        "hard": {"n_stations": 25, "n_lines": 20, "transfer_density": 0.7},
    }
    d = defaults[difficulty]

    n_stations = kwargs.get("n_stations", d["n_stations"])
    n_lines = kwargs.get("n_lines", d["n_lines"])
    transfer_density = kwargs.get("transfer_density", d["transfer_density"])
    seed = kwargs.get("seed", None)
    add_turnarounds = kwargs.get("add_turnarounds", True)

    travel_time = _get_param(kwargs, "travel_time")
    dwell_time = _get_param(kwargs, "dwell_time")
    walk_time = _get_param(kwargs, "walk_time")
    params = _make_params(kwargs)

    rng = np.random.default_rng(seed)

    # Modality distribution and frequencies
    modalities = ["IC", "regional", "bus"]
    modality_probs = [0.2, 0.4, 0.4]
    freq_map = {
        "IC": kwargs.get("ic_frequency", 120),
        "regional": kwargs.get("regional_frequency", _get_param(kwargs, "frequency")),
        "bus": kwargs.get("bus_frequency", 30),
    }

    # Create station locations
    locations = [f"Loc{i}" for i in range(n_stations)]

    network = Network(period=T_PERIOD)
    line_station_map: dict[str, list[str]] = {}  # line_id -> [locations]

    for i in range(n_lines):
        # Random number of stations per line (2-4)
        n_stops = int(rng.integers(2, 5))
        n_stops = min(n_stops, n_stations)

        # Pick random locations (without replacement)
        line_locs = list(rng.choice(locations, size=n_stops, replace=False))

        # Random modality
        modality = rng.choice(modalities, p=modality_probs)
        freq = freq_map[modality]

        lid = f"L{i}"
        stations = [
            Station(id=f"{loc}_{lid}", location=loc, modality=modality)
            for loc in line_locs
        ]
        travel_times_list = [travel_time] * (len(stations) - 1)
        dwell_times_list = [dwell_time] * len(stations)

        line = Line(
            id=lid,
            modality=modality,
            stations=stations,
            travel_times=travel_times_list,
            dwell_times=dwell_times_list,
            frequency=freq,
        )
        network.add_line(line)
        line_station_map[lid] = line_locs

    # Find shared locations and create potential transfers
    # Build location -> [(line_id, station_idx)] mapping
    loc_to_lines: dict[str, list[tuple[str, int]]] = {}
    for lid, locs in line_station_map.items():
        for idx, loc in enumerate(locs):
            if loc not in loc_to_lines:
                loc_to_lines[loc] = []
            loc_to_lines[loc].append((lid, idx))

    # Create transfers where lines meet at the same location
    potential_transfers: list[tuple[str, int, str, int, str]] = []
    for loc, line_entries in loc_to_lines.items():
        if len(line_entries) < 2:
            continue
        for a, (lid_a, idx_a) in enumerate(line_entries):
            for b, (lid_b, idx_b) in enumerate(line_entries):
                if a == b:
                    continue
                if lid_a == lid_b:
                    continue
                # Transfer from line_a arriving at idx_a to line_b departing at idx_b
                # Only meaningful if arriving (not first station) and departing
                if idx_a > 0:
                    potential_transfers.append((lid_a, idx_a, lid_b, idx_b, loc))

    # Filter by density
    transfers_dict: dict[str, Transfer] = {}
    if potential_transfers:
        n_keep = max(1, int(len(potential_transfers) * transfer_density))
        kept_indices = rng.choice(
            len(potential_transfers), size=min(n_keep, len(potential_transfers)),
            replace=False,
        )
        for idx in kept_indices:
            lid_a, idx_a, lid_b, idx_b, loc = potential_transfers[idx]
            weight = float(rng.uniform(1.0, 50.0))
            tid = f"T_{lid_a}_{lid_b}_{loc}"
            # Avoid duplicate IDs
            if tid in transfers_dict:
                tid = f"{tid}_{idx}"
            t = Transfer(
                id=tid,
                from_line=lid_a,
                from_station_idx=idx_a,
                to_line=lid_b,
                to_station_idx=idx_b,
                walk_time=walk_time,
                params=params,
                weight=weight,
            )
            transfers_dict[tid] = t

    # Ensure at least 1 transfer: if none found, create one between first two lines
    if not transfers_dict and n_lines >= 2:
        # Force a shared location
        lid_a = "L0"
        lid_b = "L1"
        line_a = network.get_line(lid_a)
        line_b = network.get_line(lid_b)
        tid = "T_forced_L0_L1"
        t = Transfer(
            id=tid,
            from_line=lid_a,
            from_station_idx=len(line_a.stations) - 1,
            to_line=lid_b,
            to_station_idx=0,
            walk_time=walk_time,
            params=params,
            weight=1.0,
        )
        transfers_dict[tid] = t

    transfer_set = TransferSet(transfers=transfers_dict)

    # Optionally add turnaround constraints at line terminals
    turnaround_dict: dict[str, TurnaroundConstraint] = {}
    if add_turnarounds:
        for lid in network.line_ids:
            line = network.get_line(lid)
            if rng.random() < 0.3:  # 30% chance of turnaround
                min_ta = 10.0 if line.modality == "IC" else 5.0
                tc = TurnaroundConstraint(
                    id=f"TA_{lid}",
                    end_line=lid,
                    end_station_idx=len(line.stations) - 1,
                    start_line=lid,
                    start_station_idx=0,
                    min_turnaround=min_ta,
                )
                turnaround_dict[tc.id] = tc

    turnaround_set = TurnaroundSet(constraints=turnaround_dict) if turnaround_dict else None

    metadata = ScenarioMetadata(
        name=f"random_network_{difficulty}_seed{seed}",
        description=(
            f"Random network: {n_stations} stations, {n_lines} lines, "
            f"{len(transfers_dict)} transfers (density={transfer_density})"
        ),
        scenario_type="random_network",
        difficulty=difficulty,
        n_lines=n_lines,
        n_transfers=len(transfers_dict),
        n_turnarounds=len(turnaround_dict),
        optimal_solution=None,
        expected_cost_range=None,
        properties={
            "n_stations": n_stations,
            "transfer_density": transfer_density,
            "seed": seed,
        },
    )

    return SyntheticScenario(
        network=network,
        transfers=transfer_set,
        metadata=metadata,
        turnarounds=turnaround_set,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class SyntheticScenarioFactory:
    """Factory for creating synthetic benchmark scenarios.

    Supports 7 scenario types with 3 difficulty levels each.
    All builders accept **kwargs for full parameterization.

    Usage:
        scenario = SyntheticScenarioFactory.create("hub", difficulty="easy")
        scenario = SyntheticScenarioFactory.create("hub", difficulty="medium",
            frequency=30, w_ideal=12.0)
    """

    SCENARIO_TYPES = [
        "hub",
        "chain",
        "conflicting",
        "weighted",
        "full_cost",
        "multi_hub_mesh",
        "random_network",
    ]

    _BUILDERS = {
        "hub": _build_hub_network,
        "chain": _build_chain_network,
        "conflicting": _build_conflicting_transfers,
        "weighted": _build_weighted_transfers,
        "full_cost": _build_full_cost_network,
        "multi_hub_mesh": _build_multi_hub_mesh,
        "random_network": _build_random_network,
    }

    @classmethod
    def create(
        cls,
        scenario_type: str,
        difficulty: Literal["easy", "medium", "hard"] = "easy",
        **kwargs: Any,
    ) -> SyntheticScenario:
        """Create a synthetic scenario.

        Args:
            scenario_type: One of SCENARIO_TYPES.
            difficulty: "easy", "medium", or "hard".
            **kwargs: Builder-specific parameters. Common params:
                frequency, travel_time, dwell_time, walk_time,
                w_min, w_ideal, w_max, alpha, beta, M, n_lines, seed.

        Returns:
            A SyntheticScenario ready for benchmarking.

        Raises:
            ValueError: If scenario_type is unknown.
        """
        if scenario_type not in cls._BUILDERS:
            raise ValueError(
                f"Unknown scenario type '{scenario_type}'. "
                f"Available: {cls.SCENARIO_TYPES}"
            )

        builder = cls._BUILDERS[scenario_type]
        return builder(difficulty, **kwargs)

    @classmethod
    def list_scenarios(cls) -> list[str]:
        """Return list of available scenario types."""
        return list(cls.SCENARIO_TYPES)
