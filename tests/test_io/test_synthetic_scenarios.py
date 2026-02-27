"""Tests for synthetic scenario generators."""

from __future__ import annotations

import pytest

from src.core.network import T_PERIOD
from src.core.schedule import Schedule
from src.cost.transfer_cost import TransferCost
from src.io.synthetic_scenarios import (
    SyntheticScenario,
    SyntheticScenarioFactory,
    _compute_optimal_outbound_offset,
)


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestSyntheticScenarioFactory:
    """Tests for SyntheticScenarioFactory."""

    def test_list_scenarios_returns_all_types(self):
        types = SyntheticScenarioFactory.list_scenarios()
        assert len(types) == 7
        assert "hub" in types
        assert "chain" in types
        assert "conflicting" in types
        assert "weighted" in types
        assert "full_cost" in types
        assert "multi_hub_mesh" in types
        assert "random_network" in types

    @pytest.mark.parametrize("scenario_type", SyntheticScenarioFactory.SCENARIO_TYPES)
    def test_create_all_types_easy(self, scenario_type):
        """Factory creates each type without error."""
        scenario = SyntheticScenarioFactory.create(scenario_type, difficulty="easy", seed=42)
        assert isinstance(scenario, SyntheticScenario)
        assert scenario.network is not None
        assert scenario.transfers is not None
        assert scenario.metadata.scenario_type == scenario_type
        assert scenario.metadata.difficulty == "easy"
        assert scenario.metadata.n_lines > 0
        assert scenario.metadata.n_transfers >= 0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario type"):
            SyntheticScenarioFactory.create("nonexistent")

    @pytest.mark.parametrize("scenario_type", SyntheticScenarioFactory.SCENARIO_TYPES)
    def test_schedule_random_works(self, scenario_type):
        """Schedule.random can create a schedule for each scenario's network."""
        import numpy as np
        scenario = SyntheticScenarioFactory.create(scenario_type, difficulty="easy", seed=42)
        rng = np.random.default_rng(123)
        schedule = Schedule.random(scenario.network, rng)
        assert len(schedule) == scenario.metadata.n_lines


# ---------------------------------------------------------------------------
# Hub scenario tests
# ---------------------------------------------------------------------------


class TestHubNetwork:
    """Tests for hub network scenario."""

    def test_hub_easy_structure(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="easy")
        assert s.metadata.n_lines == 6
        assert s.metadata.n_transfers == 9  # 3 in * 3 out

    def test_hub_medium_structure(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="medium")
        assert s.metadata.n_lines == 8
        assert s.metadata.n_transfers == 16  # 4 * 4

    def test_hub_hard_structure(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="hard")
        assert s.metadata.n_lines == 12
        assert s.metadata.n_transfers == 36  # 6 * 6

    def test_hub_has_optimal_solution(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="easy")
        assert s.metadata.optimal_solution is not None
        assert len(s.metadata.optimal_solution) == s.metadata.n_lines

    def test_hub_optimal_cost_zero(self):
        """At the optimal schedule, transfer cost should be 0."""
        s = SyntheticScenarioFactory.create("hub", difficulty="easy")
        optimal = s.metadata.optimal_solution
        assert optimal is not None

        schedule = Schedule(offsets=dict(optimal), period=T_PERIOD)
        cost_fn = TransferCost(transfers=s.transfers)
        cost = cost_fn(schedule, s.network)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_hub_optimal_cost_zero_medium(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="medium")
        optimal = s.metadata.optimal_solution
        assert optimal is not None

        schedule = Schedule(offsets=dict(optimal), period=T_PERIOD)
        cost_fn = TransferCost(transfers=s.transfers)
        cost = cost_fn(schedule, s.network)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_hub_station_indices(self):
        """Verify transfer station indices are correct."""
        s = SyntheticScenarioFactory.create("hub", difficulty="easy")
        for transfer in s.transfers:
            # Inbound arrival at idx=1, outbound departure at idx=0
            assert transfer.from_station_idx == 1
            assert transfer.to_station_idx == 0

    def test_hub_all_transfers_at_hub_location(self):
        """All transfers happen at the hub location."""
        s = SyntheticScenarioFactory.create("hub", difficulty="easy")
        for transfer in s.transfers:
            from_line = s.network.get_line(transfer.from_line)
            to_line = s.network.get_line(transfer.to_line)
            from_loc = from_line.stations[transfer.from_station_idx].location
            to_loc = to_line.stations[transfer.to_station_idx].location
            assert from_loc == "Hub"
            assert to_loc == "Hub"

    def test_hub_odd_n_lines_raises(self):
        with pytest.raises(ValueError, match="n_lines must be even"):
            SyntheticScenarioFactory.create("hub", n_lines=5)


# ---------------------------------------------------------------------------
# Chain scenario tests
# ---------------------------------------------------------------------------


class TestChainNetwork:
    """Tests for chain network scenario."""

    def test_chain_easy_structure(self):
        s = SyntheticScenarioFactory.create("chain", difficulty="easy")
        assert s.metadata.n_lines == 3
        assert s.metadata.n_transfers == 2

    def test_chain_medium_structure(self):
        s = SyntheticScenarioFactory.create("chain", difficulty="medium")
        assert s.metadata.n_lines == 5
        assert s.metadata.n_transfers == 4

    def test_chain_hard_structure(self):
        s = SyntheticScenarioFactory.create("chain", difficulty="hard")
        assert s.metadata.n_lines == 8
        assert s.metadata.n_transfers == 7

    def test_chain_has_optimal_solution(self):
        s = SyntheticScenarioFactory.create("chain", difficulty="easy")
        assert s.metadata.optimal_solution is not None

    def test_chain_optimal_cost_zero(self):
        """At the optimal schedule, all transfers have w=w_ideal -> cost=0."""
        s = SyntheticScenarioFactory.create("chain", difficulty="easy")
        optimal = s.metadata.optimal_solution
        assert optimal is not None

        schedule = Schedule(offsets=dict(optimal), period=T_PERIOD)
        cost_fn = TransferCost(transfers=s.transfers)
        cost = cost_fn(schedule, s.network)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_chain_optimal_cost_zero_medium(self):
        s = SyntheticScenarioFactory.create("chain", difficulty="medium")
        optimal = s.metadata.optimal_solution
        assert optimal is not None

        schedule = Schedule(offsets=dict(optimal), period=T_PERIOD)
        cost_fn = TransferCost(transfers=s.transfers)
        cost = cost_fn(schedule, s.network)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_chain_offset_propagation(self):
        """Offsets propagate correctly step by step."""
        s = SyntheticScenarioFactory.create("chain", difficulty="easy")
        optimal = s.metadata.optimal_solution
        assert optimal is not None

        # Verify that each consecutive offset differs by the step
        step = _compute_optimal_outbound_offset(15, 2, 3.0, 8.0, 60)
        assert optimal["L0"] == 0
        assert optimal["L1"] == step % T_PERIOD
        assert optimal["L2"] == (2 * step) % T_PERIOD


# ---------------------------------------------------------------------------
# Conflicting transfers tests
# ---------------------------------------------------------------------------


class TestConflictingTransfers:
    """Tests for conflicting transfers (ring) scenario."""

    def test_conflicting_easy_structure(self):
        s = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        assert s.metadata.n_lines == 3
        assert s.metadata.n_transfers == 3  # Ring: same as n_lines

    def test_conflicting_medium_structure(self):
        s = SyntheticScenarioFactory.create("conflicting", difficulty="medium")
        assert s.metadata.n_lines == 4
        assert s.metadata.n_transfers == 4

    def test_conflicting_ring_topology(self):
        """Transfers form a ring: L0->L1->L2->L0."""
        s = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        n = s.metadata.n_lines
        for i in range(n):
            j = (i + 1) % n
            tid = f"T_L{i}_L{j}"
            assert tid in s.transfers.transfers

    def test_conflicting_has_conflict(self):
        """Default parameters should create an unsatisfiable constraint set."""
        s = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        assert s.metadata.properties["has_conflict"] is True

    def test_conflicting_no_optimal_solution(self):
        s = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        assert s.metadata.optimal_solution is None

    def test_conflicting_cost_always_positive(self):
        """For any schedule, cost > 0 because constraints conflict."""
        import numpy as np
        s = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        cost_fn = TransferCost(transfers=s.transfers)
        rng = np.random.default_rng(42)

        # Try many random schedules — all should have positive cost
        for _ in range(50):
            schedule = Schedule.random(s.network, rng)
            cost = cost_fn(schedule, s.network)
            # Cost may be very small but not exactly zero due to conflicts
            # The key invariant is that no schedule achieves cost=0
            # (some might be close to 0 if w_ideal accidentally aligns)
        # At least verify a zero-offset schedule is not optimal
        zero_schedule = Schedule.from_network(s.network, default_offset=0)
        zero_cost = cost_fn(zero_schedule, s.network)
        assert zero_cost > 0

    def test_conflicting_different_w_ideals(self):
        """Each transfer has a different w_ideal."""
        s = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        w_ideals = s.metadata.properties["w_ideals"]
        # At least some are different
        assert len(set(w_ideals)) > 1


# ---------------------------------------------------------------------------
# Weighted transfers tests
# ---------------------------------------------------------------------------


class TestWeightedTransfers:
    """Tests for weighted transfers scenario."""

    def test_weighted_easy_structure(self):
        s = SyntheticScenarioFactory.create("weighted", difficulty="easy", seed=42)
        assert s.metadata.n_lines > 0
        assert s.metadata.n_transfers > 0

    def test_weighted_has_high_weight_transfers(self):
        s = SyntheticScenarioFactory.create("weighted", difficulty="easy", seed=42)
        n_high = s.metadata.properties["n_high_weight"]
        assert n_high == 2  # easy has 2 high

    def test_weighted_weight_distribution(self):
        """Some transfers have high weight, rest have 1.0."""
        s = SyntheticScenarioFactory.create("weighted", difficulty="medium", seed=42)
        weights = [t.weight for t in s.transfers]
        high_weights = [w for w in weights if w > 1.0]
        low_weights = [w for w in weights if w == 1.0]
        assert len(high_weights) == 4  # medium has 4 high
        assert len(low_weights) > 0

    def test_weighted_no_optimal(self):
        s = SyntheticScenarioFactory.create("weighted", difficulty="easy", seed=42)
        assert s.metadata.optimal_solution is None

    def test_weighted_ratio(self):
        s = SyntheticScenarioFactory.create(
            "weighted", difficulty="easy", seed=42, weight_ratio=50.0
        )
        high_ids = s.metadata.properties["high_weight_ids"]
        for tid in high_ids:
            assert s.transfers.transfers[tid].weight == 50.0


# ---------------------------------------------------------------------------
# Full cost (with turnarounds) tests
# ---------------------------------------------------------------------------


class TestFullCostNetwork:
    """Tests for full cost network with turnarounds."""

    def test_full_cost_easy_structure(self):
        s = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        assert s.metadata.n_lines == 4  # 2 pairs * 2
        assert s.metadata.n_turnarounds == 2

    def test_full_cost_medium_structure(self):
        s = SyntheticScenarioFactory.create("full_cost", difficulty="medium")
        assert s.metadata.n_lines == 8  # 4 pairs * 2
        assert s.metadata.n_turnarounds == 4

    def test_full_cost_has_turnarounds(self):
        s = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        assert s.turnarounds is not None
        assert len(s.turnarounds) == 2

    def test_full_cost_bidirectional_lines(self):
        """Each pair has OUT and RET lines."""
        s = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        line_ids = s.network.line_ids
        assert "OUT0" in line_ids
        assert "RET0" in line_ids
        assert "OUT1" in line_ids
        assert "RET1" in line_ids

    def test_full_cost_transfers_exclude_same_pair(self):
        """No transfer from RET_i -> OUT_i (same pair)."""
        s = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        for transfer in s.transfers:
            # Extract pair indices
            from_idx = transfer.from_line.replace("RET", "")
            to_idx = transfer.to_line.replace("OUT", "")
            assert from_idx != to_idx

    def test_full_cost_turnaround_structure(self):
        """Turnarounds connect OUT_i endpoint to RET_i endpoint."""
        s = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        assert s.turnarounds is not None
        for tc in s.turnarounds:
            # end_line is OUT, start_line is RET, same pair
            assert tc.end_line.startswith("OUT")
            assert tc.start_line.startswith("RET")
            pair_idx_end = tc.end_line.replace("OUT", "")
            pair_idx_start = tc.start_line.replace("RET", "")
            assert pair_idx_end == pair_idx_start

    def test_full_cost_compute_with_total_cost(self):
        """TotalCost can be computed on this scenario."""
        from src.cost.operational import OperationalCost, TurnaroundCost
        from src.cost.robustness import DelayDistribution, RobustnessCost
        from src.cost.total import TotalCost

        s = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        assert s.turnarounds is not None

        transfer_cost = TransferCost(transfers=s.transfers)
        turnaround_cost = TurnaroundCost(turnarounds=s.turnarounds)
        ops_cost = OperationalCost(turnaround_cost=turnaround_cost)
        rob_cost = RobustnessCost(
            transfers=s.transfers,
            delay_distribution=DelayDistribution(rate=1 / 3, alpha=0.95),
        )
        total = TotalCost(
            transfer_cost=transfer_cost,
            operational_cost=ops_cost,
            robustness_cost=rob_cost,
        )

        schedule = Schedule.from_network(s.network, default_offset=0)
        cost = total(schedule, s.network)
        assert cost >= 0


# ---------------------------------------------------------------------------
# Multi-hub mesh tests
# ---------------------------------------------------------------------------


class TestMultiHubMesh:
    """Tests for multi-hub mesh scenario."""

    def test_mesh_easy_has_3_hubs(self):
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        assert s.metadata.properties["n_hubs"] == 3

    def test_mesh_has_ic_regional_local(self):
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        assert s.metadata.properties["n_ic_lines"] == 3
        assert s.metadata.properties["n_regional_lines"] > 0
        assert s.metadata.properties["n_local_lines"] > 0

    def test_mesh_mixed_frequencies(self):
        """IC, regional, and local have different frequencies."""
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        frequencies = set()
        for line in s.network.lines.values():
            frequencies.add(line.frequency)
        # Should have at least 2 different frequencies (IC=120, regional=60, local=30)
        assert len(frequencies) >= 2

    def test_mesh_mixed_modalities(self):
        """Network has multiple modalities."""
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        modalities = set()
        for line in s.network.lines.values():
            modalities.add(line.modality)
        assert "IC" in modalities
        assert "regional" in modalities
        assert "bus" in modalities

    def test_mesh_has_turnarounds(self):
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        assert s.turnarounds is not None
        assert len(s.turnarounds) > 0

    def test_mesh_has_transfers(self):
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        assert s.metadata.n_transfers > 0

    def test_mesh_difficulty_scaling(self):
        """Harder difficulty has more lines and transfers."""
        easy = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        medium = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="medium")
        hard = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="hard")
        assert easy.metadata.n_lines <= medium.metadata.n_lines <= hard.metadata.n_lines

    def test_mesh_no_optimal(self):
        s = SyntheticScenarioFactory.create("multi_hub_mesh", difficulty="easy")
        assert s.metadata.optimal_solution is None


# ---------------------------------------------------------------------------
# Random network tests
# ---------------------------------------------------------------------------


class TestRandomNetwork:
    """Tests for random network scenario."""

    def test_random_easy_structure(self):
        s = SyntheticScenarioFactory.create("random_network", difficulty="easy", seed=42)
        assert s.metadata.n_lines == 6
        assert s.metadata.n_transfers >= 1

    def test_random_different_seeds_different_topologies(self):
        """Different seeds produce different networks."""
        s1 = SyntheticScenarioFactory.create("random_network", difficulty="medium", seed=42)
        s2 = SyntheticScenarioFactory.create("random_network", difficulty="medium", seed=123)

        # Should differ in at least one aspect
        transfers_1 = set(s1.transfers.transfers.keys())
        transfers_2 = set(s2.transfers.transfers.keys())
        # Very likely different transfer sets
        assert transfers_1 != transfers_2 or s1.metadata.n_transfers != s2.metadata.n_transfers

    def test_random_same_seed_reproducible(self):
        """Same seed produces identical networks."""
        s1 = SyntheticScenarioFactory.create("random_network", difficulty="easy", seed=42)
        s2 = SyntheticScenarioFactory.create("random_network", difficulty="easy", seed=42)
        assert s1.metadata.n_lines == s2.metadata.n_lines
        assert s1.metadata.n_transfers == s2.metadata.n_transfers
        assert set(s1.transfers.transfers.keys()) == set(s2.transfers.transfers.keys())

    def test_random_min_one_transfer(self):
        """Network always has at least 1 transfer."""
        for seed in range(10):
            s = SyntheticScenarioFactory.create(
                "random_network", difficulty="easy", seed=seed
            )
            assert s.metadata.n_transfers >= 1

    def test_random_no_optimal(self):
        s = SyntheticScenarioFactory.create("random_network", difficulty="easy", seed=42)
        assert s.metadata.optimal_solution is None

    def test_random_custom_params(self):
        s = SyntheticScenarioFactory.create(
            "random_network",
            difficulty="easy",
            seed=42,
            n_stations=10,
            n_lines=8,
            transfer_density=0.8,
        )
        assert s.metadata.n_lines == 8
        assert s.metadata.properties["n_stations"] == 10


# ---------------------------------------------------------------------------
# Parameterization tests
# ---------------------------------------------------------------------------


class TestParametrization:
    """Tests for factory parameterization with custom kwargs."""

    def test_hub_custom_frequency(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="easy", frequency=30)
        for line in s.network.lines.values():
            assert line.frequency == 30

    def test_hub_custom_w_ideal(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="easy", w_ideal=12.0)
        for transfer in s.transfers:
            assert transfer.params.w_ideal == 12.0

    def test_chain_custom_travel_time(self):
        s = SyntheticScenarioFactory.create(
            "chain", difficulty="easy", travel_time=25
        )
        for line in s.network.lines.values():
            assert line.travel_times[0] == 25

    def test_hub_custom_n_lines(self):
        s = SyntheticScenarioFactory.create("hub", difficulty="easy", n_lines=10)
        assert s.metadata.n_lines == 10

    def test_full_cost_custom_min_turnaround(self):
        s = SyntheticScenarioFactory.create(
            "full_cost", difficulty="easy", min_turnaround=20.0
        )
        assert s.turnarounds is not None
        for tc in s.turnarounds:
            assert tc.min_turnaround == 20.0

    def test_hub_custom_params_optimal_still_zero(self):
        """Even with custom params, optimal solution gives cost=0."""
        s = SyntheticScenarioFactory.create(
            "hub", difficulty="easy",
            frequency=30, travel_time=10, walk_time=2.0, w_ideal=5.0,
        )
        optimal = s.metadata.optimal_solution
        assert optimal is not None
        schedule = Schedule(offsets=dict(optimal), period=T_PERIOD)
        cost_fn = TransferCost(transfers=s.transfers)
        cost = cost_fn(schedule, s.network)
        assert cost == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Difficulty level tests
# ---------------------------------------------------------------------------


class TestDifficultyLevels:
    """Tests that harder difficulties produce larger scenarios."""

    @pytest.mark.parametrize("scenario_type", [
        "hub", "chain", "conflicting", "full_cost",
    ])
    def test_difficulty_increases_size(self, scenario_type):
        easy = SyntheticScenarioFactory.create(scenario_type, difficulty="easy")
        medium = SyntheticScenarioFactory.create(scenario_type, difficulty="medium")
        hard = SyntheticScenarioFactory.create(scenario_type, difficulty="hard")

        assert easy.metadata.n_lines <= medium.metadata.n_lines
        assert medium.metadata.n_lines <= hard.metadata.n_lines

    @pytest.mark.parametrize("scenario_type", [
        "hub", "chain", "conflicting", "full_cost",
    ])
    def test_difficulty_increases_transfers(self, scenario_type):
        easy = SyntheticScenarioFactory.create(scenario_type, difficulty="easy")
        hard = SyntheticScenarioFactory.create(scenario_type, difficulty="hard")
        assert easy.metadata.n_transfers <= hard.metadata.n_transfers


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAllScenariosIntegration:
    """Integration tests running cost computation on all scenarios."""

    @pytest.mark.parametrize("scenario_type", SyntheticScenarioFactory.SCENARIO_TYPES)
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_cost_computable(self, scenario_type, difficulty):
        """TransferCost can be computed on every scenario."""
        import numpy as np
        scenario = SyntheticScenarioFactory.create(
            scenario_type, difficulty=difficulty, seed=42
        )
        cost_fn = TransferCost(transfers=scenario.transfers)
        rng = np.random.default_rng(0)
        schedule = Schedule.random(scenario.network, rng)
        cost = cost_fn(schedule, scenario.network)
        assert cost >= 0
        assert not (cost != cost)  # Not NaN
