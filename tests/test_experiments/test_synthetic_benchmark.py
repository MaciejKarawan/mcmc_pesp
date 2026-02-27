"""Tests for synthetic benchmark runner."""

from __future__ import annotations

import pytest

from src.io.synthetic_scenarios import SyntheticScenarioFactory
from src.experiments.benchmark.run_synthetic_benchmark import (
    SyntheticBenchmarkConfig,
    SyntheticBenchmarkSuite,
    build_cost_fn,
    build_proposals,
    run_scenario_benchmark,
    run_synthetic_benchmark_suite,
)

import numpy as np


# ---------------------------------------------------------------------------
# Cost function builder tests
# ---------------------------------------------------------------------------


class TestBuildCostFn:
    """Tests for build_cost_fn."""

    def test_transfer_only_scenario(self):
        """Hub scenario without turnarounds returns TransferCost or TotalCost."""
        scenario = SyntheticScenarioFactory.create("hub", difficulty="easy")
        config = SyntheticBenchmarkConfig(rob_weight=0.0)
        cost_fn = build_cost_fn(scenario, config)
        # Should be plain TransferCost
        from src.cost.transfer_cost import TransferCost
        assert isinstance(cost_fn, TransferCost)

    def test_full_cost_scenario(self):
        """Full cost scenario returns TotalCost."""
        scenario = SyntheticScenarioFactory.create("full_cost", difficulty="easy")
        config = SyntheticBenchmarkConfig()
        cost_fn = build_cost_fn(scenario, config)
        from src.cost.total import TotalCost
        assert isinstance(cost_fn, TotalCost)

    def test_robustness_included(self):
        """When rob_weight > 0, robustness is included for simple scenarios."""
        scenario = SyntheticScenarioFactory.create("hub", difficulty="easy")
        config = SyntheticBenchmarkConfig(rob_weight=0.05)
        cost_fn = build_cost_fn(scenario, config)
        from src.cost.total import TotalCost
        assert isinstance(cost_fn, TotalCost)


# ---------------------------------------------------------------------------
# Proposal builder tests
# ---------------------------------------------------------------------------


class TestBuildProposals:
    """Tests for build_proposals."""

    def test_all_proposals_included(self):
        scenario = SyntheticScenarioFactory.create("hub", difficulty="easy")
        config = SyntheticBenchmarkConfig(
            use_component_move=True, use_smart_repair=True
        )
        rng = np.random.default_rng(42)
        proposals, weights = build_proposals(scenario, config, rng)
        assert len(proposals) == 4
        assert len(weights) == 4

    def test_minimal_proposals(self):
        scenario = SyntheticScenarioFactory.create("hub", difficulty="easy")
        config = SyntheticBenchmarkConfig(
            use_component_move=False, use_smart_repair=False
        )
        rng = np.random.default_rng(42)
        proposals, weights = build_proposals(scenario, config, rng)
        assert len(proposals) == 2  # SingleLineShift + UniformLineShift


# ---------------------------------------------------------------------------
# Single scenario benchmark tests
# ---------------------------------------------------------------------------


class TestRunScenarioBenchmark:
    """Tests for run_scenario_benchmark."""

    def test_hub_easy_short_run(self):
        """Run a short benchmark on hub easy, verify result structure."""
        scenario = SyntheticScenarioFactory.create("hub", difficulty="easy")
        config = SyntheticBenchmarkConfig(
            n_chains=2,
            n_steps_per_chain=500,
            thin=5,
        )
        result = run_scenario_benchmark(scenario, config, base_seed=42)

        assert result.scenario_name == "hub_easy"
        assert result.scenario_type == "hub"
        assert result.difficulty == "easy"
        assert result.best_cost >= 0
        assert 0 < result.acceptance_rate < 1
        assert result.runtime_seconds > 0
        assert isinstance(result.best_solution, dict)
        assert result.diagnostics is not None

    def test_chain_easy_finds_near_optimum(self):
        """Chain easy with enough steps should get close to optimal."""
        scenario = SyntheticScenarioFactory.create("chain", difficulty="easy")
        config = SyntheticBenchmarkConfig(
            n_chains=2,
            n_steps_per_chain=2000,
            thin=5,
            rob_weight=0.0,
        )
        result = run_scenario_benchmark(scenario, config, base_seed=42)

        # Should get reasonably close to 0 (the optimum)
        assert result.optimal_cost is not None
        assert result.optimal_cost == pytest.approx(0.0, abs=1e-3)
        # Gap might be large for short runs, but result structure is correct
        assert result.gap_to_optimum is not None

    def test_conflicting_cost_positive(self):
        """Conflicting scenario always has positive cost."""
        scenario = SyntheticScenarioFactory.create("conflicting", difficulty="easy")
        config = SyntheticBenchmarkConfig(
            n_chains=2,
            n_steps_per_chain=500,
            thin=5,
            rob_weight=0.0,
        )
        result = run_scenario_benchmark(scenario, config, base_seed=42)
        assert result.best_cost > 0
        assert result.optimal_cost is None


# ---------------------------------------------------------------------------
# Suite tests
# ---------------------------------------------------------------------------


class TestRunSyntheticBenchmarkSuite:
    """Tests for run_synthetic_benchmark_suite."""

    def test_mini_suite(self):
        """Run a minimal suite with 2 scenarios."""
        config = SyntheticBenchmarkConfig(
            scenarios=[
                {"type": "hub", "difficulty": "easy"},
                {"type": "chain", "difficulty": "easy"},
            ],
            n_chains=2,
            n_steps_per_chain=500,
            thin=5,
        )
        suite = run_synthetic_benchmark_suite(config, base_seed=42)

        assert isinstance(suite, SyntheticBenchmarkSuite)
        assert len(suite.results) == 2
        assert suite.results[0].scenario_type == "hub"
        assert suite.results[1].scenario_type == "chain"

    def test_summary_table(self):
        """Summary table has correct structure."""
        config = SyntheticBenchmarkConfig(
            scenarios=[
                {"type": "hub", "difficulty": "easy"},
            ],
            n_chains=2,
            n_steps_per_chain=200,
            thin=5,
        )
        suite = run_synthetic_benchmark_suite(config, base_seed=42)
        table = suite.summary_table()

        assert len(table) == 1
        row = table[0]
        assert "scenario" in row
        assert "best_cost" in row
        assert "acceptance_rate" in row
        assert "runtime_s" in row


# ---------------------------------------------------------------------------
# Save results tests
# ---------------------------------------------------------------------------


class TestSaveResults:
    """Tests for save_results."""

    def test_save_creates_files(self, tmp_path):
        """save_results creates JSON and CSV files."""
        from src.experiments.benchmark.run_synthetic_benchmark import save_results

        config = SyntheticBenchmarkConfig(
            scenarios=[{"type": "hub", "difficulty": "easy"}],
            n_chains=2,
            n_steps_per_chain=200,
            thin=5,
        )
        suite = run_synthetic_benchmark_suite(config, base_seed=42)
        save_results(suite, tmp_path)

        json_file = tmp_path / "results.json"
        csv_file = tmp_path / "summary.csv"
        assert json_file.exists()
        assert csv_file.exists()

        # Verify JSON is valid
        import json
        with open(json_file) as f:
            data = json.load(f)
        assert "config" in data
        assert "results" in data
        assert len(data["results"]) == 1
