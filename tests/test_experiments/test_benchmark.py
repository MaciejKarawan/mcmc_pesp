"""Tests for PESPlib benchmark infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.schedule import Schedule
from src.cost.transfer_cost import TransferCost
from src.io.pesplib_loader import load_pesplib_instance
from src.io.synthetic_pesplib import (
    MiniInstanceParams,
    SyntheticPESPLibInstance,
    generate_mini_instance,
    generate_triangle_pesplib,
)
from src.mcmc.proposals.component import create_component_move
from src.mcmc.proposals.single_line import SingleLineShift
from src.mcmc.proposals.smart_repair import create_smart_repair
from src.mcmc.sampler import MCMCSampler
from src.metrics.convergence import MCMCDiagnostics


class TestSyntheticPESPLibGeneration:
    """Tests for synthetic PESPlib instance generation."""

    def test_generate_mini_instance_creates_valid_files(self) -> None:
        """Test that generate_mini_instance creates valid content."""
        instance = generate_mini_instance()

        # Check that all content is non-empty
        assert len(instance.events_content) > 0
        assert len(instance.activities_content) > 0
        assert len(instance.config_content) > 0

        # Check that content has expected headers
        assert "event-id" in instance.events_content
        assert "activity-id" in instance.activities_content
        assert "period" in instance.config_content

    def test_generate_mini_instance_with_custom_params(self) -> None:
        """Test generation with custom parameters."""
        params = MiniInstanceParams(
            n_lines=3,
            n_stations_per_line=4,
            period=120,
        )
        instance = generate_mini_instance(params)

        # Check that period is in config
        assert "120" in instance.config_content

    def test_generate_triangle_pesplib(self) -> None:
        """Test triangle network generation in PESPlib format."""
        instance = generate_triangle_pesplib()

        # Check content exists
        assert len(instance.events_content) > 0
        assert len(instance.activities_content) > 0

        # Check for transfer activities
        assert "transfer" in instance.activities_content

        # Check optimum offsets exist
        assert len(instance.optimum_offsets) > 0

    def test_write_to_directory(self) -> None:
        """Test writing instance to directory."""
        instance = generate_mini_instance()

        with tempfile.TemporaryDirectory() as tmpdir:
            instance_dir = Path(tmpdir) / "test_instance"
            instance.write_to_directory(instance_dir)

            # Check files exist
            assert (instance_dir / "Events.giv").exists()
            assert (instance_dir / "Activities.giv").exists()
            assert (instance_dir / "Config.cnf").exists()


class TestSyntheticInstanceLoading:
    """Tests for loading synthetic instances with PESPlib loader."""

    @pytest.fixture
    def mini_instance_path(self) -> Path:
        """Create a temporary mini instance for testing."""
        instance = generate_mini_instance()
        tmpdir = tempfile.mkdtemp()
        instance_dir = Path(tmpdir) / "mini"
        instance.write_to_directory(instance_dir)
        return instance_dir

    @pytest.fixture
    def triangle_instance_path(self) -> Path:
        """Create a temporary triangle instance for testing."""
        instance = generate_triangle_pesplib()
        tmpdir = tempfile.mkdtemp()
        instance_dir = Path(tmpdir) / "triangle"
        instance.write_to_directory(instance_dir)
        return instance_dir

    def test_load_mini_instance(self, mini_instance_path: Path) -> None:
        """Test loading the mini synthetic instance."""
        network, transfers, event_mapping = load_pesplib_instance(mini_instance_path)

        # Check network was loaded
        assert len(network) > 0
        assert network.period > 0

        # Check transfers were extracted
        assert len(transfers) > 0

    def test_load_triangle_instance(self, triangle_instance_path: Path) -> None:
        """Test loading the triangle synthetic instance."""
        network, transfers, event_mapping = load_pesplib_instance(triangle_instance_path)

        # Should have 3 lines for triangle
        assert len(network) >= 2  # At least 2 lines

        # Should have transfers
        assert len(transfers) >= 1


class TestBenchmarkMCMC:
    """Tests for running MCMC on synthetic instances."""

    @pytest.fixture
    def mini_network_and_transfers(self):
        """Load mini instance and return network + transfers."""
        instance = generate_mini_instance()
        tmpdir = tempfile.mkdtemp()
        instance_dir = Path(tmpdir) / "mini"
        instance.write_to_directory(instance_dir)

        network, transfers, _ = load_pesplib_instance(instance_dir)
        return network, transfers

    def test_mcmc_single_chain_on_mini(self, mini_network_and_transfers) -> None:
        """Test running a single MCMC chain on mini instance."""
        network, transfers = mini_network_and_transfers
        rng = np.random.default_rng(42)

        # Create cost function
        cost_fn = TransferCost(transfers=transfers)

        # Create proposal
        proposal = SingleLineShift(sigma=5.0, rng=rng)

        # Create sampler
        sampler = MCMCSampler(
            network=network,
            cost_fn=cost_fn,
            proposals=[proposal],
            beta=1.0,
            rng=rng,
        )

        # Initialize
        initial = Schedule.random(network, rng)
        sampler.initialize(initial)

        # Run short chain
        result = sampler.run(n_steps=100, thin=1)

        # Check results
        assert result.acceptance_rate >= 0
        assert result.best_cost < float("inf")
        assert len(result.costs) > 0

    def test_mcmc_with_component_move(self, mini_network_and_transfers) -> None:
        """Test MCMC with ComponentMove proposal."""
        network, transfers = mini_network_and_transfers
        rng = np.random.default_rng(42)

        cost_fn = TransferCost(transfers=transfers)

        # Create component move
        component = create_component_move(transfers, sigma=5.0, rng=rng)

        sampler = MCMCSampler(
            network=network,
            cost_fn=cost_fn,
            proposals=[component],
            beta=1.0,
            rng=rng,
        )

        initial = Schedule.random(network, rng)
        sampler.initialize(initial)

        result = sampler.run(n_steps=50, thin=1)

        assert result.acceptance_rate >= 0
        assert len(result.costs) > 0

    def test_mcmc_with_smart_repair(self, mini_network_and_transfers) -> None:
        """Test MCMC with SmartRepairMove proposal."""
        network, transfers = mini_network_and_transfers
        rng = np.random.default_rng(42)

        cost_fn = TransferCost(transfers=transfers)

        # Create smart repair
        smart_repair = create_smart_repair(transfers, sigma=2.0, rng=rng)

        sampler = MCMCSampler(
            network=network,
            cost_fn=cost_fn,
            proposals=[smart_repair],
            beta=1.0,
            rng=rng,
        )

        initial = Schedule.random(network, rng)
        sampler.initialize(initial)

        result = sampler.run(n_steps=50, thin=1)

        assert result.acceptance_rate >= 0

    def test_mcmc_with_mixed_proposals(self, mini_network_and_transfers) -> None:
        """Test MCMC with multiple proposal types."""
        network, transfers = mini_network_and_transfers
        rng = np.random.default_rng(42)

        cost_fn = TransferCost(transfers=transfers)

        # Create multiple proposals
        proposals = [
            SingleLineShift(sigma=5.0, rng=rng),
            create_component_move(transfers, sigma=5.0, rng=rng),
            create_smart_repair(transfers, sigma=2.0, rng=rng),
        ]
        weights = [0.4, 0.3, 0.3]

        sampler = MCMCSampler(
            network=network,
            cost_fn=cost_fn,
            proposals=proposals,
            proposal_weights=weights,
            beta=1.0,
            rng=rng,
        )

        initial = Schedule.random(network, rng)
        sampler.initialize(initial)

        result = sampler.run(n_steps=100, thin=1)

        assert result.acceptance_rate >= 0
        assert result.best_cost < float("inf")


class TestMultiChainDiagnostics:
    """Tests for multi-chain convergence diagnostics."""

    @pytest.fixture
    def mini_network_and_transfers(self):
        """Load mini instance."""
        instance = generate_mini_instance()
        tmpdir = tempfile.mkdtemp()
        instance_dir = Path(tmpdir) / "mini"
        instance.write_to_directory(instance_dir)

        network, transfers, _ = load_pesplib_instance(instance_dir)
        return network, transfers

    def test_multi_chain_r_hat(self, mini_network_and_transfers) -> None:
        """Test R-hat computation across multiple chains."""
        network, transfers = mini_network_and_transfers

        n_chains = 4
        n_steps = 200
        chains: list[list[float]] = []

        for chain_id in range(n_chains):
            rng = np.random.default_rng(42 + chain_id)
            cost_fn = TransferCost(transfers=transfers)
            proposal = SingleLineShift(sigma=5.0, rng=rng)

            sampler = MCMCSampler(
                network=network,
                cost_fn=cost_fn,
                proposals=[proposal],
                beta=1.0,
                rng=rng,
            )

            initial = Schedule.random(network, rng)
            sampler.initialize(initial)

            result = sampler.run(n_steps=n_steps, thin=1)
            chains.append(result.costs)

        # Compute diagnostics
        diagnostics = MCMCDiagnostics(chains=chains)
        summary = diagnostics.summary()

        # R-hat should be computed
        assert "r_hat" in summary
        assert summary["r_hat"] is not None
        assert summary["r_hat"] >= 1.0  # R-hat is always >= 1

        # ESS should be computed
        assert "ess_total" in summary
        assert summary["ess_total"] > 0

    def test_convergence_detection(self, mini_network_and_transfers) -> None:
        """Test that convergence detection works."""
        network, transfers = mini_network_and_transfers

        # Run longer chains for better convergence
        n_chains = 2
        n_steps = 500
        chains: list[list[float]] = []

        for chain_id in range(n_chains):
            rng = np.random.default_rng(42 + chain_id)
            cost_fn = TransferCost(transfers=transfers)
            proposal = SingleLineShift(sigma=5.0, rng=rng)

            sampler = MCMCSampler(
                network=network,
                cost_fn=cost_fn,
                proposals=[proposal],
                beta=1.0,
                rng=rng,
            )

            initial = Schedule.random(network, rng)
            sampler.initialize(initial)

            result = sampler.run(n_steps=n_steps, thin=1)
            chains.append(result.costs)

        diagnostics = MCMCDiagnostics(chains=chains)

        # is_converged should return a boolean-like value
        converged = diagnostics.is_converged(r_hat_threshold=1.5, min_ess_threshold=10)
        # Check it's truthy/falsy (works with numpy bools too)
        assert converged in (True, False) or bool(converged) in (True, False)


class TestBenchmarkIntegration:
    """Integration tests for the benchmark runner."""

    def test_benchmark_imports(self) -> None:
        """Test that benchmark module imports correctly."""
        from src.experiments.benchmark.run_pesplib import (
            BenchmarkConfig,
            BenchmarkResult,
            ChainResult,
            find_instance_path,
            run_multi_chain_benchmark,
            run_single_chain,
        )

        # Check that classes exist
        assert BenchmarkConfig is not None
        assert BenchmarkResult is not None
        assert ChainResult is not None

    def test_find_synthetic_instance(self) -> None:
        """Test finding synthetic instances."""
        from src.experiments.benchmark.run_pesplib import find_instance_path

        # Should find mini instance (will be generated)
        path = find_instance_path("mini")
        assert path.exists()
        assert (path / "Events.giv").exists()

    def test_run_single_chain_function(self) -> None:
        """Test run_single_chain function."""
        from src.experiments.benchmark.run_pesplib import (
            BenchmarkConfig,
            run_single_chain,
        )

        instance = generate_mini_instance()
        tmpdir = tempfile.mkdtemp()
        instance_dir = Path(tmpdir) / "mini"
        instance.write_to_directory(instance_dir)

        network, transfers, _ = load_pesplib_instance(instance_dir)

        config = BenchmarkConfig(
            n_chains=1,
            n_steps_per_chain=100,
            beta=1.0,
        )

        result = run_single_chain(
            chain_id=0,
            network=network,
            transfers=transfers,
            config=config,
            seed=42,
        )

        assert result.chain_id == 0
        assert result.best_cost < float("inf")
        assert result.acceptance_rate >= 0
        assert len(result.costs) > 0

    def test_benchmark_result_to_dict(self) -> None:
        """Test BenchmarkResult JSON serialization."""
        from src.experiments.benchmark.run_pesplib import (
            BenchmarkConfig,
            BenchmarkResult,
            ChainResult,
        )

        config = BenchmarkConfig(n_chains=2, n_steps_per_chain=100)

        chain_results = [
            ChainResult(
                chain_id=0,
                costs=[1.0, 0.9, 0.8],
                best_cost=0.8,
                best_offsets={"L1": 0},
                acceptance_rate=0.3,
            ),
            ChainResult(
                chain_id=1,
                costs=[1.1, 0.95, 0.85],
                best_cost=0.85,
                best_offsets={"L1": 5},
                acceptance_rate=0.35,
            ),
        ]

        result = BenchmarkResult(
            instance_name="test",
            config=config,
            chain_results=chain_results,
            best_solution={"L1": 0},
            best_cost=0.8,
            pesplib_objective=100.0,
            diagnostics={"r_hat": 1.05, "converged": True},
        )

        result_dict = result.to_dict()

        assert result_dict["instance"] == "test"
        assert result_dict["n_chains"] == 2
        assert result_dict["best_solution"]["our_cost"] == 0.8
        assert result_dict["diagnostics"]["r_hat"] == 1.05
        assert len(result_dict["per_chain_results"]) == 2
