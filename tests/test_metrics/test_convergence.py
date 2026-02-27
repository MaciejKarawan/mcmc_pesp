"""Tests for convergence diagnostics module."""

import numpy as np
import pytest

from src.metrics.convergence import (
    MCMCDiagnostics,
    compute_ess,
    compute_ess_simple,
    compute_r_hat,
    diagnose_chains,
)


class TestComputeRHat:
    """Tests for R-hat (Gelman-Rubin) statistic."""

    def test_identical_chains_r_hat_near_one(self) -> None:
        """Identical chains should have R-hat ≈ 1.0."""
        # Create 4 identical chains
        rng = np.random.default_rng(42)
        base_chain = rng.normal(0, 1, 1000)
        chains = [base_chain.copy() for _ in range(4)]

        r_hat = compute_r_hat(chains)
        assert abs(r_hat - 1.0) < 0.01

    def test_well_mixed_chains_r_hat_near_one(self) -> None:
        """Well-mixed chains from same distribution should have R-hat ≈ 1.0."""
        rng = np.random.default_rng(42)
        chains = [rng.normal(0, 1, 1000) for _ in range(4)]

        r_hat = compute_r_hat(chains)
        # Should be close to 1, typically < 1.05 for well-mixed chains
        assert r_hat < 1.1

    def test_different_means_high_r_hat(self) -> None:
        """Chains with different means should have high R-hat."""
        rng = np.random.default_rng(42)
        # Chains with very different means
        chain1 = rng.normal(0, 0.1, 1000)
        chain2 = rng.normal(10, 0.1, 1000)  # Very different mean
        chain3 = rng.normal(0, 0.1, 1000)
        chain4 = rng.normal(10, 0.1, 1000)

        r_hat = compute_r_hat([chain1, chain2, chain3, chain4])
        # Should be >> 1 due to between-chain variance
        assert r_hat > 1.5

    def test_single_chain_raises_error(self) -> None:
        """R-hat requires at least 2 chains."""
        chain = np.random.default_rng(42).normal(0, 1, 100)
        with pytest.raises(ValueError, match="at least 2 chains"):
            compute_r_hat([chain])

    def test_different_lengths_raises_error(self) -> None:
        """Chains must have same length."""
        chain1 = np.zeros(100)
        chain2 = np.zeros(200)
        with pytest.raises(ValueError, match="same length"):
            compute_r_hat([chain1, chain2])

    def test_short_chains_raises_error(self) -> None:
        """Chains must have at least 2 samples."""
        chain1 = np.array([1.0])
        chain2 = np.array([2.0])
        with pytest.raises(ValueError, match="at least 2 samples"):
            compute_r_hat([chain1, chain2])

    def test_constant_chains(self) -> None:
        """Constant chains should have R-hat = 1.0."""
        chain1 = np.ones(100) * 5.0
        chain2 = np.ones(100) * 5.0

        r_hat = compute_r_hat([chain1, chain2])
        assert r_hat == 1.0


class TestComputeESS:
    """Tests for Effective Sample Size."""

    def test_independent_samples_ess_near_n(self) -> None:
        """Independent samples should have ESS ≈ N."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, 1000)

        ess = compute_ess(chain)
        # ESS should be close to N for independent samples
        assert ess > 800  # Allow some variance

    def test_correlated_samples_ess_less_than_n(self) -> None:
        """Correlated samples should have ESS < N."""
        rng = np.random.default_rng(42)

        # Create highly correlated chain (random walk)
        n = 1000
        chain = np.zeros(n)
        chain[0] = rng.normal(0, 1)
        for i in range(1, n):
            chain[i] = 0.95 * chain[i - 1] + rng.normal(0, 0.1)

        ess = compute_ess(chain)
        # ESS should be much smaller than N due to correlation
        assert ess < n / 2

    def test_constant_chain_ess_equals_n(self) -> None:
        """Constant chain has variance 0, ESS = N."""
        chain = np.ones(100) * 5.0
        ess = compute_ess(chain)
        assert ess == 100.0

    def test_short_chain_raises_error(self) -> None:
        """Chain must have at least 4 samples."""
        chain = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="at least 4 samples"):
            compute_ess(chain)

    def test_ess_bounded_by_n(self) -> None:
        """ESS should never exceed N."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, 500)

        ess = compute_ess(chain)
        assert ess <= 500

    def test_ess_at_least_one(self) -> None:
        """ESS should be at least 1."""
        rng = np.random.default_rng(42)
        n = 100
        chain = np.zeros(n)
        chain[0] = 0
        for i in range(1, n):
            chain[i] = 0.99 * chain[i - 1] + rng.normal(0, 0.01)

        ess = compute_ess(chain)
        assert ess >= 1.0


class TestComputeESSSimple:
    """Tests for simple ESS implementation."""

    def test_consistent_with_fft_version(self) -> None:
        """Simple version should give similar results to FFT version."""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, 500)

        ess_fft = compute_ess(chain)
        ess_simple = compute_ess_simple(chain)

        # Should be reasonably close
        assert abs(ess_fft - ess_simple) / ess_fft < 0.1


class TestMCMCDiagnostics:
    """Tests for MCMCDiagnostics class."""

    @pytest.fixture
    def converged_chains(self) -> list[list[float]]:
        """Create chains that appear converged."""
        rng = np.random.default_rng(42)
        return [list(rng.normal(0, 1, 500)) for _ in range(4)]

    @pytest.fixture
    def unconverged_chains(self) -> list[list[float]]:
        """Create chains that are not converged."""
        rng = np.random.default_rng(42)
        # Different means
        return [
            list(rng.normal(0, 1, 500)),
            list(rng.normal(5, 1, 500)),
            list(rng.normal(0, 1, 500)),
            list(rng.normal(5, 1, 500)),
        ]

    def test_r_hat_converged(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        assert diag.r_hat() < 1.1

    def test_r_hat_unconverged(self, unconverged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=unconverged_chains)
        assert diag.r_hat() > 1.1

    def test_ess_returns_list(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        ess_values = diag.ess()

        assert len(ess_values) == 4
        for ess in ess_values:
            assert ess > 0

    def test_ess_total(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        assert diag.ess_total() == sum(diag.ess())

    def test_min_ess(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        assert diag.min_ess() == min(diag.ess())

    def test_mean_ess(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        ess_values = diag.ess()
        expected = sum(ess_values) / len(ess_values)
        assert abs(diag.mean_ess() - expected) < 1e-10

    def test_is_converged_true(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        # With sufficient samples from same distribution
        assert diag.is_converged(r_hat_threshold=1.1, min_ess_threshold=50)

    def test_is_converged_false_r_hat(
        self, unconverged_chains: list[list[float]]
    ) -> None:
        diag = MCMCDiagnostics(chains=unconverged_chains)
        assert not diag.is_converged(r_hat_threshold=1.1, min_ess_threshold=50)

    def test_is_converged_false_ess(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        # Require very high ESS that won't be met
        assert not diag.is_converged(r_hat_threshold=1.1, min_ess_threshold=10000)

    def test_summary(self, converged_chains: list[list[float]]) -> None:
        diag = MCMCDiagnostics(chains=converged_chains)
        summary = diag.summary()

        assert "n_chains" in summary
        assert summary["n_chains"] == 4
        assert "chain_lengths" in summary
        assert "ess_per_chain" in summary
        assert "ess_total" in summary
        assert "ess_min" in summary
        assert "ess_mean" in summary
        assert "r_hat" in summary
        assert "converged" in summary

    def test_single_chain_no_r_hat(self) -> None:
        """Single chain can't compute R-hat."""
        rng = np.random.default_rng(42)
        chains = [list(rng.normal(0, 1, 500))]

        diag = MCMCDiagnostics(chains=chains)

        with pytest.raises(ValueError, match="at least 2 chains"):
            diag.r_hat()

        # But summary should work
        summary = diag.summary()
        assert summary["r_hat"] is None

    def test_empty_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="At least one chain"):
            MCMCDiagnostics(chains=[])

    def test_short_chains_rejected(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            MCMCDiagnostics(chains=[[1.0, 2.0, 3.0]])


class TestDiagnoseChains:
    """Tests for convenience function."""

    def test_diagnose_returns_summary(self) -> None:
        rng = np.random.default_rng(42)
        chains = [list(rng.normal(0, 1, 500)) for _ in range(4)]

        summary = diagnose_chains(chains)

        assert "n_chains" in summary
        assert "r_hat" in summary
        assert "converged" in summary
        assert "r_hat_threshold" in summary
        assert "min_ess_threshold" in summary

    def test_custom_thresholds(self) -> None:
        rng = np.random.default_rng(42)
        chains = [list(rng.normal(0, 1, 500)) for _ in range(4)]

        summary = diagnose_chains(
            chains, r_hat_threshold=1.05, min_ess_threshold=200
        )

        assert summary["r_hat_threshold"] == 1.05
        assert summary["min_ess_threshold"] == 200


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_nearly_constant_chains(self) -> None:
        """Test with chains that have very small variance."""
        rng = np.random.default_rng(42)
        # Tiny variance
        chains = [
            list(5.0 + rng.normal(0, 1e-10, 100)),
            list(5.0 + rng.normal(0, 1e-10, 100)),
        ]

        diag = MCMCDiagnostics(chains=chains)
        r_hat = diag.r_hat()

        # Should handle gracefully
        assert not np.isnan(r_hat)
        assert not np.isinf(r_hat)

    def test_large_value_chains(self) -> None:
        """Test with chains having large values."""
        rng = np.random.default_rng(42)
        chains = [list(rng.normal(1e6, 100, 500)) for _ in range(4)]

        diag = MCMCDiagnostics(chains=chains)

        # Should work without overflow
        r_hat = diag.r_hat()
        assert not np.isnan(r_hat)

        ess_values = diag.ess()
        for ess in ess_values:
            assert not np.isnan(ess)
