"""Tests for proposal adaptation module."""

import pytest

from src.mcmc.adaptation import ProposalAdapter


class TestProposalAdapter:
    """Tests for ProposalAdapter."""

    def test_containment_mode_a(self) -> None:
        """Mode A: p_j >= epsilon for all j."""
        adapter = ProposalAdapter(n_proposals=3, mode="sampling", epsilon=0.05)

        # Simulate many steps with only proposal 0 accepted
        for _ in range(100):
            adapter.update(0, True)
            adapter.update(1, False)
            adapter.update(2, False)

        weights = adapter.get_weights()
        for w in weights:
            assert w >= adapter.epsilon

    def test_gamma_diminishes(self) -> None:
        """gamma_n -> 0 as n -> infinity."""
        adapter = ProposalAdapter(n_proposals=2, mode="sampling")

        gammas = []
        for i in range(100):
            gammas.append(adapter.gamma())
            adapter.update(0, True)

        # gamma should be decreasing
        for i in range(1, len(gammas)):
            assert gammas[i] <= gammas[i - 1]

        # gamma should be small after many steps
        assert adapter.gamma() < 0.1

    def test_sigma_bounded_mode_a(self) -> None:
        """Mode A: sigma in [sigma_min, sigma_max]."""
        adapter = ProposalAdapter(
            n_proposals=2,
            mode="sampling",
            sigma_min=0.5,
            sigma_max=50.0,
        )

        # Simulate extreme acceptance rates
        for _ in range(200):
            adapter.update(0, True)  # Very high acceptance
            adapter.update(1, False)  # Very low acceptance

        assert adapter.get_sigma(0) <= adapter.sigma_max
        assert adapter.get_sigma(0) >= adapter.sigma_min
        assert adapter.get_sigma(1) <= adapter.sigma_max
        assert adapter.get_sigma(1) >= adapter.sigma_min

    def test_weights_sum_to_one(self) -> None:
        """Weights should always sum to approximately 1."""
        adapter = ProposalAdapter(n_proposals=3, mode="optimization")

        for i in range(50):
            adapter.update(i % 3, i % 2 == 0)

        weights = adapter.get_weights()
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_initial_weights_uniform(self) -> None:
        """Initial weights should be uniform."""
        adapter = ProposalAdapter(n_proposals=4, mode="sampling")
        weights = adapter.get_weights()

        for w in weights:
            assert abs(w - 0.25) < 1e-10

    def test_optimization_mode_no_bounds(self) -> None:
        """Mode B: no sigma bounds enforced by get_sigma."""
        adapter = ProposalAdapter(
            n_proposals=2,
            mode="optimization",
            sigma_min=1.0,
            sigma_max=10.0,
        )

        # In optimization mode, sigma can go beyond bounds
        # (bounds are only for sampling mode)
        for _ in range(200):
            adapter.update(0, True)

        # get_sigma should return the raw value in optimization mode
        sigma = adapter.get_sigma(0)
        assert sigma > 0  # Just check it's positive
