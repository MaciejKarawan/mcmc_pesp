"""MCMC convergence diagnostics: R-hat and Effective Sample Size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def compute_r_hat(chains: Sequence[NDArray[np.floating]]) -> float:
    """Compute Gelman-Rubin R-hat statistic for convergence assessment.

    R-hat measures the ratio of total variance to within-chain variance.
    Values close to 1.0 indicate convergence; values > 1.1 suggest
    chains haven't mixed well.

    Formula:
        W = mean of within-chain variances
        B = N * variance of chain means
        V_hat = (N-1)/N * W + 1/N * B
        R_hat = sqrt(V_hat / W)

    Args:
        chains: List of M chains, each of length N.
                All chains must have the same length.

    Returns:
        R-hat value. Convergence indicated by R-hat < 1.1.

    Raises:
        ValueError: If fewer than 2 chains or chains have different lengths.
    """
    if len(chains) < 2:
        raise ValueError("R-hat requires at least 2 chains")

    chains_array = [np.asarray(c, dtype=np.float64) for c in chains]
    lengths = [len(c) for c in chains_array]

    if len(set(lengths)) != 1:
        raise ValueError(f"All chains must have same length, got {lengths}")

    M = len(chains_array)  # Number of chains
    N = lengths[0]  # Length of each chain

    if N < 2:
        raise ValueError("Chains must have at least 2 samples")

    # Compute chain means
    chain_means = np.array([np.mean(c) for c in chains_array])
    overall_mean = np.mean(chain_means)

    # Between-chain variance (B)
    B = N / (M - 1) * np.sum((chain_means - overall_mean) ** 2)

    # Within-chain variance (W)
    # Using ddof=1 for sample variance
    chain_vars = np.array([np.var(c, ddof=1) for c in chains_array])
    W = np.mean(chain_vars)

    # Handle edge case of zero variance
    if W == 0:
        if B == 0:
            return 1.0  # All chains identical and constant
        return float("inf")  # Chains constant but different means

    # Pooled variance estimate
    V_hat = (N - 1) / N * W + 1 / N * B

    # R-hat
    r_hat = np.sqrt(V_hat / W)

    return float(r_hat)


def compute_ess(
    chain: NDArray[np.floating],
    max_lag: int | None = None,
) -> float:
    """Compute Effective Sample Size for a single chain.

    ESS estimates the number of independent samples equivalent to the
    correlated MCMC chain. Higher ESS means better mixing.

    Formula:
        ESS = N / (1 + 2 * sum(autocorrelations))

    Uses Geyer's initial positive sequence method to truncate
    autocorrelation sum when consecutive pairs become non-positive.

    Args:
        chain: Array of MCMC samples.
        max_lag: Maximum lag for autocorrelation. Defaults to N//2.

    Returns:
        Effective sample size (can be < N for correlated samples).

    Raises:
        ValueError: If chain is too short.
    """
    chain = np.asarray(chain, dtype=np.float64)
    N = len(chain)

    if N < 4:
        raise ValueError("Chain must have at least 4 samples for ESS")

    if max_lag is None:
        max_lag = N // 2

    max_lag = min(max_lag, N - 1)

    # Compute mean and variance
    mean = np.mean(chain)
    var = np.var(chain, ddof=0)  # Population variance

    # Handle constant chain
    if var == 0:
        return float(N)

    # Compute autocorrelations using FFT for efficiency
    # Normalize chain
    centered = chain - mean

    # Use FFT-based autocorrelation
    fft = np.fft.fft(centered, n=2 * N)
    acf_full = np.fft.ifft(fft * np.conj(fft)).real[:N]
    acf = acf_full / acf_full[0]  # Normalize by variance

    # Compute sum of autocorrelations with Geyer's truncation
    # Sum consecutive pairs and stop when pair sum is negative
    autocorr_sum = 0.0

    for k in range(1, max_lag + 1, 2):
        if k + 1 <= max_lag:
            pair_sum = acf[k] + acf[k + 1]
        else:
            pair_sum = acf[k]

        if pair_sum < 0:
            break

        autocorr_sum += pair_sum

    # ESS formula
    ess = N / (1 + 2 * autocorr_sum)

    # Clamp to reasonable bounds
    ess = max(1.0, min(ess, float(N)))

    return ess


def compute_ess_simple(
    chain: NDArray[np.floating],
    max_lag: int | None = None,
) -> float:
    """Compute ESS using simple (non-FFT) autocorrelation.

    Slower but more straightforward implementation.
    Useful for verification and small chains.

    Args:
        chain: Array of MCMC samples.
        max_lag: Maximum lag for autocorrelation.

    Returns:
        Effective sample size.
    """
    chain = np.asarray(chain, dtype=np.float64)
    N = len(chain)

    if N < 4:
        raise ValueError("Chain must have at least 4 samples")

    if max_lag is None:
        max_lag = N // 2

    mean = np.mean(chain)
    var = np.var(chain, ddof=0)

    if var == 0:
        return float(N)

    autocorrs = []

    for k in range(1, max_lag + 1):
        c = np.mean((chain[:-k] - mean) * (chain[k:] - mean)) / var
        autocorrs.append(c)

        # Geyer's stopping rule: stop when consecutive autocorrs sum negative
        if len(autocorrs) >= 2:
            if autocorrs[-1] + autocorrs[-2] < 0:
                # Remove the last two and stop
                autocorrs = autocorrs[:-2]
                break

    ess = N / (1 + 2 * sum(autocorrs))
    return max(1.0, min(ess, float(N)))


@dataclass
class MCMCDiagnostics:
    """Diagnostics container for multiple MCMC chains.

    Provides methods to assess convergence across multiple chains
    using R-hat and ESS statistics.

    Args:
        chains: List of chains, where each chain is a list of samples.
    """

    chains: list[list[float]]

    def __post_init__(self) -> None:
        if len(self.chains) < 1:
            raise ValueError("At least one chain required")
        for i, chain in enumerate(self.chains):
            if len(chain) < 4:
                raise ValueError(f"Chain {i} too short: {len(chain)} < 4")

    def r_hat(self) -> float:
        """Compute R-hat statistic across all chains.

        Returns:
            R-hat value. Convergence indicated by < 1.1.

        Raises:
            ValueError: If fewer than 2 chains.
        """
        if len(self.chains) < 2:
            raise ValueError("R-hat requires at least 2 chains")

        arrays = [np.array(c) for c in self.chains]
        return compute_r_hat(arrays)

    def ess(self) -> list[float]:
        """Compute ESS for each chain.

        Returns:
            List of ESS values, one per chain.
        """
        return [compute_ess(np.array(c)) for c in self.chains]

    def ess_total(self) -> float:
        """Compute total ESS across all chains.

        Returns:
            Sum of ESS across chains.
        """
        return sum(self.ess())

    def min_ess(self) -> float:
        """Get minimum ESS across chains.

        Returns:
            Minimum ESS value.
        """
        return min(self.ess())

    def mean_ess(self) -> float:
        """Get mean ESS across chains.

        Returns:
            Average ESS value.
        """
        ess_values = self.ess()
        return sum(ess_values) / len(ess_values)

    def is_converged(
        self,
        r_hat_threshold: float = 1.1,
        min_ess_threshold: float = 100,
    ) -> bool:
        """Check if chains have converged.

        Convergence requires both:
        - R-hat < threshold (chains have mixed)
        - min ESS > threshold (enough effective samples)

        Args:
            r_hat_threshold: Maximum acceptable R-hat.
            min_ess_threshold: Minimum acceptable ESS per chain.

        Returns:
            True if converged, False otherwise.
        """
        if len(self.chains) < 2:
            # Can't compute R-hat with single chain
            return self.min_ess() > min_ess_threshold

        return (
            self.r_hat() < r_hat_threshold
            and self.min_ess() > min_ess_threshold
        )

    def summary(self) -> dict:
        """Get summary of convergence diagnostics.

        Returns:
            Dictionary with diagnostic values.
        """
        ess_values = self.ess()

        result = {
            "n_chains": len(self.chains),
            "chain_lengths": [len(c) for c in self.chains],
            "ess_per_chain": ess_values,
            "ess_total": sum(ess_values),
            "ess_min": min(ess_values),
            "ess_mean": sum(ess_values) / len(ess_values),
        }

        if len(self.chains) >= 2:
            result["r_hat"] = self.r_hat()
            result["converged"] = self.is_converged()
        else:
            result["r_hat"] = None
            result["converged"] = self.min_ess() > 100

        return result


def diagnose_chains(
    chains: list[list[float]],
    r_hat_threshold: float = 1.1,
    min_ess_threshold: float = 100,
) -> dict:
    """Convenience function to diagnose multiple chains.

    Args:
        chains: List of MCMC chains.
        r_hat_threshold: R-hat threshold for convergence.
        min_ess_threshold: Minimum ESS for convergence.

    Returns:
        Dictionary with diagnostic summary.
    """
    diag = MCMCDiagnostics(chains=chains)
    summary = diag.summary()
    summary["r_hat_threshold"] = r_hat_threshold
    summary["min_ess_threshold"] = min_ess_threshold
    return summary
