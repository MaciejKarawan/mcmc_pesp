"""Tests for GTFS weekday A/B benchmark runner."""

from __future__ import annotations

from src.experiments.benchmark.run_gtfs_weekday_ab import (
    GTFSABConfig,
    run_gtfs_weekday_ab,
)


def test_run_gtfs_weekday_ab_improves_or_matches_baseline(tiny_gtfs_feed) -> None:
    config = GTFSABConfig(
        gtfs_path=str(tiny_gtfs_feed),
        weekday="monday",
        n_chains=2,
        n_steps_per_chain=200,
        thin=5,
        burn_in_fraction=0.1,
        seed=123,
        include_robustness=False,
    )

    result = run_gtfs_weekday_ab(config)
    payload = result.to_dict()

    assert payload["optimized"]["best_cost"] <= payload["baseline"]["cost"]
    assert payload["n_lines"] >= 2
    assert payload["n_transfers"] > 0
    assert len(payload["chains"]) == 2


def test_run_gtfs_weekday_ab_handles_line_cap(tiny_gtfs_feed) -> None:
    config = GTFSABConfig(
        gtfs_path=str(tiny_gtfs_feed),
        weekday="monday",
        n_chains=1,
        n_steps_per_chain=100,
        thin=5,
        burn_in_fraction=0.0,
        seed=7,
        include_robustness=False,
        max_lines=1,
        use_component_move=False,
        use_smart_repair=False,
    )

    result = run_gtfs_weekday_ab(config)
    payload = result.to_dict()

    assert payload["n_lines"] == 1
    assert payload["optimized"]["best_cost"] <= payload["baseline"]["cost"]
