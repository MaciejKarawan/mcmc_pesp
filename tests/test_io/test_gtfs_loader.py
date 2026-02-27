"""Tests for GTFS weekday loader."""

from __future__ import annotations

from src.io.gtfs_loader import load_gtfs_weekday


def test_load_gtfs_weekday_filters_services(tiny_gtfs_feed) -> None:
    result = load_gtfs_weekday(
        feed_path=tiny_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
    )

    active_service_ids = result.metadata["active_service_ids"]
    assert "WD" in active_service_ids
    assert "WE" not in active_service_ids


def test_load_gtfs_weekday_keeps_modalities(tiny_gtfs_feed) -> None:
    result = load_gtfs_weekday(
        feed_path=tiny_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
    )

    modalities = {line.modality for line in result.network.lines.values()}
    assert "bus" in modalities
    assert "tram" in modalities


def test_load_gtfs_weekday_builds_baseline_and_transfers(tiny_gtfs_feed) -> None:
    result = load_gtfs_weekday(
        feed_path=tiny_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
    )

    assert set(result.baseline_schedule.offsets.keys()) == set(result.network.line_ids)
    assert len(result.transfers) > 0
    assert result.metadata["n_transfers"] == len(result.transfers)


def test_load_gtfs_weekday_frequency_weights(wroclaw_focus_gtfs_feed) -> None:
    uniform_result = load_gtfs_weekday(
        feed_path=wroclaw_focus_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
        transfer_weight_mode="uniform",
    )
    result = load_gtfs_weekday(
        feed_path=wroclaw_focus_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
        transfer_weight_mode="frequency",
    )

    weights = [transfer.weight for transfer in result.transfers]
    uniform_weights = [transfer.weight for transfer in uniform_result.transfers]
    assert len(weights) > 0
    assert min(weights) > 0
    assert max(weights) > min(weights)
    assert max(uniform_weights) == min(uniform_weights) == 1.0
    assert abs(sum(weights) / len(weights) - 1.0) < 1e-9


def test_load_gtfs_weekday_focus_wroclaw_region_filters_far_hub(
    wroclaw_focus_gtfs_feed,
) -> None:
    all_result = load_gtfs_weekday(
        feed_path=wroclaw_focus_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
    )
    focus_result = load_gtfs_weekday(
        feed_path=wroclaw_focus_gtfs_feed,
        weekday="monday",
        min_trips_per_pattern=2,
        focus_wroclaw_region=True,
        focus_nearest_count=4,
        focus_buffer_hubs_count=2,
    )

    assert len(focus_result.transfers) < len(all_result.transfers)
    assert "focus_locations" in focus_result.metadata

    far_locations = {"FH", "FA", "FB"}
    for transfer in focus_result.transfers:
        from_location = focus_result.network.get_line(transfer.from_line).stations[
            transfer.from_station_idx
        ].location
        to_location = focus_result.network.get_line(transfer.to_line).stations[
            transfer.to_station_idx
        ].location
        assert from_location not in far_locations
        assert to_location not in far_locations
