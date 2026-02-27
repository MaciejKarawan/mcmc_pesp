"""GTFS loader for periodic weekday instances.

Builds Network + TransferSet + baseline Schedule from a GTFS feed
for a selected weekday profile.
"""

from __future__ import annotations

import csv
import io
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any
from zipfile import ZipFile

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import DEFAULT_TRANSFER_PARAMS, Transfer, TransferSet

WEEKDAY_NAMES = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

ROUTE_TYPE_TO_MODALITY = {
    "0": "tram",
    "1": "subway",
    "2": "rail",
    "3": "bus",
    "4": "ferry",
    "5": "cable_tram",
    "6": "aerial_lift",
    "7": "funicular",
    "11": "trolleybus",
    "12": "monorail",
}


@dataclass(frozen=True)
class GTFSLoadResult:
    """Result of loading a weekday GTFS slice."""

    network: Network
    transfers: TransferSet
    baseline_schedule: Schedule
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _StopInfo:
    stop_id: str
    location: str
    name: str
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class _TripStopTime:
    stop_id: str
    arrival_sec: int
    departure_sec: int
    stop_sequence: int


@dataclass
class _LineAccumulator:
    route_id: str
    direction_id: str
    stop_sequence: tuple[str, ...]
    trip_ids: list[str] = field(default_factory=list)
    service_ids: set[str] = field(default_factory=set)
    first_departures: list[int] = field(default_factory=list)
    travel_samples: list[list[int]] = field(default_factory=list)
    dwell_samples: list[list[int]] = field(default_factory=list)
    route_type: str = ""

    def add_trip(
        self,
        trip_id: str,
        service_id: str,
        first_departure: int,
        travel_times: list[int],
        dwell_times: list[int],
    ) -> None:
        if not self.travel_samples:
            self.travel_samples = [[] for _ in range(len(travel_times))]
        if not self.dwell_samples:
            self.dwell_samples = [[] for _ in range(len(dwell_times))]

        if len(self.travel_samples) != len(travel_times):
            return
        if len(self.dwell_samples) != len(dwell_times):
            return

        self.trip_ids.append(trip_id)
        self.service_ids.add(service_id)
        self.first_departures.append(first_departure)

        for i, value in enumerate(travel_times):
            self.travel_samples[i].append(value)
        for i, value in enumerate(dwell_times):
            self.dwell_samples[i].append(value)

    @property
    def n_trips(self) -> int:
        return len(self.trip_ids)


class _GTFSReader:
    def __init__(self, feed_path: str | Path):
        self.feed_path = Path(feed_path)
        if not self.feed_path.exists():
            raise FileNotFoundError(f"GTFS path does not exist: {self.feed_path}")
        self._is_zip = self.feed_path.is_file() and self.feed_path.suffix.lower() == ".zip"
        if not self._is_zip and not self.feed_path.is_dir():
            raise ValueError(f"GTFS path must be a directory or .zip: {self.feed_path}")

    def exists(self, filename: str) -> bool:
        try:
            self._read_text(filename)
            return True
        except FileNotFoundError:
            return False

    def read_csv(self, filename: str) -> list[dict[str, str]]:
        text = self._read_text(filename)
        reader = csv.DictReader(io.StringIO(text))
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
        return rows

    def _read_text(self, filename: str) -> str:
        if self._is_zip:
            return self._read_zip_text(filename)

        path = self.feed_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing GTFS file: {path}")
        return path.read_text(encoding="utf-8-sig")

    def _read_zip_text(self, filename: str) -> str:
        with ZipFile(self.feed_path) as zf:
            candidates = [
                name for name in zf.namelist()
                if name.lower() == filename.lower() or name.lower().endswith(f"/{filename.lower()}")
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"Missing GTFS file '{filename}' in zip: {self.feed_path}"
                )
            selected = min(candidates, key=len)
            with zf.open(selected) as handle:
                return handle.read().decode("utf-8-sig")


def _parse_hhmmss_to_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid GTFS time format: '{value}'")
    hours, minutes, seconds = (int(p) for p in parts)
    return hours * 3600 + minutes * 60 + seconds


def _coerce_time(
    arrival_raw: str,
    departure_raw: str,
    previous_departure: int | None,
) -> tuple[int, int]:
    if not arrival_raw and not departure_raw:
        raise ValueError("Both arrival_time and departure_time are empty")

    if arrival_raw:
        arrival_sec = _parse_hhmmss_to_seconds(arrival_raw)
    else:
        arrival_sec = _parse_hhmmss_to_seconds(departure_raw)

    if departure_raw:
        departure_sec = _parse_hhmmss_to_seconds(departure_raw)
    else:
        departure_sec = arrival_sec

    if previous_departure is not None:
        while arrival_sec < previous_departure:
            arrival_sec += 24 * 3600

    while departure_sec < arrival_sec:
        departure_sec += 24 * 3600

    if previous_departure is not None:
        while departure_sec < previous_departure:
            departure_sec += 24 * 3600

    return arrival_sec, departure_sec


def _safe_round_median(values: list[int], minimum: int) -> int:
    if not values:
        return minimum
    return max(minimum, int(round(float(median(values)))))


def _derive_frequency_minutes(first_departures: list[int], period: int) -> int:
    if len(first_departures) < 2:
        return period

    sorted_dep = sorted(first_departures)
    diffs = [b - a for a, b in zip(sorted_dep[:-1], sorted_dep[1:]) if b > a]
    if not diffs:
        return period
    return _safe_round_median(diffs, 1) // 60 if _safe_round_median(diffs, 1) >= 60 else 1


def _modality_from_route_type(route_type: str) -> str:
    route_type_clean = route_type.strip()
    return ROUTE_TYPE_TO_MODALITY.get(route_type_clean, f"route_type_{route_type_clean or 'unknown'}")


def _parse_optional_float(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _normalize_text(value: str) -> str:
    text = (value or "").translate(str.maketrans({
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }))
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower()


def _haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _read_stops(reader: _GTFSReader) -> dict[str, _StopInfo]:
    rows = reader.read_csv("stops.txt")
    result: dict[str, _StopInfo] = {}
    for row in rows:
        stop_id = row.get("stop_id", "")
        if not stop_id:
            continue
        parent_station = row.get("parent_station", "")
        location = parent_station if parent_station else stop_id
        result[stop_id] = _StopInfo(
            stop_id=stop_id,
            location=location,
            name=row.get("stop_name", ""),
            lat=_parse_optional_float(row.get("stop_lat", "")),
            lon=_parse_optional_float(row.get("stop_lon", "")),
        )
    return result


def _active_services_for_weekday(
    calendar_rows: list[dict[str, str]],
    weekday: str,
    fallback_service_ids: set[str],
) -> set[str]:
    if not calendar_rows:
        return set(fallback_service_ids)

    services = {
        row.get("service_id", "")
        for row in calendar_rows
        if row.get(weekday, "0") == "1" and row.get("service_id", "")
    }
    return services if services else set(fallback_service_ids)


def _build_trip_stop_times(
    stop_times_rows: list[dict[str, str]],
) -> dict[str, list[_TripStopTime]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in stop_times_rows:
        trip_id = row.get("trip_id", "")
        if not trip_id:
            continue
        grouped.setdefault(trip_id, []).append(row)

    result: dict[str, list[_TripStopTime]] = {}
    for trip_id, rows in grouped.items():
        try:
            sorted_rows = sorted(rows, key=lambda r: int(r.get("stop_sequence", "0")))
        except ValueError:
            continue

        parsed: list[_TripStopTime] = []
        previous_departure: int | None = None
        for row in sorted_rows:
            stop_id = row.get("stop_id", "")
            if not stop_id:
                continue
            try:
                sequence = int(row.get("stop_sequence", "0"))
                arrival_sec, departure_sec = _coerce_time(
                    row.get("arrival_time", ""),
                    row.get("departure_time", ""),
                    previous_departure,
                )
            except (TypeError, ValueError):
                continue

            previous_departure = departure_sec
            parsed.append(
                _TripStopTime(
                    stop_id=stop_id,
                    arrival_sec=arrival_sec,
                    departure_sec=departure_sec,
                    stop_sequence=sequence,
                )
            )

        if len(parsed) >= 2:
            result[trip_id] = parsed

    return result


def _trip_travel_and_dwell(stop_times: list[_TripStopTime]) -> tuple[list[int], list[int]]:
    dwell: list[int] = []
    for st in stop_times:
        dwell.append(max(0, st.departure_sec - st.arrival_sec))

    travel: list[int] = []
    for i in range(len(stop_times) - 1):
        segment = stop_times[i + 1].arrival_sec - stop_times[i].departure_sec
        while segment < 0:
            segment += 24 * 3600
        travel.append(segment)

    return travel, dwell


def _line_intensity(period: int, frequency: int) -> float:
    if frequency <= 0:
        return 1.0
    return max(1.0, period / float(frequency))


def _location_has_keyword(
    location_id: str,
    location_names: dict[str, set[str]],
    keywords: set[str],
) -> bool:
    if not keywords:
        return False
    variants = {location_id}
    variants.update(location_names.get(location_id, set()))
    normalized_variants = [_normalize_text(v) for v in variants]
    for keyword in keywords:
        for value in normalized_variants:
            if keyword in value:
                return True
    return False


def _mean_coordinate(samples: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not samples:
        return None
    lat = sum(point[0] for point in samples) / len(samples)
    lon = sum(point[1] for point in samples) / len(samples)
    return lat, lon


def _select_wroclaw_focus_locations(
    location_members: dict[str, dict[str, int]],
    location_names: dict[str, set[str]],
    location_coords: dict[str, tuple[float, float]],
    line_locations: dict[str, set[str]],
    line_trip_counts: dict[str, int],
    nearest_count: int,
    buffer_hubs_count: int,
    include_keywords: list[str] | None,
) -> tuple[set[str], dict[str, Any]]:
    keywords = {_normalize_text(k) for k in (include_keywords or ["olawa", "brzeg"]) if k}
    wroclaw_keyword = _normalize_text("wroclaw")
    glowny_keyword = _normalize_text("wroclaw glowny")

    all_locations = set(location_members.keys())
    wroclaw_locations = {
        loc for loc in all_locations
        if _location_has_keyword(loc, location_names, {wroclaw_keyword})
    }

    if not wroclaw_locations:
        return set(all_locations), {
            "focus_mode": "fallback_all_no_wroclaw",
            "focus_locations": sorted(all_locations),
        }

    anchor_candidates = [
        loc for loc in wroclaw_locations
        if _location_has_keyword(loc, location_names, {glowny_keyword})
    ]
    anchor = anchor_candidates[0] if anchor_candidates else sorted(wroclaw_locations)[0]

    anchor_coord = location_coords.get(anchor)
    if anchor_coord is None:
        wroclaw_coord_samples = [
            location_coords[loc] for loc in wroclaw_locations if loc in location_coords
        ]
        if wroclaw_coord_samples:
            anchor_coord = _mean_coordinate(wroclaw_coord_samples)

    nearest_locations: list[str] = []
    if anchor_coord is not None and nearest_count > 0:
        distances: list[tuple[float, str]] = []
        for loc, coord in location_coords.items():
            if loc in wroclaw_locations:
                continue
            dist = _haversine_km(anchor_coord[0], anchor_coord[1], coord[0], coord[1])
            distances.append((dist, loc))
        distances.sort(key=lambda item: (item[0], item[1]))
        nearest_locations = [loc for _, loc in distances[:nearest_count]]

    keyword_locations = {
        loc for loc in all_locations
        if _location_has_keyword(loc, location_names, keywords)
    }

    wroclaw_line_ids: set[str] = set()
    for loc in wroclaw_locations:
        wroclaw_line_ids.update(location_members.get(loc, {}).keys())

    buffer_scores: dict[str, int] = {}
    for line_id in wroclaw_line_ids:
        for loc in line_locations.get(line_id, set()):
            if loc in wroclaw_locations:
                continue
            buffer_scores[loc] = buffer_scores.get(loc, 0) + line_trip_counts.get(line_id, 0)

    buffer_candidates = sorted(
        buffer_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    buffer_locations = [
        loc for loc, _ in buffer_candidates
        if loc not in keyword_locations and loc not in nearest_locations
    ][:max(0, buffer_hubs_count)]

    focus_locations = set(wroclaw_locations)
    focus_locations.update(nearest_locations)
    focus_locations.update(keyword_locations)
    focus_locations.update(buffer_locations)

    metadata = {
        "focus_mode": "wroclaw_region",
        "focus_anchor": anchor,
        "focus_wroclaw_locations": sorted(wroclaw_locations),
        "focus_nearest_locations": sorted(nearest_locations),
        "focus_keyword_locations": sorted(keyword_locations),
        "focus_buffer_locations": sorted(buffer_locations),
        "focus_locations": sorted(focus_locations),
    }
    return focus_locations, metadata


def load_gtfs_weekday(
    feed_path: str | Path,
    weekday: str = "monday",
    period: int = 1440,
    min_trips_per_pattern: int = 2,
    walk_time: float = 3.0,
    max_lines: int | None = None,
    max_transfers_per_location: int | None = 400,
    transfer_weight_mode: str = "uniform",
    focus_wroclaw_region: bool = False,
    focus_nearest_count: int = 4,
    focus_buffer_hubs_count: int = 4,
    focus_include_keywords: list[str] | None = None,
) -> GTFSLoadResult:
    """Load GTFS feed and build a periodic weekday instance.

    Args:
        feed_path: Path to GTFS directory or .zip.
        weekday: Weekday profile name.
        period: Scheduling period in minutes.
        min_trips_per_pattern: Minimum trips required per line pattern.
        walk_time: Transfer walking time in minutes.
        max_lines: Optional cap for number of generated lines.
        max_transfers_per_location: Optional cap per location for directed pairs.
        transfer_weight_mode: "uniform" or "frequency".
        focus_wroclaw_region: Restrict transfers to Wroclaw-centric focus region.
        focus_nearest_count: Number of nearest stations around Wroclaw to include.
        focus_buffer_hubs_count: Number of high-volume buffer stations to include.
        focus_include_keywords: Additional station keywords to include (default Oława/Brzeg).

    Returns:
        GTFSLoadResult with network, transfer set and baseline schedule.
    """
    weekday_norm = weekday.lower()
    if weekday_norm not in WEEKDAY_NAMES:
        raise ValueError(f"weekday must be one of {sorted(WEEKDAY_NAMES)}, got '{weekday}'")
    if transfer_weight_mode not in {"uniform", "frequency"}:
        raise ValueError(
            "transfer_weight_mode must be 'uniform' or 'frequency', "
            f"got '{transfer_weight_mode}'"
        )

    reader = _GTFSReader(feed_path)
    required = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
    for filename in required:
        if not reader.exists(filename):
            raise FileNotFoundError(f"Missing required GTFS file: {filename}")

    stops = _read_stops(reader)
    routes_rows = reader.read_csv("routes.txt")
    trips_rows = reader.read_csv("trips.txt")
    stop_times_rows = reader.read_csv("stop_times.txt")
    calendar_rows = reader.read_csv("calendar.txt") if reader.exists("calendar.txt") else []

    route_type_by_id = {
        row.get("route_id", ""): row.get("route_type", "")
        for row in routes_rows
        if row.get("route_id", "")
    }

    all_service_ids = {
        row.get("service_id", "") for row in trips_rows if row.get("service_id", "")
    }
    active_service_ids = _active_services_for_weekday(
        calendar_rows=calendar_rows,
        weekday=weekday_norm,
        fallback_service_ids=all_service_ids,
    )

    trip_stop_times = _build_trip_stop_times(stop_times_rows)

    grouped: dict[tuple[str, str, tuple[str, ...]], _LineAccumulator] = {}
    for trip in trips_rows:
        trip_id = trip.get("trip_id", "")
        route_id = trip.get("route_id", "")
        service_id = trip.get("service_id", "")
        if not trip_id or not route_id or service_id not in active_service_ids:
            continue
        if trip_id not in trip_stop_times:
            continue

        stop_times = trip_stop_times[trip_id]
        stop_sequence = tuple(st.stop_id for st in stop_times)
        direction_id = trip.get("direction_id", "") or "na"
        key = (route_id, direction_id, stop_sequence)

        if key not in grouped:
            grouped[key] = _LineAccumulator(
                route_id=route_id,
                direction_id=direction_id,
                stop_sequence=stop_sequence,
                route_type=route_type_by_id.get(route_id, ""),
            )

        travel_sec, dwell_sec = _trip_travel_and_dwell(stop_times)
        first_dep_sec = stop_times[0].departure_sec
        grouped[key].add_trip(
            trip_id=trip_id,
            service_id=service_id,
            first_departure=first_dep_sec,
            travel_times=travel_sec,
            dwell_times=dwell_sec,
        )

    candidates = [
        group for group in grouped.values() if group.n_trips >= min_trips_per_pattern
    ]
    effective_min_trips = min_trips_per_pattern
    if not candidates and min_trips_per_pattern > 1:
        candidates = [group for group in grouped.values() if group.n_trips >= 1]
        effective_min_trips = 1

    candidates.sort(key=lambda g: (-g.n_trips, g.route_id, g.direction_id))
    if max_lines is not None:
        candidates = candidates[:max_lines]

    network = Network(period=period)
    baseline_offsets: dict[str, int] = {}
    line_trip_counts: dict[str, int] = {}
    line_service_ids: dict[str, list[str]] = {}
    location_members: dict[str, dict[str, int]] = {}
    location_names: dict[str, set[str]] = {}
    location_coord_samples: dict[str, list[tuple[float, float]]] = {}
    line_locations: dict[str, set[str]] = {}

    line_counters: dict[tuple[str, str], int] = {}
    for group in candidates:
        pair = (group.route_id, group.direction_id)
        line_counters[pair] = line_counters.get(pair, 0) + 1
        line_id = f"{group.route_id}:{group.direction_id}:{line_counters[pair]}"

        travel_minutes = [
            _safe_round_median(samples, 0) // 60 for samples in group.travel_samples
        ]
        dwell_minutes = [
            _safe_round_median(samples, 0) // 60 for samples in group.dwell_samples
        ]
        frequency = _derive_frequency_minutes(group.first_departures, period)
        if frequency <= 0:
            frequency = period

        stations: list[Station] = []
        touched_locations: set[str] = set()
        for idx, stop_id in enumerate(group.stop_sequence):
            stop_info = stops.get(
                stop_id,
                _StopInfo(
                    stop_id=stop_id,
                    location=stop_id,
                    name=stop_id,
                    lat=None,
                    lon=None,
                ),
            )
            stations.append(
                Station(
                    id=stop_id,
                    location=stop_info.location,
                    modality=_modality_from_route_type(group.route_type),
                )
            )
            location_members.setdefault(stop_info.location, {})
            if line_id not in location_members[stop_info.location]:
                location_members[stop_info.location][line_id] = idx
            touched_locations.add(stop_info.location)
            if stop_info.name:
                location_names.setdefault(stop_info.location, set()).add(stop_info.name)
            if stop_info.lat is not None and stop_info.lon is not None:
                location_coord_samples.setdefault(stop_info.location, []).append(
                    (stop_info.lat, stop_info.lon)
                )

        line = Line(
            id=line_id,
            modality=_modality_from_route_type(group.route_type),
            stations=stations,
            travel_times=travel_minutes,
            dwell_times=dwell_minutes,
            frequency=frequency,
        )
        network.add_line(line)

        baseline_offsets[line_id] = (int(round(median(group.first_departures))) // 60) % period
        line_trip_counts[line_id] = group.n_trips
        line_service_ids[line_id] = sorted(group.service_ids)
        line_locations[line_id] = touched_locations

    if len(network) == 0:
        raise ValueError(
            "No lines were built from GTFS data. "
            "Try lowering min_trips_per_pattern or changing weekday."
        )

    location_coords: dict[str, tuple[float, float]] = {}
    for location, samples in location_coord_samples.items():
        mean_coord = _mean_coordinate(samples)
        if mean_coord is not None:
            location_coords[location] = mean_coord

    focus_metadata: dict[str, Any] = {}
    focus_locations = set(location_members.keys())
    if focus_wroclaw_region:
        focus_locations, focus_metadata = _select_wroclaw_focus_locations(
            location_members=location_members,
            location_names=location_names,
            location_coords=location_coords,
            line_locations=line_locations,
            line_trip_counts=line_trip_counts,
            nearest_count=focus_nearest_count,
            buffer_hubs_count=focus_buffer_hubs_count,
            include_keywords=focus_include_keywords,
        )

    transfers_dict: dict[str, Transfer] = {}
    transfer_counter = 0
    raw_weights: dict[str, float] = {}
    for location, members in sorted(location_members.items()):
        if location not in focus_locations:
            continue

        line_ids = sorted(members.keys())
        if len(line_ids) < 2:
            continue

        ordered_pairs: list[tuple[str, str]] = []
        for from_line in line_ids:
            for to_line in line_ids:
                if from_line != to_line:
                    ordered_pairs.append((from_line, to_line))

        if max_transfers_per_location is not None:
            ordered_pairs = ordered_pairs[:max_transfers_per_location]

        for from_line, to_line in ordered_pairs:
            transfer_counter += 1
            transfer_id = f"gtfs_tr_{transfer_counter}"

            if transfer_weight_mode == "frequency":
                from_intensity = _line_intensity(period, network.get_line(from_line).frequency)
                to_intensity = _line_intensity(period, network.get_line(to_line).frequency)
                raw_weight = math.sqrt(from_intensity * to_intensity)
            else:
                raw_weight = 1.0
            raw_weights[transfer_id] = raw_weight

            transfers_dict[transfer_id] = Transfer(
                id=transfer_id,
                from_line=from_line,
                from_station_idx=members[from_line],
                to_line=to_line,
                to_station_idx=members[to_line],
                walk_time=walk_time,
                params=DEFAULT_TRANSFER_PARAMS,
                weight=raw_weight,
            )

    if transfer_weight_mode == "frequency" and raw_weights:
        mean_weight = sum(raw_weights.values()) / len(raw_weights)
        if mean_weight > 0:
            for transfer in transfers_dict.values():
                transfer.weight = transfer.weight / mean_weight

    transfers = TransferSet(transfers=transfers_dict)
    baseline_schedule = Schedule(offsets=baseline_offsets, period=period)

    metadata: dict[str, Any] = {
        "feed_path": str(feed_path),
        "weekday": weekday_norm,
        "active_service_ids": sorted(active_service_ids),
        "effective_min_trips_per_pattern": effective_min_trips,
        "n_lines": len(network),
        "n_transfers": len(transfers),
        "line_trip_counts": line_trip_counts,
        "line_service_ids": line_service_ids,
        "transfer_weight_mode": transfer_weight_mode,
        "focus_wroclaw_region": focus_wroclaw_region,
    }
    metadata.update(focus_metadata)
    if focus_metadata:
        def _location_label(location_id: str) -> str:
            names = sorted(location_names.get(location_id, set()))
            return names[0] if names else location_id

        for key in [
            "focus_wroclaw_locations",
            "focus_nearest_locations",
            "focus_keyword_locations",
            "focus_buffer_locations",
            "focus_locations",
        ]:
            locations = focus_metadata.get(key, [])
            if locations:
                metadata[key.replace("_locations", "_names")] = [
                    _location_label(location_id) for location_id in locations
                ]

    return GTFSLoadResult(
        network=network,
        transfers=transfers,
        baseline_schedule=baseline_schedule,
        metadata=metadata,
    )
