"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import DEFAULT_TRANSFER_PARAMS, Transfer, TransferSet


@pytest.fixture
def simple_two_line_network() -> Network:
    """Create a simple network with two lines meeting at B.

    L1: A -> B (15 min travel, 2 min dwell)
    L2: B -> C (20 min travel, 2 min dwell)
    """
    network = Network()

    stations_l1 = [
        Station(id="A_L1", location="A", modality="bus"),
        Station(id="B_L1", location="B", modality="bus"),
    ]
    line1 = Line(
        id="L1",
        modality="bus",
        stations=stations_l1,
        travel_times=[15],
        dwell_times=[2, 2],
        frequency=60,
    )

    stations_l2 = [
        Station(id="B_L2", location="B", modality="bus"),
        Station(id="C_L2", location="C", modality="bus"),
    ]
    line2 = Line(
        id="L2",
        modality="bus",
        stations=stations_l2,
        travel_times=[20],
        dwell_times=[2, 2],
        frequency=60,
    )

    network.add_line(line1)
    network.add_line(line2)
    return network


@pytest.fixture
def transfer_l1_to_l2() -> Transfer:
    """Create a transfer from L1 to L2 at location B."""
    return Transfer(
        id="T_B_L1_L2",
        from_line="L1",
        from_station_idx=1,
        to_line="L2",
        to_station_idx=0,
        walk_time=3.0,
        params=DEFAULT_TRANSFER_PARAMS,
        weight=1.0,
    )


@pytest.fixture
def simple_transfer_set(transfer_l1_to_l2: Transfer) -> TransferSet:
    """Create a TransferSet with the L1->L2 transfer."""
    return TransferSet(transfers={transfer_l1_to_l2.id: transfer_l1_to_l2})


@pytest.fixture
def tiny_gtfs_feed(tmp_path: Path) -> Path:
    """Create a minimal GTFS feed with weekday + weekend services."""
    feed_dir = tmp_path / "tiny_gtfs"
    feed_dir.mkdir()

    (feed_dir / "stops.txt").write_text(
        "\n".join(
            [
                "stop_id,stop_name,parent_station",
                "A1,Stop A,",
                "HUB1,Hub Platform 1,HUB",
                "HUB2,Hub Platform 2,HUB",
                "C1,Stop C,",
                "D1,Stop D,",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "routes.txt").write_text(
        "\n".join(
            [
                "route_id,agency_id,route_short_name,route_long_name,route_type",
                "R_BUS,AG,B1,Bus Line,3",
                "R_TRAM,AG,T1,Tram Line,0",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "trips.txt").write_text(
        "\n".join(
            [
                "route_id,service_id,trip_id,direction_id",
                "R_BUS,WD,BUS1,0",
                "R_BUS,WD,BUS2,0",
                "R_TRAM,WD,TRAM1,0",
                "R_TRAM,WD,TRAM2,0",
                "R_TRAM,WE,TRAM_WE,0",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "stop_times.txt").write_text(
        "\n".join(
            [
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
                "BUS1,07:00:00,07:00:00,A1,1",
                "BUS1,07:10:00,07:11:00,HUB1,2",
                "BUS1,07:20:00,07:20:00,C1,3",
                "BUS2,07:30:00,07:30:00,A1,1",
                "BUS2,07:40:00,07:41:00,HUB1,2",
                "BUS2,07:50:00,07:50:00,C1,3",
                "TRAM1,07:05:00,07:05:00,D1,1",
                "TRAM1,07:14:00,07:15:00,HUB2,2",
                "TRAM1,07:25:00,07:25:00,C1,3",
                "TRAM2,07:25:00,07:25:00,D1,1",
                "TRAM2,07:34:00,07:35:00,HUB2,2",
                "TRAM2,07:45:00,07:45:00,C1,3",
                "TRAM_WE,10:00:00,10:00:00,D1,1",
                "TRAM_WE,10:10:00,10:10:00,HUB2,2",
                "TRAM_WE,10:20:00,10:20:00,C1,3",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "calendar.txt").write_text(
        "\n".join(
            [
                "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date",
                "WD,1,1,1,1,1,0,0,20250101,20261231",
                "WE,0,0,0,0,0,1,1,20250101,20261231",
            ]
        ),
        encoding="utf-8",
    )

    return feed_dir


@pytest.fixture
def wroclaw_focus_gtfs_feed(tmp_path: Path) -> Path:
    """Create a GTFS feed with Wroclaw and a far-away transfer hub."""
    feed_dir = tmp_path / "wroclaw_focus_gtfs"
    feed_dir.mkdir()

    (feed_dir / "stops.txt").write_text(
        "\n".join(
            [
                "stop_id,stop_name,parent_station,stop_lat,stop_lon",
                "WG,Wroclaw Glowny,,51.098,17.036",
                "WM,Wroclaw Mikolajow,,51.117,17.001",
                "WB,Wroclaw Brochow,,51.091,17.084",
                "OL,Olawa,,50.946,17.292",
                "BR,Brzeg,,50.860,17.467",
                "LG,Legnica,,51.210,16.161",
                "FH,FarHub,,50.000,15.000",
                "FA,FarA,,50.010,15.020",
                "FB,FarB,,50.020,15.010",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "routes.txt").write_text(
        "\n".join(
            [
                "route_id,agency_id,route_short_name,route_long_name,route_type",
                "R1,AG,R1,WG-WM-OL,2",
                "R2,AG,R2,BR-WG-WB,2",
                "R3,AG,R3,LG-WG,2",
                "R4,AG,R4,FH-FA,2",
                "R5,AG,R5,FB-FH,2",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "trips.txt").write_text(
        "\n".join(
            [
                "route_id,service_id,trip_id,direction_id",
                "R1,WD,R1A,0",
                "R1,WD,R1B,0",
                "R2,WD,R2A,0",
                "R2,WD,R2B,0",
                "R3,WD,R3A,0",
                "R3,WD,R3B,0",
                "R4,WD,R4A,0",
                "R4,WD,R4B,0",
                "R5,WD,R5A,0",
                "R5,WD,R5B,0",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "stop_times.txt").write_text(
        "\n".join(
            [
                "trip_id,arrival_time,departure_time,stop_id,stop_sequence",
                "R1A,07:00:00,07:00:00,WG,1",
                "R1A,07:10:00,07:10:00,WM,2",
                "R1A,07:20:00,07:20:00,OL,3",
                "R1B,07:30:00,07:30:00,WG,1",
                "R1B,07:40:00,07:40:00,WM,2",
                "R1B,07:50:00,07:50:00,OL,3",
                "R2A,07:05:00,07:05:00,BR,1",
                "R2A,07:15:00,07:15:00,WG,2",
                "R2A,07:25:00,07:25:00,WB,3",
                "R2B,07:35:00,07:35:00,BR,1",
                "R2B,07:45:00,07:45:00,WG,2",
                "R2B,07:55:00,07:55:00,WB,3",
                "R3A,07:02:00,07:02:00,LG,1",
                "R3A,07:18:00,07:18:00,WG,2",
                "R3B,08:02:00,08:02:00,LG,1",
                "R3B,08:18:00,08:18:00,WG,2",
                "R4A,07:00:00,07:00:00,FH,1",
                "R4A,07:10:00,07:10:00,FA,2",
                "R4B,07:30:00,07:30:00,FH,1",
                "R4B,07:40:00,07:40:00,FA,2",
                "R5A,07:05:00,07:05:00,FB,1",
                "R5A,07:15:00,07:15:00,FH,2",
                "R5B,07:35:00,07:35:00,FB,1",
                "R5B,07:45:00,07:45:00,FH,2",
            ]
        ),
        encoding="utf-8",
    )

    (feed_dir / "calendar.txt").write_text(
        "\n".join(
            [
                "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date",
                "WD,1,1,1,1,1,0,0,20250101,20261231",
            ]
        ),
        encoding="utf-8",
    )

    return feed_dir
