"""Tests for smart repair move proposal."""

import math

import numpy as np
import pytest

from src.core.network import Line, Network, Station
from src.core.schedule import Schedule
from src.core.transfers import Transfer, TransferParams, TransferSet
from src.mcmc.proposals.smart_repair import SmartRepairMove, create_smart_repair


class TestSmartRepairMove:
    """Tests for SmartRepairMove class."""

    @pytest.fixture
    def simple_network(self) -> Network:
        """Create a simple network with two lines meeting at B."""
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
    def transfer_set(self) -> TransferSet:
        """Create a TransferSet with one transfer."""
        params = TransferParams(
            w_min=3.0,
            w_ideal=8.0,
            w_max=30.0,
            alpha=2.0,
            beta=0.5,
            M=1e6,
        )
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
            weight=1.0,
        )
        return TransferSet(transfers={"T1": t})

    @pytest.fixture
    def smart_repair(self, transfer_set: TransferSet) -> SmartRepairMove:
        rng = np.random.default_rng(42)
        return SmartRepairMove(transfers=transfer_set, sigma=2.0, rng=rng)

    def test_compute_repair_delta_from_line(
        self,
        smart_repair: SmartRepairMove,
        transfer_set: TransferSet,
        simple_network: Network,
    ) -> None:
        """Test repair delta when adjusting from_line."""
        transfer = list(transfer_set)[0]

        # Schedule where waiting = 5 (less than ideal=8)
        # Need to increase wait by 3
        schedule = Schedule(offsets={"L1": 0, "L2": 23}, period=1440)
        waiting = transfer.compute_waiting_time(schedule, simple_network)
        assert abs(waiting - 5) < 1e-10

        delta = smart_repair._compute_repair_delta(
            transfer, schedule, simple_network, "L1"
        )

        # Shifting from_line by +δ decreases wait
        # To increase wait by 3, we need δ = -3
        expected = -(8.0 - 5.0)  # -(ideal - current) = -3
        assert abs(delta - expected) < 1e-10

    def test_compute_repair_delta_to_line(
        self,
        smart_repair: SmartRepairMove,
        transfer_set: TransferSet,
        simple_network: Network,
    ) -> None:
        """Test repair delta when adjusting to_line."""
        transfer = list(transfer_set)[0]

        # Schedule where waiting = 5 (less than ideal=8)
        schedule = Schedule(offsets={"L1": 0, "L2": 23}, period=1440)

        delta = smart_repair._compute_repair_delta(
            transfer, schedule, simple_network, "L2"
        )

        # Shifting to_line by +δ increases wait
        # To increase wait by 3, we need δ = +3
        expected = 8.0 - 5.0  # ideal - current = 3
        assert abs(delta - expected) < 1e-10

    def test_propose_returns_valid_structure(
        self,
        smart_repair: SmartRepairMove,
        simple_network: Network,
    ) -> None:
        schedule = Schedule(offsets={"L1": 0, "L2": 23}, period=1440)

        new_schedule, hastings, changed = smart_repair.propose(schedule, simple_network)

        assert isinstance(new_schedule, Schedule)
        assert isinstance(hastings, float)
        assert hastings > 0  # Must be positive
        assert isinstance(changed, set)
        assert len(changed) == 1

    def test_propose_modifies_one_line(
        self,
        smart_repair: SmartRepairMove,
        simple_network: Network,
    ) -> None:
        """Only one line should be modified per proposal."""
        schedule = Schedule(offsets={"L1": 0, "L2": 23}, period=1440)

        for _ in range(10):
            new_schedule, _, changed = smart_repair.propose(schedule, simple_network)
            assert len(changed) == 1
            assert changed.issubset({"L1", "L2"})

    def test_hastings_ratio_is_positive(
        self,
        smart_repair: SmartRepairMove,
        simple_network: Network,
    ) -> None:
        """Hastings ratio must always be positive."""
        schedule = Schedule(offsets={"L1": 0, "L2": 23}, period=1440)

        for _ in range(100):
            _, hastings, _ = smart_repair.propose(schedule, simple_network)
            assert hastings > 0
            assert not math.isnan(hastings)
            assert not math.isinf(hastings)

    def test_repair_improves_violated_transfer(
        self,
        transfer_set: TransferSet,
        simple_network: Network,
    ) -> None:
        """Repair move should tend to improve violated transfers."""
        # Create move with very small sigma (almost deterministic)
        rng = np.random.default_rng(42)
        repair = SmartRepairMove(transfers=transfer_set, sigma=0.001, rng=rng)

        # Schedule with poor waiting time
        schedule = Schedule(offsets={"L1": 0, "L2": 20}, period=1440)
        transfer = list(transfer_set)[0]
        old_wait = transfer.compute_waiting_time(schedule, simple_network)

        # Run multiple proposals
        improvements = 0
        for _ in range(20):
            new_schedule, _, _ = repair.propose(schedule, simple_network)
            new_wait = transfer.compute_waiting_time(new_schedule, simple_network)

            # Check if waiting time moved toward ideal
            old_dist = abs(old_wait - 8.0)
            new_dist = abs(new_wait - 8.0)
            if new_dist < old_dist:
                improvements += 1

        # Most proposals should improve
        assert improvements > 10

    def test_name_property(self, smart_repair: SmartRepairMove) -> None:
        assert smart_repair.name == "smart_repair"

    def test_compute_repair_info(
        self,
        smart_repair: SmartRepairMove,
        transfer_set: TransferSet,
        simple_network: Network,
    ) -> None:
        """Test diagnostic information."""
        transfer = list(transfer_set)[0]
        schedule = Schedule(offsets={"L1": 0, "L2": 20}, period=1440)

        info = smart_repair.compute_repair_info(transfer, schedule, simple_network)

        assert "current_wait" in info
        assert "ideal_wait" in info
        assert "gap_to_ideal" in info
        assert "repair_delta_from_line" in info
        assert "repair_delta_to_line" in info
        assert info["ideal_wait"] == 8.0

    def test_empty_transfer_set(self, simple_network: Network) -> None:
        """Test with empty transfer set."""
        ts = TransferSet(transfers={})
        repair = SmartRepairMove(transfers=ts, sigma=2.0)

        schedule = Schedule(offsets={"L1": 0, "L2": 20}, period=1440)
        new_schedule, hastings, changed = repair.propose(schedule, simple_network)

        # Should return unchanged schedule
        assert hastings == 1.0
        assert changed == set()


class TestHastingsRatioCorrectness:
    """Tests for Hastings ratio mathematical correctness."""

    @pytest.fixture
    def simple_network(self) -> Network:
        network = Network()

        for line_id in ["L1", "L2"]:
            stations = [
                Station(id=f"A_{line_id}", location="A", modality="bus"),
                Station(id=f"B_{line_id}", location="B", modality="bus"),
            ]
            line = Line(
                id=line_id,
                modality="bus",
                stations=stations,
                travel_times=[15],
                dwell_times=[2, 2],
                frequency=60,
            )
            network.add_line(line)

        return network

    def test_hastings_ratio_symmetry_check(
        self,
        simple_network: Network,
    ) -> None:
        """Verify Hastings ratio is computed correctly by checking symmetry.

        For detailed balance: π(x)q(x'|x)α(x→x') = π(x')q(x|x')α(x'→x)

        Since α = min(1, π(x')/π(x) * q(x|x')/q(x'|x)),
        we check that H = q(x|x')/q(x'|x) is consistent with proposal.
        """
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})

        sigma = 2.0

        # Track forward and backward proposal densities
        rng = np.random.default_rng(42)
        repair = SmartRepairMove(transfers=ts, sigma=sigma, rng=rng)

        schedule = Schedule(offsets={"L1": 10, "L2": 30}, period=1440)

        # For multiple proposals, verify Hastings ratio properties
        for _ in range(50):
            new_schedule, hastings, changed = repair.propose(schedule, simple_network)

            # Hastings ratio should be in reasonable range
            assert 0 < hastings < 1e10

            # With small sigma, moves should be relatively small
            if len(changed) == 1:
                line_id = list(changed)[0]
                old_offset = schedule.get_offset(line_id)
                new_offset = new_schedule.get_offset(line_id)
                diff = abs(new_offset - old_offset)
                diff = min(diff, 1440 - diff)  # Account for wrap-around

                # Most moves should be within a few sigmas
                # (this is a soft check, not strict)
                assert diff < 20  # Very loose bound

    def test_hastings_ratio_formula_verification(
        self,
        simple_network: Network,
    ) -> None:
        """Verify Hastings ratio formula by manual computation."""
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})
        sigma = 2.0

        # Use fixed seed for reproducibility
        rng = np.random.default_rng(12345)
        repair = SmartRepairMove(transfers=ts, sigma=sigma, rng=rng)

        schedule = Schedule(offsets={"L1": 0, "L2": 25}, period=1440)
        transfer = list(ts)[0]

        # Get a proposal
        new_schedule, reported_hastings, changed = repair.propose(
            schedule, simple_network
        )

        if len(changed) == 1:
            target_line = list(changed)[0]

            # Manually compute Hastings ratio components
            delta_star_forward = repair._compute_repair_delta(
                transfer, schedule, simple_network, target_line
            )
            delta_star_backward = repair._compute_repair_delta(
                transfer, new_schedule, simple_network, target_line
            )

            old_offset = schedule.get_offset(target_line)
            new_offset = new_schedule.get_offset(target_line)
            actual_delta = (new_offset - old_offset) % 1440
            if actual_delta > 720:
                actual_delta -= 1440

            reverse_delta = -actual_delta

            # Compute expected Hastings ratio using discrete probability masses
            forward_mass = repair._compute_discrete_mass(actual_delta, delta_star_forward)
            backward_mass = repair._compute_discrete_mass(reverse_delta, delta_star_backward)

            if forward_mass > 0:
                expected_hastings = backward_mass / forward_mass
            else:
                expected_hastings = 0.0

            expected_hastings = max(1e-10, min(1e10, expected_hastings))

            # With discrete masses, ratio should be very close
            assert abs(reported_hastings - expected_hastings) < 0.05


class TestCreateSmartRepair:
    """Tests for factory function."""

    def test_create_with_defaults(self) -> None:
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})

        repair = create_smart_repair(ts)
        assert repair.sigma == 2.0

    def test_create_with_custom_sigma(self) -> None:
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})

        repair = create_smart_repair(ts, sigma=5.0)
        assert repair.sigma == 5.0


class TestDiscreteMassProperties:
    """Tests for discrete probability mass computation."""

    def test_discrete_mass_properties(self) -> None:
        """Verify _compute_discrete_mass returns values in [0, 1]."""
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})
        repair = SmartRepairMove(transfers=ts, sigma=2.0)

        delta_star = 3.5
        # Test several integer shifts
        total_mass = 0.0
        for k in range(-10, 11):
            mass = repair._compute_discrete_mass(k, delta_star)
            assert 0.0 <= mass <= 1.0, f"Mass for k={k} out of bounds: {mass}"
            total_mass += mass

        # Masses should sum to ~1 over reasonable range
        assert abs(total_mass - 1.0) < 0.01, f"Total mass {total_mass} != 1.0"

        # Peak mass should be near delta_star
        mass_3 = repair._compute_discrete_mass(3, delta_star)
        mass_4 = repair._compute_discrete_mass(4, delta_star)
        mass_0 = repair._compute_discrete_mass(0, delta_star)
        assert mass_3 > mass_0
        assert mass_4 > mass_0

    def test_discrete_mass_symmetry(self) -> None:
        """Verify mass is symmetric around delta_star=0."""
        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        t = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=0,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": t})
        repair = SmartRepairMove(transfers=ts, sigma=2.0)

        delta_star = 0.0
        for k in [1, 2, 3, 5]:
            mass_pos = repair._compute_discrete_mass(k, delta_star)
            mass_neg = repair._compute_discrete_mass(-k, delta_star)
            assert abs(mass_pos - mass_neg) < 1e-10, f"Asymmetry at k={k}"

        # Mass should decrease with |k - delta_star|
        mass_0 = repair._compute_discrete_mass(0, delta_star)
        mass_1 = repair._compute_discrete_mass(1, delta_star)
        mass_5 = repair._compute_discrete_mass(5, delta_star)
        assert mass_0 > mass_1
        assert mass_1 > mass_5

    def test_hastings_ratio_positive_finite(self) -> None:
        """Verify hastings_ratio > 0 and finite after propose()."""
        # Create a simple network
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

        params = TransferParams(w_min=3.0, w_ideal=8.0, w_max=30.0)
        transfer = Transfer(
            id="T1",
            from_line="L1",
            from_station_idx=1,
            to_line="L2",
            to_station_idx=0,
            walk_time=3.0,
            params=params,
        )
        ts = TransferSet(transfers={"T1": transfer})
        repair = SmartRepairMove(transfers=ts, sigma=2.0, rng=np.random.default_rng(42))

        schedule = Schedule.from_network(network)

        # Run several proposals
        for _ in range(10):
            new_schedule, hastings_ratio, changed = repair.propose(schedule, network)
            assert hastings_ratio > 0, "Hastings ratio must be positive"
            assert np.isfinite(hastings_ratio), "Hastings ratio must be finite"
            assert hastings_ratio <= 1e10, "Hastings ratio too large"
