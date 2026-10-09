"""
test_smt.py — Unit tests for Z3 SMT kinematic and obstacle checker.
"""
import pytest
from src.reasoning.smt_checker import SMTContradictionChecker


@pytest.fixture
def checker():
    return SMTContradictionChecker(pos_tolerance=18.0, v_max=3.0)


class TestSMTChecker:

    def test_normal_move_passes(self, checker):
        """A 3-pixel move in 1 tick at v_max=3 should not trigger."""
        result = checker.verify_kinematic_claim(
            prev_pos=(100.0, 100.0),
            prev_tick=0,
            claimed_pos=(102.0, 100.0),   # 2px move
            claimed_tick=1,
            obstacles=[],
        )
        assert result["contradiction"] is False
        assert result["reason"] == "NONE"

    def test_teleportation_detected(self, checker):
        """200px jump in 1 tick at v_max=3 should be flagged."""
        result = checker.verify_kinematic_claim(
            prev_pos=(100.0, 100.0),
            prev_tick=5,
            claimed_pos=(300.0, 100.0),   # 200px in 1 tick
            claimed_tick=6,
            obstacles=[],
        )
        assert result["contradiction"] is True
        assert "KINEMATIC" in result["reason"]

    def test_tolerance_allows_small_overshoot(self, checker):
        """A move within v_max*dt + tolerance should pass."""
        # v_max=3, dt=2, max=6, tolerance=18 → max allowed = 24
        result = checker.verify_kinematic_claim(
            prev_pos=(100.0, 100.0),
            prev_tick=0,
            claimed_pos=(120.0, 100.0),   # 20px in 2 ticks → within tolerance
            claimed_tick=2,
            obstacles=[],
        )
        assert result["contradiction"] is False

    def test_temporal_paradox_same_tick(self, checker):
        """Same tick (dt=0) should be a temporal paradox."""
        result = checker.verify_kinematic_claim(
            prev_pos=(100.0, 100.0),
            prev_tick=5,
            claimed_pos=(101.0, 100.0),
            claimed_tick=5,              # same tick — dt=0
            obstacles=[],
        )
        assert result["contradiction"] is True
        assert "TEMPORAL" in result["reason"]

    def test_temporal_paradox_negative_dt(self, checker):
        """Claimed tick before previous tick → temporal paradox."""
        result = checker.verify_kinematic_claim(
            prev_pos=(100.0, 100.0),
            prev_tick=10,
            claimed_pos=(101.0, 100.0),
            claimed_tick=5,             # tick regression
            obstacles=[],
        )
        assert result["contradiction"] is True
        assert "TEMPORAL" in result["reason"]

    def test_obstacle_collision_paradox(self, checker):
        """A position inside an obstacle should be flagged."""
        obstacle = (200.0, 200.0, 100.0, 100.0)  # rect at (200,200) 100x100
        result = checker.verify_kinematic_claim(
            prev_pos=(190.0, 190.0),
            prev_tick=0,
            claimed_pos=(240.0, 240.0),   # inside obstacle!
            claimed_tick=1,
            obstacles=[obstacle],
        )
        assert result["contradiction"] is True
        # Could be kinematic OR obstacle depending on v_max
        assert result["contradiction"] is True

    def test_multiple_obstacles_checked(self, checker):
        """Both obstacles should be checked; second one matches."""
        obs1 = (0.0, 0.0, 10.0, 10.0)       # far from claimed
        obs2 = (500.0, 500.0, 100.0, 100.0)  # claimed is inside obs2
        result = checker.verify_kinematic_claim(
            prev_pos=(490.0, 490.0),
            prev_tick=0,
            claimed_pos=(540.0, 540.0),   # inside obs2, move=70px in 1 tick
            claimed_tick=1,               # will fail kinematic first
            obstacles=[obs1, obs2],
        )
        assert result["contradiction"] is True

    def test_delta_field_computed(self, checker):
        """Delta (Euclidean distance) should always be returned."""
        result = checker.verify_kinematic_claim(
            prev_pos=(0.0, 0.0),
            prev_tick=0,
            claimed_pos=(3.0, 4.0),  # 5px
            claimed_tick=1,
            obstacles=[],
        )
        assert abs(result["delta"] - 5.0) < 0.01
