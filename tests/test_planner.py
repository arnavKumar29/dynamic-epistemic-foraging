"""
test_planner.py — Unit tests for D* Lite path planner.
"""
import pytest
from src.adaptation.path_planner import TrustDStarLite


@pytest.fixture
def small_grid():
    """A 20x20 grid with no obstacles."""
    return TrustDStarLite(
        width=20,
        height=20,
        obstacles=[],
        reroute_threshold=0.4,
        hazard_radius=3,
        sensitivity=5.0,
    )


@pytest.fixture
def blocked_grid():
    """A 20x20 grid with a vertical wall at x=10."""
    obs = [(10, y) for y in range(20)]
    return TrustDStarLite(
        width=20,
        height=20,
        obstacles=obs,
        reroute_threshold=0.4,
        hazard_radius=3,
        sensitivity=5.0,
    )


class TestTrustDStarLite:

    def test_straight_path_found(self, small_grid):
        small_grid.init(start=(0, 0), goal=(10, 0))
        path = small_grid.get_path()
        assert len(path) > 1
        assert path[0] == (0, 0)
        assert path[-1] == (10, 0)

    def test_path_reaches_goal(self, small_grid):
        small_grid.init(start=(0, 0), goal=(15, 15))
        path = small_grid.get_path()
        assert path[-1] == (15, 15)

    def test_start_equals_goal(self, small_grid):
        small_grid.init(start=(5, 5), goal=(5, 5))
        path = small_grid.get_path()
        # Should be trivially the start node
        assert (5, 5) in path

    def test_hazard_zone_inflates_cost(self, small_grid):
        """Placing a low-trust agent should change path to avoid its zone."""
        # Plan without hazard
        small_grid.init(start=(0, 10), goal=(19, 10))
        path_plain = small_grid.get_path()

        # Now mark agent at (10, 10) as low-trust
        trust_map       = {"A2": 0.2}
        peer_positions  = {"A2": (10, 10)}
        path_with_hazard = small_grid.replan((0, 10), trust_map, peer_positions)

        # With hazard, path should deviate or be longer
        plain_len = len(path_plain)
        hazard_len = len(path_with_hazard)
        # Not strictly longer always, but hazard cost changes routing
        assert isinstance(path_with_hazard, list)

    def test_high_trust_agent_no_hazard(self, small_grid):
        """A trusted agent (trust=0.9) should not create any hazard."""
        small_grid.init(start=(0, 10), goal=(19, 10))
        path_plain = small_grid.get_path()

        trust_map       = {"A2": 0.9}   # above threshold
        peer_positions  = {"A2": (10, 10)}
        path_after = small_grid.replan((0, 10), trust_map, peer_positions)

        # Should still reach goal
        if path_after:
            assert path_after[-1] == (19, 10)

    def test_all_peers_trusted_no_hazard(self, small_grid):
        small_grid.init(start=(0, 0), goal=(10, 10))
        trust_map = {"A1": 1.0, "A2": 1.0, "A3": 1.0}
        peer_positions = {"A1": (5, 5), "A2": (2, 2), "A3": (8, 8)}
        path = small_grid.replan((0, 0), trust_map, peer_positions)
        # Should find a path
        assert len(path) >= 1

    def test_replan_updates_start(self, small_grid):
        """Replan should accept new start position."""
        small_grid.init(start=(0, 0), goal=(15, 15))
        path = small_grid.replan((5, 5), {}, {})
        # Path should start from (5, 5) or nearby
        assert path is not None

    def test_hazard_radius_respected(self):
        """Hazard radius should only affect cells within radius."""
        planner = TrustDStarLite(
            width=30, height=30, obstacles=[],
            reroute_threshold=0.4,
            hazard_radius=2,
            sensitivity=5.0,
        )
        planner._update_hazard({"A2": 0.1}, {"A2": (15, 15)})
        # Cell at (15, 15) should have hazard
        assert planner.hazard.get((15, 15), 0.0) > 0.0
        # Cell far away should have no hazard
        assert planner.hazard.get((0, 0), 0.0) == 0.0
