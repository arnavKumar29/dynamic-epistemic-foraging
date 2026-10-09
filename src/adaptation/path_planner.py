"""
path_planner.py — D* Lite path planner with trust-modulated hazard cost field.

Why D* Lite (Koenig & Likhachev, 2002)?
-----------------------------------------
D* Lite is an incremental search algorithm that re-uses previous search
results. When trust changes, only affected vertices need re-expansion.
  - First plan : comparable cost to A*.
  - Trust update: O(k) repair (k = changed hazard cells), NOT O(n log n)
    full re-search.

This is critical: with 60+ trust updates per second across 4 agents,
full A* re-search every tick would be prohibitively slow.

Trust hazard field
------------------
For each peer j with trust < reroute_threshold:
  For cells within hazard_radius of j's position:
    hazard[cell] += (1 - trust_j) × (radius - distance) × sensitivity

Edge cost
---------
  W(u→v) = euclidean(u,v) × (1 + hazard[v])

Agents navigate around distrusted peers' zones without hardcoding any
explicit avoidance rule — the cost structure makes distrusted regions
unattractive, and the agent reroutes naturally.

Continuous-angle steering integration
---------------------------------------
In addition to the discrete grid planner, get_avoidance_vector(pos) returns
a continuous-space repulsion vector derived from the hazard field.  The
main simulation loop blends this into the FSM target steering so robots can
move at any angle, not just along grid directions.

All internal state is inspectable via:
  - get_hazard_snapshot()   : Serializable current hazard field.
  - last_replan_stats       : Details of the most recent replan event.
  - cost_to_goal()          : Current estimated path cost.
"""
from __future__ import annotations

import math
import heapq
from typing import Dict, List, Optional, Set, Tuple


class TrustDStarLite:
    """
    D* Lite planner on a discrete grid with dynamic trust hazard costs.

    Parameters
    ----------
    width             : Arena width in cells.
    height            : Arena height in cells.
    obstacles         : Set of blocked (x, y) cells.
    reroute_threshold : Trust below this → apply hazard cost.
    hazard_radius     : Cells around low-trust peer that receive hazard cost.
    sensitivity       : Multiplier for hazard cost strength.
    """

    INF = float("inf")

    def __init__(
        self,
        width:             int,
        height:            int,
        obstacles:         List[Tuple[int, int]],
        reroute_threshold: float = 0.40,
        hazard_radius:     int   = 12,
        sensitivity:       float = 7.0,
    ) -> None:
        self.W           = width
        self.H           = height
        self.obs         = set(obstacles)
        self.threshold   = reroute_threshold
        self.radius      = hazard_radius
        self.sensitivity = sensitivity

        # D* Lite state tables
        self.g:   Dict[Tuple[int,int], float] = {}
        self.rhs: Dict[Tuple[int,int], float] = {}
        self._U:  list                         = []   # priority queue
        self._counter: int                     = 0    # tie-break counter
        self._entry_finder: Dict[Tuple[int,int], tuple] = {}
        self.km: float = 0.0

        self.start:        Tuple[int,int] = (0, 0)
        self.goal:         Tuple[int,int] = (0, 0)
        self._last_start:  Tuple[int,int] = (0, 0)

        # Trust hazard field: cell → cost additive
        self.hazard: Dict[Tuple[int,int], float] = {}

        # Introspection
        self._last_replan_stats: dict = {}
        self._current_path: List[Tuple[int,int]] = []

    # ------------------------------------------------------------------ #
    # Cost & neighbors                                                     #
    # ------------------------------------------------------------------ #
    def _in_bounds(self, s: Tuple[int,int]) -> bool:
        return 0 <= s[0] < self.W and 0 <= s[1] < self.H and s not in self.obs

    def _neighbors(self, s: Tuple[int,int]) -> List[Tuple[int,int]]:
        """8-connected grid neighbors (supports diagonal movement)."""
        x, y = s
        return [
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0) and self._in_bounds((x + dx, y + dy))
        ]

    def _edge_cost(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        if b in self.obs:
            return self.INF
        base   = math.hypot(b[0] - a[0], b[1] - a[1])
        hazard = self.hazard.get(b, 0.0)
        return base * (1.0 + hazard)

    def _h(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        """Euclidean heuristic — admissible for the hazard cost function."""
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ------------------------------------------------------------------ #
    # D* Lite core internals                                               #
    # ------------------------------------------------------------------ #
    def _key(self, s: Tuple[int,int]) -> Tuple[float,float]:
        g_rhs = min(self.g.get(s, self.INF), self.rhs.get(s, self.INF))
        return (g_rhs + self._h(self.start, s) + self.km, g_rhs)

    def _push(self, s: Tuple[int,int]) -> None:
        if s in self._entry_finder:
            return
        k = self._key(s)
        self._counter += 1
        entry = (k, self._counter, s)
        self._entry_finder[s] = entry
        heapq.heappush(self._U, entry)

    def _remove(self, s: Tuple[int,int]) -> None:
        self._entry_finder.pop(s, None)

    def _pop(self) -> Optional[Tuple]:
        while self._U:
            k, count, s = heapq.heappop(self._U)
            if self._entry_finder.get(s) == (k, count, s):
                self._entry_finder.pop(s, None)
                return k, s
        return None

    def _update_vertex(self, s: Tuple[int,int]) -> None:
        if s != self.goal:
            preds = self._neighbors(s)
            if preds:
                self.rhs[s] = min(
                    self._edge_cost(s, n) + self.g.get(n, self.INF)
                    for n in preds
                )
            else:
                self.rhs[s] = self.INF
        self._remove(s)
        if self.g.get(s, self.INF) != self.rhs.get(s, self.INF):
            self._push(s)

    def _compute_path(self) -> None:
        while self._U:
            res = self._pop()
            if res is None:
                break
            k, s = res
            k_start = self._key(self.start)
            if k >= k_start and self.rhs.get(self.start, self.INF) == self.g.get(self.start, self.INF):
                break
            if self.g.get(s, self.INF) > self.rhs.get(s, self.INF):
                self.g[s] = self.rhs[s]
                for p in self._neighbors(s):
                    self._update_vertex(p)
            else:
                self.g[s] = self.INF
                self._update_vertex(s)
                for p in self._neighbors(s):
                    self._update_vertex(p)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def init(self, start: Tuple[int,int], goal: Tuple[int,int]) -> None:
        """Initialize D* Lite for a new planning episode."""
        self.g.clear()
        self.rhs.clear()
        self._U.clear()
        self._entry_finder.clear()
        self.km = 0.0

        self.start       = start
        self.goal        = goal
        self._last_start = start

        self.rhs[goal] = 0.0
        self._push(goal)
        self._compute_path()

    def get_path(self, max_steps: int = 2000) -> List[Tuple[int,int]]:
        """Extract best path from current start to goal by greedy g-descent."""
        cur  = self.start
        path = [cur]
        seen: Set[Tuple[int,int]] = {cur}

        for _ in range(max_steps):
            if cur == self.goal:
                break
            nbrs = self._neighbors(cur)
            if not nbrs:
                return []
            nxt = min(nbrs, key=lambda n: self._edge_cost(cur, n) + self.g.get(n, self.INF))
            if self.g.get(nxt, self.INF) == self.INF:
                return []
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
            path.append(cur)

        self._current_path = path
        return path

    def replan(
        self,
        new_start:       Tuple[int,int],
        trust_map:       Dict[str, float],
        peer_positions:  Dict[str, Tuple[int,int]],
        peer_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[int,int]]:
        """
        Incremental D* Lite replan after trust/position changes.
        Only updates vertices whose hazard cost changed — O(k) not O(n log n).
        """
        # Accumulate path displacement for consistency
        self.km += self._h(self._last_start, new_start)
        self._last_start = self.start = new_start

        # Compute old→new hazard field diff
        old_hazard    = dict(self.hazard)
        old_path_cost = self.cost_to_goal()
        self._update_hazard(trust_map, peer_positions, peer_thresholds)

        # Re-expand only changed cells and their neighbors
        changed_cells = set(old_hazard) | set(self.hazard)
        n_changed = 0
        for cell in changed_cells:
            if abs(old_hazard.get(cell, 0.0) - self.hazard.get(cell, 0.0)) > 1e-9:
                self._update_vertex(cell)
                for nbr in self._neighbors(cell):
                    self._update_vertex(nbr)
                n_changed += 1

        self._compute_path()
        path = self.get_path()
        new_path_cost = self.cost_to_goal()

        # Record replan statistics for dashboard
        avoided = [
            pid for pid, t in trust_map.items()
            if t < (peer_thresholds.get(pid, self.threshold) if peer_thresholds else self.threshold)
            and pid in peer_positions
        ]
        self._last_replan_stats = {
            "n_cells_changed":    n_changed,
            "total_hazard_cells": len(self.hazard),
            "old_path_cost":      round(old_path_cost, 3),
            "new_path_cost":      round(new_path_cost, 3),
            "cost_delta":         round(new_path_cost - old_path_cost, 3),
            "peers_avoided":      avoided,
            "path_length":        len(path),
        }

        return path

    def set_trust(
        self,
        trust_map:       Dict[str, float],
        peer_positions:  Dict[str, Tuple[int,int]],
        peer_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set hazard field from trust map (call before first init)."""
        self._update_hazard(trust_map, peer_positions, peer_thresholds)

    def cost_to_goal(self) -> float:
        """Return the planner's current estimated cost from start to goal."""
        return self.g.get(self.start, self.INF)

    @property
    def last_replan_stats(self) -> dict:
        """Statistics from the most recent replan call (for dashboard)."""
        return self._last_replan_stats

    def get_hazard_snapshot(self) -> Dict[str, float]:
        """
        Return a JSON-serializable snapshot of the current hazard field.
        Keys are "x,y" strings; values are float cost additions.
        Used for periodic logging and the D* Lite planner tab in the dashboard.
        """
        return {f"{k[0]},{k[1]}": round(v, 4) for k, v in self.hazard.items()}

    def get_avoidance_vector(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Continuous-space avoidance vector derived from the hazard field.

        Returns a (vx, vy) repulsion vector that pushes the agent away from
        high-hazard zones.  This is blended into the FSM steering target in
        main.py to give true omnidirectional trust-aware avoidance without
        snapping to grid directions.

        Physics analogy: each hazard cell acts as a repulsive charge; the
        force falls off with inverse distance squared.
        """
        fx, fy = 0.0, 0.0
        px, py = pos
        for (hx, hy), hcost in self.hazard.items():
            dx, dy = px - hx, py - hy
            dist_sq = dx * dx + dy * dy
            if dist_sq < 0.25:   # avoid singularity
                continue
            dist = math.sqrt(dist_sq)
            magnitude = hcost / (dist_sq + 1.0)
            fx += magnitude * (dx / dist)
            fy += magnitude * (dy / dist)
        # Clamp to reasonable magnitude
        mag = math.hypot(fx, fy)
        if mag > 8.0:
            fx, fy = fx / mag * 8.0, fy / mag * 8.0
        return (fx, fy)

    # ------------------------------------------------------------------ #
    # Hazard field                                                         #
    # ------------------------------------------------------------------ #
    def _update_hazard(
        self,
        trust_map:       Dict[str, float],
        peer_positions:  Dict[str, Tuple[int,int]],
        peer_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Recompute the hazard field from the current trust map.

        Only peers with trust < reroute_threshold generate hazard.
        Hazard cost falls off linearly with distance from the peer:
            cost(cell) += (1 - trust) × (radius - dist) × sensitivity
        """
        self.hazard.clear()
        for pid, pos in peer_positions.items():
            trust = trust_map.get(pid, 1.0)
            threshold = peer_thresholds.get(pid, self.threshold) if peer_thresholds else self.threshold
            if trust >= threshold:
                continue       # this peer is trusted — no hazard
            px, py = int(pos[0]), int(pos[1])
            for dx in range(-self.radius, self.radius + 1):
                for dy in range(-self.radius, self.radius + 1):
                    cell = (px + dx, py + dy)
                    d    = math.hypot(dx, dy)
                    if d <= self.radius and self._in_bounds(cell):
                        cost = (1.0 - trust) * (self.radius - d) * self.sensitivity
                        self.hazard[cell] = self.hazard.get(cell, 0.0) + cost
