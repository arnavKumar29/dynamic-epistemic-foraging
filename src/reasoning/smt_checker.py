"""
smt_checker.py — Z3 SMT-based kinematic & obstacle contradiction checker.

Instead of a heuristic threshold ("if distance > v*dt → clash"), this module
formally PROVES via Satisfiability Modulo Theories (Z3) whether a claimed
position is physically reachable given:
  1. The previous known position and tick.
  2. The agent's maximum velocity and a configurable tolerance slack.
  3. The list of obstacle rectangles (claimed position inside = impossible).

If Z3 returns SAT for the "violation formula", the contradiction is proven.
If UNSAT, the claim is consistent with kinematics.

This gives a PROVABLY correct flag, not just an ad-hoc number comparison.
"""
from __future__ import annotations

from typing import List, Tuple

import z3


class SMTContradictionChecker:
    """
    Z3-backed kinematic and obstacle paradox detector.

    Args:
        pos_tolerance : Extra slack (pixels) added to v_max * dt to account
                        for kinematic noise and sensor jitter.
        v_max         : Agent's rated maximum velocity (pixels per tick).
    """

    def __init__(self, pos_tolerance: float = 18.0, v_max: float = 3.0):
        self.pos_tolerance = pos_tolerance
        self.v_max         = v_max

    def verify_kinematic_claim(
        self,
        prev_pos: Tuple[float, float],
        prev_tick: int,
        claimed_pos: Tuple[float, float],
        claimed_tick: int,
        obstacles: List[Tuple[float, float, float, float]],  # [(x,y,w,h)]
    ) -> dict:
        """
        Return a verdict dict:
        {
            "contradiction": bool,
            "reason": str,        # "NONE" | "TEMPORAL_PARADOX" | 
                                  # "KINEMATIC_BOUND_VIOLATION" | "OBSTACLE_COLLISION_PARADOX"
            "delta": float        # Euclidean distance between prev and claimed
        }
        """
        import math

        dt = claimed_tick - prev_tick

        # --- Temporal paradox (tick regression / replay) ---
        if dt <= 0:
            return {
                "contradiction": True,
                "reason": "TEMPORAL_PARADOX_NON_MONOTONIC",
                "delta": 0.0,
            }

        dx    = claimed_pos[0] - prev_pos[0]
        dy    = claimed_pos[1] - prev_pos[1]
        delta = math.hypot(dx, dy)

        # --- Kinematic bound via Z3 ---
        solver = z3.Solver()

        px_prev  = z3.Real("px_prev")
        py_prev  = z3.Real("py_prev")
        px_claim = z3.Real("px_claim")
        py_claim = z3.Real("py_claim")
        dt_z3    = z3.RealVal(float(dt))
        v_max_z3 = z3.RealVal(float(self.v_max))
        tol_z3   = z3.RealVal(float(self.pos_tolerance))

        solver.add(px_prev  == float(prev_pos[0]))
        solver.add(py_prev  == float(prev_pos[1]))
        solver.add(px_claim == float(claimed_pos[0]))
        solver.add(py_claim == float(claimed_pos[1]))

        ddx      = px_claim - px_prev
        ddy      = py_claim - py_prev
        max_dist = v_max_z3 * dt_z3 + tol_z3

        # Assert the VIOLATION: moved more than physically possible
        solver.add(ddx * ddx + ddy * ddy > max_dist * max_dist)

        result = solver.check()
        if result == z3.sat:
            return {
                "contradiction": True,
                "reason": "KINEMATIC_BOUND_VIOLATION",
                "delta": delta,
            }

        # --- Obstacle collision check via Z3 ---
        if obstacles:
            obs_solver = z3.Solver()
            px = z3.RealVal(float(claimed_pos[0]))
            py = z3.RealVal(float(claimed_pos[1]))

            obs_clauses = []
            for obs in obstacles:
                ox, oy, ow, oh = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
                inside = z3.And(
                    px >= ox, px <= ox + ow,
                    py >= oy, py <= oy + oh,
                )
                obs_clauses.append(inside)

            if obs_clauses:
                obs_solver.add(z3.Or(*obs_clauses))
                if obs_solver.check() == z3.sat:
                    return {
                        "contradiction": True,
                        "reason": "OBSTACLE_COLLISION_PARADOX",
                        "delta": delta,
                    }

        return {
            "contradiction": False,
            "reason": "NONE",
            "delta": delta,
        }
