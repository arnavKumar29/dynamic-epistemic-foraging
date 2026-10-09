"""
theory_of_mind.py — Bayesian Theory of Mind (BToM) module.

Implements Baker et al.'s rational agent model:
  "Bayesian Theory of Mind: Modeling Joint Belief-Desire Attribution"

For each incoming peer position claim, this module:
  1. Runs Z3 SMT kinematic check (is the claim physically reachable?).
  2. Runs Boltzmann-rational intent inference: which goal best explains
     the observed trajectory (forage, explore, return_to_base)?
  3. Combines: if kinematic clash AND no goal can explain the move
     (max posterior < 0.15), this is deception, not confusion.

Returns a structured verdict with: contradiction flag, anomaly delta,
explanation string, and per-goal posterior distribution.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from src.reasoning.belief import Belief, KnowledgeBase
from src.reasoning.smt_checker import SMTContradictionChecker


class BayesianToM:
    """
    Kinematic + intent inference Theory of Mind.

    Args:
        max_velocity   : Agent's rated max velocity (pixels/tick).
        pos_tolerance  : SMT slack for kinematic bound.
        obstacles      : List of obstacle rects [(x,y,w,h)] for SMT obstacle check.
        rationality    : β in Boltzmann-rational model. Higher = more rational.
                         Typical: 1.5 – 3.0.
        deception_threshold : If max intent posterior < this AND kinematic clash,
                              classify as deception (not just error).
    """

    def __init__(
        self,
        max_velocity: float,
        pos_tolerance: float = 18.0,
        obstacles: Optional[List] = None,
        rationality: float = 2.0,
        deception_threshold: float = 0.15,
    ):
        self.max_velocity         = max_velocity
        self.obstacles            = obstacles or []
        self.rationality          = rationality
        self.deception_threshold  = deception_threshold
        self._smt = SMTContradictionChecker(
            pos_tolerance=pos_tolerance, v_max=max_velocity
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def verify(
        self,
        peer_id: str,
        claimed_pos: Tuple[float, float],
        claimed_tick: int,
        kb: KnowledgeBase,
        goals: Dict[str, Tuple[float, float]],
    ) -> Tuple[bool, float, str, Dict[str, float]]:
        """
        Verify a peer's claimed position against kinematics and intent.

        Returns:
            (is_contradiction, anomaly_delta, explanation, intent_posteriors)
        """
        history = kb.history(peer_id, "position", n=2)

        if len(history) < 1:
            uniform = {g: 1.0 / len(goals) for g in goals} if goals else {}
            return False, 0.0, "First contact — no baseline history.", uniform

        last: Belief = history[-1]

        # 1. SMT kinematic + obstacle check
        smt_result = self._smt.verify_kinematic_claim(
            prev_pos=last.value,
            prev_tick=last.timestamp,
            claimed_pos=claimed_pos,
            claimed_tick=claimed_tick,
            obstacles=self.obstacles,
        )

        delta = smt_result["delta"]

        # 2. Temporal paradox fast-path
        if smt_result["reason"] == "TEMPORAL_PARADOX_NON_MONOTONIC":
            return (
                True,
                999.0,
                f"Temporal clash: tick regression dt={claimed_tick - last.timestamp} (replay attack?)",
                {},
            )

        # 3. Intent inference (Boltzmann-rational)
        intent_post = self._infer_intent(
            prev_pos=last.value,
            claimed_pos=claimed_pos,
            goals=goals,
            dt=max(1, claimed_tick - last.timestamp),
        )

        # 4. Final verdict
        kinematic_clash = smt_result["contradiction"]

        if kinematic_clash:
            best_intent     = max(intent_post.values(), default=0.0)
            best_goal       = max(intent_post, key=intent_post.get, default="unknown")
            if best_intent < self.deception_threshold:
                expl = (
                    f"DECEPTION: Velocity violation Δ={delta:.1f}px "
                    f"({smt_result['reason']}) AND no intent explains it "
                    f"(best_intent={best_intent:.2f} for '{best_goal}'). "
                    "Posteriors: " + str({k: f"{v:.2f}" for k, v in intent_post.items()})
                )
            else:
                expl = (
                    f"Velocity violation Δ={delta:.1f}px ({smt_result['reason']}). "
                    f"Possible intent: '{best_goal}' (p={best_intent:.2f})."
                )
            return True, delta, expl, intent_post

        # 5. Obstacle paradox (separate from kinematic)
        if smt_result["reason"] == "OBSTACLE_COLLISION_PARADOX":
            return (
                True,
                delta,
                f"Spatial paradox: claimed position {claimed_pos} is inside a known obstacle.",
                intent_post,
            )

        best_goal = max(intent_post, key=intent_post.get, default="unknown")
        expl = (
            f"Claim verified. Intent: '{best_goal}' "
            f"(p={intent_post.get(best_goal, 0.0):.2f}). Δ={delta:.1f}px."
        )
        return False, delta, expl, intent_post

    # ------------------------------------------------------------------ #
    # Private: Boltzmann-rational intent inference                        #
    # ------------------------------------------------------------------ #
    def _infer_intent(
        self,
        prev_pos: Tuple[float, float],
        claimed_pos: Tuple[float, float],
        goals: Dict[str, Tuple[float, float]],
        dt: int,
    ) -> Dict[str, float]:
        """
        P(goal | trajectory) ∝ exp(-β * [angular_error + 0.3 * speed_error])

        angular_error : angle between actual move vector and toward-goal vector.
        speed_error   : how far the move speed deviates from max_velocity.
        """
        if not goals:
            return {}

        move_x = claimed_pos[0] - prev_pos[0]
        move_y = claimed_pos[1] - prev_pos[1]
        move_dist = math.hypot(move_x, move_y)

        likelihoods = {}
        for goal_name, (gx, gy) in goals.items():
            to_goal_x = gx - prev_pos[0]
            to_goal_y = gy - prev_pos[1]
            to_goal_d = math.hypot(to_goal_x, to_goal_y)

            if move_dist < 1e-6 or to_goal_d < 1e-6:
                ang_err = 0.0
            else:
                dot     = move_x * to_goal_x + move_y * to_goal_y
                cos_ang = max(-1.0, min(1.0, dot / (move_dist * to_goal_d)))
                ang_err = math.acos(cos_ang)  # 0 = heading straight for goal

            max_possible = self.max_velocity * dt
            speed_err    = max(0.0, move_dist - max_possible) / max(max_possible, 1e-6)

            likelihoods[goal_name] = math.exp(
                -self.rationality * (ang_err + 0.3 * speed_err)
            )

        total = sum(likelihoods.values()) or 1.0
        return {k: v / total for k, v in likelihoods.items()}
