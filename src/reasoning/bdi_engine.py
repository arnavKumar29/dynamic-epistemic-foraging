"""
bdi_engine.py — BDI (Belief-Desire-Intention) cognitive engine.

Wires together:
  - KnowledgeBase : symbolic belief store with AGM revision.
  - BayesianToM   : kinematic + intent verification for incoming claims.

On each tick, the BDI engine:
  1. Ingests a list of raw decoded messages from the network layer.
  2. Translates each message into symbolic Belief atoms.
  3. Runs BayesianToM on position claims to check for epistemic clashes.
  4. Returns a list of EpistemicEvent dicts (one per processed message).

EpistemicEvent schema:
  {
    "peer_id"       : str,          # e.g. "A2"
    "contradiction" : bool,
    "delta"         : float,        # anomaly magnitude (pixels)
    "reason"        : str,          # SMT reason code
    "explanation"   : str,          # human-readable
    "intent_post"   : dict,         # per-goal Bayesian posteriors
    "noise_high"    : bool          # heuristic: network noise flag
  }
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.reasoning.belief import Belief, KnowledgeBase
from src.reasoning.theory_of_mind import BayesianToM


class BDIEngine:
    """
    BDI cognitive engine for one agent.

    Args:
        agent_id    : String identifier, e.g. "A0".
        obstacles   : List of obstacle rects [(x,y,w,h)] for ToM SMT.
        max_velocity: For ToM kinematic threshold.
        pos_tolerance: SMT slack (pixels).
        goals       : Dict of named goals {name: (x,y)}.
    """

    def __init__(
        self,
        agent_id: str,
        obstacles: List[Tuple[float, float, float, float]],
        max_velocity: float = 3.0,
        pos_tolerance: float = 18.0,
        goals: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.agent_id = agent_id
        self.goals    = goals or {}
        self.kb       = KnowledgeBase()

        self.tom = BayesianToM(
            max_velocity=max_velocity,
            pos_tolerance=pos_tolerance,
            obstacles=obstacles,
        )

        # BDI state
        self.desires: List[str]        = ["EXPLORE", "FORAGE", "RETURN"]
        self.active_desire: str        = "FORAGE"
        self.current_intention: Optional[str] = "EXECUTE_PATH"

    # ------------------------------------------------------------------ #
    # Core belief update loop                                             #
    # ------------------------------------------------------------------ #
    def update_beliefs(self, incoming_messages: List[Dict[str, Any]]) -> List[Dict]:
        """
        Process all incoming decoded messages from the network layer.
        Returns list of EpistemicEvent dicts.
        """
        events = []

        for msg in incoming_messages:
            # --- Cryptographic failure path ---
            if msg.get("CORRUPTED"):
                sender   = f"A{msg.get('sender_id', '?')}"
                event    = {
                    "peer_id":       sender,
                    "contradiction": True,
                    "delta":         999.0,
                    "reason":        "CRYPTO_AUTHENTICATION_FAILURE",
                    "explanation":   f"AES-GCM tag mismatch from {sender}: {msg.get('error', '')}",
                    "intent_post":   {},
                    "noise_high":    False,
                }
                events.append(event)
                continue

            # --- Normal message: translate to beliefs ---
            sender_id_raw = msg.get("sender_id", msg.get("id", -1))
            sender        = f"A{sender_id_raw}"
            tick          = int(msg.get("tick", 0))
            x             = float(msg.get("x", 0.0))
            y             = float(msg.get("y", 0.0))
            heading       = float(msg.get("heading", 0.0))
            status        = str(msg.get("status", "UNKNOWN"))

            # Build position belief
            pos_belief = Belief(
                subject=sender,
                predicate="position",
                value=(x, y),
                confidence=0.95,
                timestamp=tick,
                source="COMM",
            )

            # Heading belief
            hdg_belief = Belief(
                subject=sender,
                predicate="heading",
                value=heading,
                confidence=0.90,
                timestamp=tick,
                source="COMM",
            )

            # Status belief
            sts_belief = Belief(
                subject=sender,
                predicate="status",
                value=status,
                confidence=0.95,
                timestamp=tick,
                source="COMM",
            )

            # ToM verification (only on position claims)
            is_clash, delta, explanation, intent_post = self.tom.verify(
                peer_id=sender,
                claimed_pos=(x, y),
                claimed_tick=tick,
                kb=self.kb,
                goals=self.goals,
            )

            # Heuristic noise flag: low delta on noisy periods
            # (in a real system this would come from link-layer metrics)
            noise_high = delta > 0.0 and delta < 12.0  # small jitter = noise

            event = {
                "peer_id":       sender,
                "contradiction": is_clash,
                "delta":         delta,
                "reason":        "CONTRADICTION" if is_clash else "NONE",
                "explanation":   explanation,
                "intent_post":   intent_post,
                "noise_high":    noise_high,
            }

            # Commit to KB only if no clash (don't poison beliefs with
            # forged data; keep last-known good position for future ToM)
            if not is_clash:
                self.kb.add(pos_belief)
                self.kb.add(hdg_belief)
                self.kb.add(sts_belief)

            events.append(event)

        return events

    # ------------------------------------------------------------------ #
    # Deliberation                                                        #
    # ------------------------------------------------------------------ #
    def deliberate(self) -> str:
        """Simple reactive desire → intention selection."""
        # Future extension: factor in trust scores, battery, resource proximity
        return self.current_intention or "EXECUTE_PATH"

    def inject_self_belief(self, x: float, y: float, heading: float, tick: int) -> None:
        """Register own ground-truth position into KB (sensor input)."""
        self.kb.add(Belief(
            subject=self.agent_id,
            predicate="position",
            value=(x, y),
            confidence=1.0,
            timestamp=tick,
            source="SENSOR",
        ))
        self.kb.add(Belief(
            subject=self.agent_id,
            predicate="heading",
            value=heading,
            confidence=1.0,
            timestamp=tick,
            source="SENSOR",
        ))
