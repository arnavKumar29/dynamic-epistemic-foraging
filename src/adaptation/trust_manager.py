"""
trust_manager.py — Beta-Binomial Dirichlet Trust Engine with asymmetric recovery.

Theory
------
Implements Jøsang's Beta Reputation System (2002) with three key extensions:

1. ASYMMETRIC UPDATE  (Jøsang & Ismail 2002; Mui et al. 2002)
   Punishment is fast and heavy; recovery is slow and throttled.
   In real trust systems, rebuilding trust after deception takes far longer
   than losing it.

2. GRUDGE MEMORY
   Tracks the minimum trust ever observed for each peer (one-way ratchet).
   Once trust drops below 0.3 ("caught lying"), recovery is permanently
   throttled by a grudge factor.  This prevents the yo-yo effect where
   trust bounces back immediately after an attack window ends.

3. TEMPORAL DECAY (Aging)
   Old evidence decays each tick via multiplicative aging so distant past
   honesty doesn't fully offset recent attacks.

Trust model
-----------
   E[trust_j] = α_j / (α_j + β_j)              [Beta distribution mean]
   Var[trust_j] = α·β / ((α+β)²·(α+β+1))       [epistemic uncertainty]

Grudge recovery tiers
---------------------
   min_trust_ever ≥ 0.60  →  NORMAL   (rate × 1.00)
   min_trust_ever ≥ 0.40  →  MODERATE (rate × 0.40)
   min_trust_ever ≥ 0.20  →  SLOW     (rate × 0.12)
   min_trust_ever < 0.20  →  FROZEN   (rate × 0.03)  — nearly frozen

All internal state is accessible via full_state() for the Streamlit
Beta Distribution Inspector tab.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


class BetaTrustEngine:
    """
    Beta-Binomial trust manager with asymmetric recovery and grudge memory.

    Parameters
    ----------
    alpha0  : Initial α (default 4.0 → starting trust ~0.80).
    beta0   : Initial β (default 1.0).
    aging   : Temporal decay factor λ per update (0.97 recommended).
    """

    ALPHA_MAX = 60.0
    BETA_MAX  = 60.0
    ALPHA_MIN = 0.10
    BETA_MIN  = 0.10

    # Grudge recovery tiers: (min_trust_threshold, recovery_multiplier, tier_name)
    GRUDGE_TIERS: List[Tuple[float, float, str]] = [
        (0.60, 1.00, "NORMAL"),    # agent was never badly distrusted
        (0.40, 0.40, "MODERATE"),  # was somewhat distrusted at worst
        (0.20, 0.12, "SLOW"),      # was significantly distrusted
        (0.00, 0.03, "FROZEN"),    # severely distrusted — grudge nearly permanent
    ]

    def __init__(
        self,
        alpha0: float = 4.0,
        beta0:  float = 1.0,
        aging:  float = 0.97,
    ) -> None:
        self.alpha0 = alpha0
        self.beta0  = beta0
        self.aging  = aging
        # peer_id → (alpha, beta)
        self._params:    Dict[str, Tuple[float, float]] = {}
        # peer_id → minimum trust ever observed (one-way grudge ratchet)
        self._min_trust: Dict[str, float]               = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def get(self, peer_id: str) -> Tuple[float, float]:
        """Return (trust, uncertainty) for peer."""
        a, b  = self._ensure(peer_id)
        trust       = a / (a + b)
        uncertainty = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return trust, uncertainty

    def get_trust(self, peer_id: str) -> float:
        return self.get(peer_id)[0]

    def update(
        self,
        peer_id:       str,
        p_attack:      float,
        anomaly_delta: float,
        verdict_label: str = "HONEST",
    ) -> dict:
        """
        Apply one verdict to the Beta trust model for this peer.

        Asymmetric update logic
        -----------------------
        ATTACK verdict  → heavy punishment: quadratic severity scaling.
        FAULT  verdict  → moderate punishment (unreliable but not malicious).
        HONEST verdict  → gradual recovery, throttled by grudge rate.

        Returns a rich dict with old/new trust, deltas, Beta parameters,
        aging applied, grudge tier, and recovery throttle flag.
        """
        a, b = self._ensure(peer_id)

        # Step 1: Apply temporal decay (aging)
        a_before_aging, b_before_aging = a, b
        a = 1.0 + (a - 1.0) * self.aging
        b = 1.0 + (b - 1.0) * self.aging
        aging_applied = (abs(a - a_before_aging) + abs(b - b_before_aging)) > 1e-6

        # Step 2: Compute severity and evidence weight
        s = min(anomaly_delta / 10.0, 1.0)   # severity ∈ [0, 1]
        w = 1.0 + s                            # evidence weight ∈ [1, 2]

        old_trust = a / (a + b)

        # Step 3: Apply verdict
        grudge_rate         = self._grudge_rate(peer_id, old_trust)
        grudge_tier         = self._grudge_tier_name(peer_id, old_trust)
        recovery_throttled  = False

        if verdict_label == "ATTACK":
            # Fast, heavy punishment — quadratic severity scaling
            b += p_attack * w * (1.5 + s * s)
        elif verdict_label == "FAULT":
            # Moderate punishment (unreliable hardware, not malicious intent)
            b += p_attack * w * 0.6
        else:
            # HONEST: recover only if very confident in honesty
            if p_attack < 0.08:
                a += (1.0 - p_attack) * w * grudge_rate
                recovery_throttled = (grudge_rate < 1.0)
            # else: no recovery this tick — insufficient evidence of honesty

        # Step 4: Clamp to valid range
        a = max(self.ALPHA_MIN, min(self.ALPHA_MAX, a))
        b = max(self.BETA_MIN,  min(self.BETA_MAX,  b))

        new_trust   = a / (a + b)
        uncertainty = (a * b) / ((a + b) ** 2 * (a + b + 1))

        # Step 5: Update grudge memory (one-way ratchet — only goes down)
        prev_min = self._min_trust.get(peer_id, new_trust)
        self._min_trust[peer_id] = min(prev_min, new_trust)

        self._params[peer_id] = (a, b)

        return {
            # Core trust values
            "peer_id":      peer_id,
            "old_trust":    round(old_trust, 5),
            "new_trust":    round(new_trust, 5),
            "delta":        round(new_trust - old_trust, 5),
            "uncertainty":  round(uncertainty, 5),
            # Beta distribution parameters
            "alpha":        round(a, 4),
            "beta":         round(b, 4),
            # Verdict context
            "verdict":      verdict_label,
            "p_attack":     round(p_attack, 5),
            "anomaly_delta": round(anomaly_delta, 3),
            "severity":     round(s, 3),
            # Grudge state
            "grudge_rate":  round(grudge_rate, 3),
            "grudge_tier":  grudge_tier,
            "min_trust_ever": round(self._min_trust.get(peer_id, new_trust), 5),
            # Update mechanics
            "aging_applied":        aging_applied,
            "recovery_throttled":   recovery_throttled,
        }

    def all_trusts(self) -> Dict[str, float]:
        """Return {peer_id: trust_score} for all tracked peers."""
        return {pid: self.get_trust(pid) for pid in self._params}

    def full_state(self) -> Dict[str, dict]:
        """
        Return complete Beta distribution state for all tracked peers.
        Used by the Streamlit Beta Distribution Inspector tab to render
        the PDF of each peer's trust belief.
        """
        result = {}
        for pid, (a, b) in self._params.items():
            trust       = a / (a + b)
            uncertainty = (a * b) / ((a + b) ** 2 * (a + b + 1))
            min_t       = self._min_trust.get(pid, trust)
            rate        = self._grudge_rate(pid, trust)
            tier        = self._grudge_tier_name(pid, trust)
            result[pid] = {
                "alpha":         round(a, 4),
                "beta":          round(b, 4),
                "trust":         round(trust, 5),
                "uncertainty":   round(uncertainty, 5),
                "min_trust_ever": round(min_t, 5),
                "grudge_rate":   round(rate, 3),
                "grudge_tier":   tier,
                # Mode of Beta distribution (for visualization)
                "mode":          round((a - 1) / (a + b - 2), 5) if (a + b) > 2 else 0.5,
                # 95% credible interval (approximate via Wilson)
                "ci_lower":      round(max(0.0, trust - 1.96 * math.sqrt(uncertainty)), 5),
                "ci_upper":      round(min(1.0, trust + 1.96 * math.sqrt(uncertainty)), 5),
            }
        return result

    def reset_peer(self, peer_id: str) -> None:
        """Reset trust for a peer (e.g., after identity refresh). Does NOT reset grudge."""
        self._params.pop(peer_id, None)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #
    def _ensure(self, peer_id: str) -> Tuple[float, float]:
        if peer_id not in self._params:
            self._params[peer_id] = (self.alpha0, self.beta0)
        return self._params[peer_id]

    def _grudge_rate(self, peer_id: str, current_trust: float) -> float:
        """Recovery multiplier based on worst trust ever seen (one-way ratchet)."""
        min_seen = self._min_trust.get(peer_id, current_trust)
        for threshold, multiplier, _ in self.GRUDGE_TIERS:
            if min_seen >= threshold:
                return multiplier
        return self.GRUDGE_TIERS[-1][1]

    def _grudge_tier_name(self, peer_id: str, current_trust: float) -> str:
        min_seen = self._min_trust.get(peer_id, current_trust)
        for threshold, _, name in self.GRUDGE_TIERS:
            if min_seen >= threshold:
                return name
        return self.GRUDGE_TIERS[-1][2]
