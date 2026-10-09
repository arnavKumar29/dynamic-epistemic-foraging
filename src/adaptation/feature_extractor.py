"""
feature_extractor.py — Feature vector extraction from epistemic events.

Converts continuous anomaly measurements from the ToM module into
discretized feature vectors suitable for the Dynamic Bayesian Network.

Per-peer running statistics track:
  - Rolling delta history (last WINDOW observations) for variance.
  - Paradox streak: consecutive anomalous observations.
  - Estimated packet-loss rate from paradox frequency.

All raw statistics are accessible via get_rolling_stats() for the
Streamlit research dashboard.
"""
from __future__ import annotations

import math
from typing import Dict, List


class FeatureExtractor:
    """
    Maintains rolling per-peer statistics and produces discrete feature vectors.

    Feature vector schema
    ---------------------
    {
      "KinematicAnomaly": int,   # 0=Low (<15 px), 1=Medium (15–80 px), 2=High (>80 px)
      "Variance":         int,   # 0=Low (<200 px²), 1=High
      "PacketLoss":       int,   # 0=Low, 1=High  — estimated from paradox frequency
    }
    """

    WINDOW          = 12    # rolling window size for variance / streak
    HIGH_ANOMALY_PX = 80.0  # delta threshold for KA=2
    MED_ANOMALY_PX  = 15.0  # delta threshold for KA=1
    HIGH_VAR_PX2    = 200.0 # variance threshold
    HIGH_LOSS_RATE  = 0.30  # paradox-frequency threshold for PacketLoss=1

    def __init__(self) -> None:
        self._history: Dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #
    def extract(self, peer_id: str, event: dict) -> dict:
        """
        Given a peer id and an EpistemicEvent dict, return a discrete
        feature vector and update rolling statistics.

        Args:
            peer_id : Peer agent identifier, e.g. "A2".
            event   : EpistemicEvent dict from BDIEngine.update_beliefs().

        Returns:
            {"KinematicAnomaly": int, "Variance": int, "PacketLoss": int}
        """
        rec = self._ensure(peer_id)

        delta = float(event.get("delta", 0.0))
        is_contradiction = bool(event.get("contradiction", False))

        # Accumulate history
        rec["deltas"].append(delta)
        if is_contradiction:
            rec["paradox_count"] += 1
            rec["streak"] += 1
        else:
            rec["streak"] = 0      # reset streak on clean observation

        rec["total_observations"] += 1

        # Keep only last WINDOW entries for variance
        if len(rec["deltas"]) > self.WINDOW:
            rec["deltas"] = rec["deltas"][-self.WINDOW:]

        variance = self._variance(rec["deltas"])

        # ── Discretize ──────────────────────────────────────────────── #
        if delta >= self.HIGH_ANOMALY_PX:
            ka = 2
        elif delta >= self.MED_ANOMALY_PX:
            ka = 1
        else:
            ka = 0

        var_disc = 1 if variance > self.HIGH_VAR_PX2 else 0

        # PacketLoss estimated from recent paradox frequency in rolling window
        n_obs   = max(len(rec["deltas"]), 1)
        n_paradox_in_window = sum(1 for d in rec["deltas"] if d >= self.MED_ANOMALY_PX)
        paradox_rate = n_paradox_in_window / n_obs
        packet_loss = 1 if paradox_rate >= self.HIGH_LOSS_RATE else 0

        return {
            "KinematicAnomaly": ka,
            "Variance":         var_disc,
            "PacketLoss":       packet_loss,
        }

    def paradox_count(self, peer_id: str) -> int:
        """Total paradox count for this peer across all observations."""
        return self._history.get(peer_id, {}).get("paradox_count", 0)

    def current_streak(self, peer_id: str) -> int:
        """Current consecutive-anomaly streak length."""
        return self._history.get(peer_id, {}).get("streak", 0)

    def recent_deltas(self, peer_id: str) -> List[float]:
        return list(self._history.get(peer_id, {}).get("deltas", []))

    def get_rolling_stats(self, peer_id: str) -> dict:
        """
        Return all rolling statistics for the research dashboard.

        Returns
        -------
        {
          "deltas":          List[float],   raw delta history (last WINDOW)
          "variance":        float,         sample variance of deltas
          "mean_delta":      float,
          "paradox_count":   int,           total contradictions ever
          "streak":          int,           current consecutive anomaly streak
          "total_obs":       int,           total ticks observed
          "paradox_rate":    float,         paradox fraction in rolling window
          "ka_level":        int,           last KA discretization
        }
        """
        rec    = self._history.get(peer_id, {})
        deltas = rec.get("deltas", [])
        n      = max(len(deltas), 1)
        n_paradox_in_window = sum(1 for d in deltas if d >= self.MED_ANOMALY_PX)

        last_delta = deltas[-1] if deltas else 0.0
        if last_delta >= self.HIGH_ANOMALY_PX:
            ka = 2
        elif last_delta >= self.MED_ANOMALY_PX:
            ka = 1
        else:
            ka = 0

        return {
            "deltas":        list(deltas),
            "variance":      self._variance(deltas),
            "mean_delta":    sum(deltas) / n,
            "paradox_count": rec.get("paradox_count", 0),
            "streak":        rec.get("streak", 0),
            "total_obs":     rec.get("total_observations", 0),
            "paradox_rate":  n_paradox_in_window / n,
            "ka_level":      ka,
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #
    def _ensure(self, peer_id: str) -> dict:
        if peer_id not in self._history:
            self._history[peer_id] = {
                "deltas":             [],
                "paradox_count":      0,
                "streak":             0,
                "total_observations": 0,
            }
        return self._history[peer_id]

    @staticmethod
    def _variance(data: List[float]) -> float:
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        return sum((x - mean) ** 2 for x in data) / len(data)
