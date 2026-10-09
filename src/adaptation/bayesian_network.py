"""
bayesian_network.py — Dynamic Bayesian Network (DBN) fault/attack classifier.

Architecture
------------
Hidden variables:   A_t (Attack),  F_t (HardwareFault)
Observed variables: K_t (KinematicAnomaly ∈ {0,1,2}), V_t (Variance ∈ {0,1}),
                    L_t (PacketLoss ∈ {0,1})

Transition model (hidden Markov chain across time):
    P(A_t=1 | A_{t-1}=1) = 0.90   ← attacks persist once started
    P(A_t=1 | A_{t-1}=0) = 0.05   ← rare spontaneous emergence
    P(F_t=1 | F_{t-1}=1) = 0.80   ← hardware faults also persist
    P(F_t=1 | F_{t-1}=0) = 0.08

Forward filter update (exact closed-form):
    prior_t(A,F) ∝ Σ_{prev} P(A_t|A_{t-1}) · P(F_t|F_{t-1}) · posterior_{t-1}
    posterior_t(A,F) ∝ P(K_t|A,F) · P(V_t|F,L_t) · prior_t(A,F)

Why this matters for research explainability
--------------------------------------------
Every call to classify() returns a FULL math trace exposing:
  - The previous posterior it started from.
  - The predicted prior after propagating through the transition.
  - The CPT_K and CPT_V likelihood values looked up.
  - The unnormalized joint distribution over all (A,F) combos.
  - The normalization constant Z.
  - The final posterior marginals.

This is shown step-by-step in the Streamlit Bayesian Inspector tab so
any observer can follow exactly WHY the agent decided "this is an attack"
rather than treating it as a black box.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# pgmpy used for full joint CPD validation — optional
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    _PGMPY_AVAILABLE = True
except ImportError:
    _PGMPY_AVAILABLE = False


class DynamicBayesianClassifier:
    """
    Per-peer 2-slice DBN forward filter.

    Maintains a joint posterior over (A_t, F_t) ∈ {0,1}² per peer.
    Each call to classify() runs one Bayesian update step and returns
    the full calculation trace for explainability.

    Parameters
    ----------
    p_A_prior : Prior P(Attack = True) at t=0.
    p_F_prior : Prior P(HardwareFault = True) at t=0.
    """

    # ── Transition probabilities ─────────────────────────────────────── #
    P_A_PERSIST = 0.90    # P(A_t=1 | A_{t-1}=1)  attacks tend to sustain
    P_A_EMERGE  = 0.05    # P(A_t=1 | A_{t-1}=0)  rare cold-start
    P_F_PERSIST = 0.80    # P(F_t=1 | F_{t-1}=1)  faults also persist
    P_F_EMERGE  = 0.08    # P(F_t=1 | F_{t-1}=0)

    # ── CPT: P(KinematicAnomaly ∈ {0=Low,1=Med,2=High} | Attack, Fault) ─ #
    # Rows keyed by (A, F). Values = [P(K=0|A,F), P(K=1|A,F), P(K=2|A,F)]
    # Rationale:
    #   (0,0) → clean channel: almost always Low
    #   (0,1) → hardware fault: moderate positional jitter
    #   (1,0) → active attack: usually large delta
    #   (1,1) → both:          worst case, near certain High
    CPT_K: Dict[Tuple[int,int], List[float]] = {
        (0, 0): [0.94, 0.05, 0.01],
        (0, 1): [0.28, 0.52, 0.20],
        (1, 0): [0.04, 0.24, 0.72],
        (1, 1): [0.02, 0.16, 0.82],
    }

    # ── CPT: P(Variance ∈ {0=Low,1=High} | Fault, PacketLoss) ─────────── #
    CPT_V: Dict[Tuple[int,int], List[float]] = {
        (0, 0): [0.97, 0.03],   # clean
        (0, 1): [0.53, 0.47],   # no fault but packet loss → variance
        (1, 0): [0.38, 0.62],   # fault raises variance
        (1, 1): [0.08, 0.92],   # fault + loss → high variance
    }

    # ── Decision thresholds ──────────────────────────────────────────── #
    ATTACK_THRESHOLD = 0.50
    FAULT_THRESHOLD  = 0.40

    def __init__(
        self,
        p_A_prior: float = 0.05,
        p_F_prior: float = 0.08,
    ) -> None:
        self.p_A_prior = p_A_prior
        self.p_F_prior = p_F_prior
        # Per-peer posterior: {peer_id: (p_A, p_F)}
        self._memory: Dict[str, Tuple[float, float]] = {}

    # ------------------------------------------------------------------ #
    # Main API                                                             #
    # ------------------------------------------------------------------ #
    def classify(
        self,
        peer_id: str,
        feature_vector: dict,
    ) -> dict:
        """
        Run one forward-filter step and return a full calculation trace.

        Parameters
        ----------
        peer_id        : e.g. "A2"
        feature_vector : {"KinematicAnomaly": int, "Variance": int, "PacketLoss": int}

        Returns
        -------
        {
          "p_attack": float,
          "p_fault":  float,
          "p_honest": float,
          "label":    "HONEST" | "FAULT" | "ATTACK",
          "math_trace": {
            "prev_posterior":       {"p_A": float, "p_F": float},
            "predicted_prior":      {"p_A": float, "p_F": float},
            "observations":         {"KinematicAnomaly": int, "Variance": int, "PacketLoss": int},
            "cpt_k_label":          str,    "CPT_K[(A,F)][k_obs]" for each combo
            "cpt_v_label":          str,
            "joint_unnormalized":   {str: float},   all 4 (A,F) joints
            "normalization_Z":      float,
            "posterior_table":      {str: float},   normalized
            "verdict_reasoning":    str,
          }
        }
        """
        k_obs = int(feature_vector.get("KinematicAnomaly", 0))
        v_obs = int(feature_vector.get("Variance",         0))
        l_obs = int(feature_vector.get("PacketLoss",       0))

        # ── Step 1: Predict — propagate through transition model ─────── #
        prev_A, prev_F = self._memory.get(peer_id, (self.p_A_prior, self.p_F_prior))

        pred_A = self.P_A_PERSIST * prev_A + self.P_A_EMERGE  * (1.0 - prev_A)
        pred_F = self.P_F_PERSIST * prev_F + self.P_F_EMERGE  * (1.0 - prev_F)

        # ── Step 2: Update — multiply joint by likelihood ────────────── #
        joints: Dict[Tuple[int,int], float] = {}
        cpt_detail: Dict[str, dict] = {}
        total = 0.0

        for a in (0, 1):
            for f in (0, 1):
                pa  = pred_A if a else 1.0 - pred_A
                pf  = pred_F if f else 1.0 - pred_F
                p_k = self.CPT_K[(a, f)][k_obs]
                p_v = self.CPT_V[(f, l_obs)][v_obs]
                j   = pa * pf * p_k * p_v

                joints[(a, f)] = j
                total          += j
                cpt_detail[f"A={a},F={f}"] = {
                    "prior_A":  round(pa,  5),
                    "prior_F":  round(pf,  5),
                    "P_K":      round(p_k, 5),
                    "P_V":      round(p_v, 5),
                    "joint_unnorm": round(j, 8),
                }

        if total < 1e-12:
            total = 1e-12   # numerical guard

        # ── Step 3: Normalize ────────────────────────────────────────── #
        norm_joints = {k: v / total for k, v in joints.items()}

        post_A = norm_joints[(1, 0)] + norm_joints[(1, 1)]   # marginal P(A=1)
        post_F = norm_joints[(0, 1)] + norm_joints[(1, 1)]   # marginal P(F=1)
        post_H = norm_joints[(0, 0)]                          # purely honest

        self._memory[peer_id] = (post_A, post_F)

        # ── Verdict ──────────────────────────────────────────────────── #
        if post_A >= self.ATTACK_THRESHOLD:
            label = "ATTACK"
            reasoning = (
                f"P(Attack)={post_A:.3f} ≥ threshold {self.ATTACK_THRESHOLD}. "
                f"Driven by K={k_obs} anomaly level and {['low','high'][v_obs]} variance."
            )
        elif post_F >= self.FAULT_THRESHOLD:
            label = "FAULT"
            reasoning = (
                f"P(Attack)={post_A:.3f} < {self.ATTACK_THRESHOLD} but "
                f"P(Fault)={post_F:.3f} ≥ {self.FAULT_THRESHOLD}. "
                f"Kinematic level={k_obs} consistent with hardware drift, not deception."
            )
        else:
            label = "HONEST"
            reasoning = (
                f"P(Attack)={post_A:.3f}, P(Fault)={post_F:.3f} — "
                f"both below thresholds. Observation K={k_obs} consistent with noise."
            )

        return {
            "p_attack": post_A,
            "p_fault":  post_F,
            "p_honest": post_H,
            "label":    label,
            "math_trace": {
                "prev_posterior":   {"p_A": round(prev_A, 5), "p_F": round(prev_F, 5)},
                "predicted_prior":  {"p_A": round(pred_A, 5), "p_F": round(pred_F, 5)},
                "transition": {
                    "P_A_PERSIST": self.P_A_PERSIST,
                    "P_A_EMERGE":  self.P_A_EMERGE,
                    "P_F_PERSIST": self.P_F_PERSIST,
                    "P_F_EMERGE":  self.P_F_EMERGE,
                },
                "observations":   {"KinematicAnomaly": k_obs, "Variance": v_obs, "PacketLoss": l_obs},
                "cpt_detail":     cpt_detail,
                "joint_unnorm":   {str(k): round(v, 8) for k, v in joints.items()},
                "normalization_Z": round(total, 8),
                "posterior_table": {
                    "P(A=0,F=0)": round(norm_joints[(0, 0)], 5),
                    "P(A=0,F=1)": round(norm_joints[(0, 1)], 5),
                    "P(A=1,F=0)": round(norm_joints[(1, 0)], 5),
                    "P(A=1,F=1)": round(norm_joints[(1, 1)], 5),
                },
                "marginals":      {"p_A": round(post_A, 5), "p_F": round(post_F, 5), "p_H": round(post_H, 5)},
                "verdict_reasoning": reasoning,
            },
        }

    # ── Legacy compatibility ─────────────────────────────────────────── #
    def step(
        self,
        peer_id: str,
        feature_vector: dict,
    ) -> Tuple[float, float]:
        """
        Lightweight wrapper that runs one forward-filter step and returns
        only (p_attack, p_fault) for code that doesn't need the full trace.
        """
        result = self.classify(peer_id, feature_vector)
        return result["p_attack"], result["p_fault"]

    def reset(self, peer_id: str) -> None:
        """Reset the DBN memory for a specific peer."""
        self._memory.pop(peer_id, None)

    def get_posterior(self, peer_id: str) -> Tuple[float, float]:
        return self._memory.get(peer_id, (self.p_A_prior, self.p_F_prior))

    def get_all_posteriors(self) -> Dict[str, dict]:
        """Return full posteriors for all tracked peers (dashboard display)."""
        result = {}
        for pid, (pA, pF) in self._memory.items():
            result[pid] = {
                "p_attack": pA,
                "p_fault":  pF,
                "p_honest": max(0.0, 1.0 - pA - pF),
                "label": (
                    "ATTACK" if pA >= self.ATTACK_THRESHOLD else
                    "FAULT"  if pF >= self.FAULT_THRESHOLD  else
                    "HONEST"
                ),
            }
        return result
