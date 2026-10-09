"""
ppo_meta_controller.py — Advanced PPO Meta-Controller for Closed-Loop Epistemic Adaptation.

Theoretical Background:
-----------------------
In multi-robot epistemic foraging under adversarial interception, static trust
reroute thresholds (e.g. fixed tau = 0.40) suffer from the classic security dilemma:
  - Over-cautious threshold (tau = 0.60): High false alarm rate under sensor noise,
    causing fleets to execute wasteful detour replans around innocent peers.
  - Under-cautious threshold (tau = 0.20): Slow reaction to stealthy or burst
    false-data injection attacks, exposing robots to collisions or poisoned maps.

The PPO Meta-Controller operates as a high-level policy over the Beta-Binomial
trust engine and the Dynamic Bayesian Network (DBN):

    s  = [trust, uncertainty, p_attack, delta_norm, noise_norm,
          ema_trust, attack_streak_norm, step_progress, last_action_norm]  (9-dim)
    a in [-0.20, +0.20]  ==>  effective tau = tau_base + a in [0.20, 0.60]

Observation Vector (9-dim):
  0. trust             — Beta-posterior mean ∈ [0, 1]
  1. uncertainty       — Beta posterior std (normalised) ∈ [0, 1]
  2. p_attack          — DBN posterior P(attack | evidence) ∈ [0, 1]
  3. delta_norm        — Normalised kinematic anomaly magnitude ∈ [0, 1]
  4. noise_norm        — Normalised sensor noise standard deviation ∈ [0, 1]
  5. ema_trust         — Exponential moving average of trust (α=0.15) ∈ [0, 1]
  6. attack_streak     — Fraction of last 10 steps with p_attack > 0.5 ∈ [0, 1]
  7. step_progress     — Normalised episode step counter ∈ [0, 1]
  8. last_action_norm  — Previous action mapped to [0, 1] for auto-correlation

High-Fidelity Training Environment (RealisticEpistemicEnv):
------------------------------------------------------------
Unlike naive synthetic sampling, RealisticEpistemicEnv runs the ACTUAL closed-loop
telemetry pipeline across 6 adversarial and operational regimes:
  1. CLEAN_COLLABORATION    — Gaussian sensor noise, honest telemetry.
  2. BURST_ATTACK           — False-data injection with large coordinate offsets.
  3. STEALTH_CREEPING_DRIFT — Subtle sub-threshold velocity drift that accumulates.
  4. INTERMITTENT_ON_OFF    — Pulsing attacks designed to exploit Beta trust recovery.
  5. HARDWARE_SENSOR_FAULT  — Sensor degradation & packet loss without malicious intent.
  6. MULTI_PHASE_MIXED      — Transitions between honest, fault, and attack phases.

Reward Shaping:
  - Proportional security bonus when isolating an attacker (scales with anomaly severity)
  - Proportional penalty when accepting attacker data (scales with injected anomaly)
  - Fault-tolerance reward: rewarding non-isolation of faulty (but innocent) peers
  - Productivity reward: non-isolation of honest peers; penalises unnecessary detours
  - Temporal smoothness regulariser: discourages high-frequency threshold jitter
  - Conservative-action bonus: small reward for choosing minimal effective action
"""

from __future__ import annotations

import os
import math
import argparse
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from src.adaptation.trust_manager import BetaTrustEngine
from src.adaptation.bayesian_network import DynamicBayesianClassifier
from src.adaptation.feature_extractor import FeatureExtractor

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    gym = None
    spaces = None
    PPO = None
    DummyVecEnv = None


if SB3_AVAILABLE:
    class RealisticEpistemicEnv(gym.Env):
        """
        Closed-loop gymnasium environment simulating real epistemic foraging dynamics.
        Integrates FeatureExtractor, DynamicBayesianClassifier, and BetaTrustEngine.
        """

        metadata = {"render_modes": []}
        OBS_DIM = 9  # trust, uncertainty, p_attack, delta_norm, noise_norm,
                     # ema_trust, attack_streak_norm, step_progress, last_action_norm

        def __init__(self, base_threshold: float = 0.40, max_steps: int = 120):
            super().__init__()
            self.base_threshold = base_threshold
            self.max_steps = max_steps
            self.current_step = 0
            self.peer_id = "A_peer"

            # 9-dimensional observation space (all normalised to [0, 1])
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32
            )
            # Action space: continuous threshold adjustment in [-0.20, +0.20]
            self.action_space = spaces.Box(
                low=-0.20, high=0.20, shape=(1,), dtype=np.float32
            )

            self.trust_engine: Optional[BetaTrustEngine] = None
            self.dbn: Optional[DynamicBayesianClassifier] = None
            self.fe: Optional[FeatureExtractor] = None

            self.scenario = "CLEAN"
            self.attack_start = 0
            self.attack_duration = 0
            self.attack_offset = 0.0
            self.stealth_drift = 0.0
            self.sensor_noise_base = 1.8
            self.last_action = 0.0

            # Extended state for richer observations
            self._ema_trust: float = 0.85
            self._ema_alpha: float = 0.15
            self._attack_history: List[float] = []  # recent p_attack window (len 10)

        def _build_obs(
            self,
            trust: float,
            unc: float,
            p_atk: float,
            delta_norm: float,
            noise_norm: float,
        ) -> np.ndarray:
            """Compose the 9-dimensional observation vector."""
            # EMA trust: slow-moving average captures trend rather than instant value
            self._ema_trust = (
                (1.0 - self._ema_alpha) * self._ema_trust
                + self._ema_alpha * trust
            )
            # Attack streak: fraction of recent window where p_attack > 0.5
            self._attack_history.append(float(p_atk > 0.50))
            if len(self._attack_history) > 10:
                self._attack_history.pop(0)
            attack_streak = float(np.mean(self._attack_history))

            step_progress = float(self.current_step) / float(self.max_steps)
            last_action_norm = float(np.clip((self.last_action + 0.20) / 0.40, 0.0, 1.0))

            return np.array(
                [
                    np.clip(trust,                  0.0, 1.0),
                    np.clip(unc * 4.0,              0.0, 1.0),
                    np.clip(p_atk,                  0.0, 1.0),
                    np.clip(delta_norm,             0.0, 1.0),
                    np.clip(noise_norm,             0.0, 1.0),
                    np.clip(self._ema_trust,        0.0, 1.0),
                    np.clip(attack_streak,          0.0, 1.0),
                    np.clip(step_progress,          0.0, 1.0),
                    np.clip(last_action_norm,       0.0, 1.0),
                ],
                dtype=np.float32,
            )

        def reset(
            self, seed: Optional[int] = None, options: Optional[dict] = None
        ) -> Tuple[np.ndarray, dict]:
            super().reset(seed=seed)
            self.current_step = 0
            self.last_action = 0.0
            self._ema_trust = 0.85
            self._attack_history = []

            # Instantiate real epistemic modules
            self.trust_engine = BetaTrustEngine(alpha0=4.0, beta0=1.0, aging=0.97)
            self.dbn = DynamicBayesianClassifier()
            self.fe = FeatureExtractor()

            # Sample realistic operational scenario
            self.scenario = self.np_random.choice(
                [
                    "CLEAN",
                    "BURST_ATTACK",
                    "STEALTH_CREEPING_DRIFT",
                    "INTERMITTENT_ON_OFF",
                    "HARDWARE_SENSOR_FAULT",
                    "MULTI_PHASE_MIXED",
                ],
                p=[0.25, 0.25, 0.15, 0.15, 0.10, 0.10],
            )

            self.attack_start = int(self.np_random.integers(10, 25))
            self.attack_duration = int(self.np_random.integers(25, 55))
            self.attack_offset = float(self.np_random.uniform(50.0, 250.0))
            self.stealth_drift = float(self.np_random.uniform(0.5, 2.5))
            self.sensor_noise_base = float(self.np_random.uniform(1.0, 3.0))

            trust, unc = self.trust_engine.get(self.peer_id)
            obs = self._build_obs(
                trust=trust, unc=unc, p_atk=0.05,
                delta_norm=0.0, noise_norm=self.sensor_noise_base / 10.0,
            )
            # Restore canonical initial state: _build_obs mutates EMA/history,
            # but at reset we want a clean baseline (ema = 0.85, no history).
            self._ema_trust = 0.85
            self._attack_history = []
            return obs, {"scenario": self.scenario}

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
            self.current_step += 1
            act = float(np.clip(action[0], -0.20, 0.20))
            eff_threshold = float(np.clip(self.base_threshold + act, 0.15, 0.75))

            is_attacking = False
            is_fault = False
            raw_anomaly = 0.0
            noise_std = self.sensor_noise_base

            # Simulate peer telemetry stream according to scenario
            if self.scenario == "CLEAN":
                raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
            elif self.scenario == "BURST_ATTACK":
                if self.attack_start <= self.current_step < self.attack_start + self.attack_duration:
                    is_attacking = True
                    raw_anomaly = self.attack_offset + float(self.np_random.normal(0, noise_std))
                else:
                    raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
            elif self.scenario == "STEALTH_CREEPING_DRIFT":
                if self.current_step >= self.attack_start:
                    is_attacking = True
                    drift_ticks = self.current_step - self.attack_start
                    raw_anomaly = 12.0 + drift_ticks * self.stealth_drift + float(self.np_random.normal(0, noise_std))
                else:
                    raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
            elif self.scenario == "INTERMITTENT_ON_OFF":
                if self.current_step >= self.attack_start:
                    cycle = (self.current_step - self.attack_start) % 20
                    if cycle < 8:
                        is_attacking = True
                        raw_anomaly = self.attack_offset + float(self.np_random.normal(0, noise_std))
                    else:
                        raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
                else:
                    raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
            elif self.scenario == "HARDWARE_SENSOR_FAULT":
                is_fault = True
                noise_std = 8.0
                raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
            elif self.scenario == "MULTI_PHASE_MIXED":
                if self.current_step < 20:
                    raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
                elif self.current_step < 40:
                    is_fault = True
                    noise_std = 7.0
                    raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))
                elif self.current_step < 65:
                    is_attacking = True
                    raw_anomaly = self.attack_offset + float(self.np_random.normal(0, noise_std))
                else:
                    raw_anomaly = abs(float(self.np_random.normal(0, noise_std)))

            # Step 1: Discrete Feature Extraction
            event = {
                "delta": raw_anomaly,
                "reason": "KINEMATIC" if raw_anomaly > 15.0 else ("NOISE" if is_fault else ""),
            }
            features = self.fe.extract(self.peer_id, event)

            # Step 2: Dynamic Bayesian Network Inference
            dbn_res = self.dbn.classify(self.peer_id, features)
            p_atk   = float(dbn_res["p_attack"])
            label   = dbn_res["label"]

            # Step 3: Beta Trust Engine Update
            self.trust_engine.update(self.peer_id, p_atk, raw_anomaly, label)
            trust, unc = self.trust_engine.get(self.peer_id)

            # Step 4: Closed-Loop Decision & Reward Evaluation
            isolated = (trust < eff_threshold)
            reward = 0.0

            # Proportional security bonus / penalty based on actual anomaly magnitude
            severity = float(np.clip(raw_anomaly / 60.0, 0.0, 3.0))

            if is_attacking:
                if isolated:
                    # Successful isolation: proportional to severity of injected data
                    # Extra bonus if agent raised threshold proactively (act > 0)
                    proactive_bonus = 0.40 * max(0.0, act)
                    reward += 2.50 + severity * 0.60 + proactive_bonus
                else:
                    # Security failure: penalty scales with anomaly magnitude
                    reward -= 3.00 + severity * 1.20
            elif is_fault:
                if not isolated:
                    # Fault tolerance: kept honest-but-noisy peer in coalition
                    # Penalise if agent over-raised threshold (unnecessary caution)
                    caution_cost = 0.60 * max(0.0, act)
                    reward += 1.40 - caution_cost
                else:
                    # Over-isolation of faulty peer: lost collaboration
                    reward -= 0.90
            else:  # Honest, clean peer
                if not isolated:
                    # Productive collaboration; slight penalty for unnecessary threshold raise
                    over_caution = 0.80 * max(0.0, act)
                    reward += 1.80 - over_caution
                else:
                    # False alarm: wasted detour, delayed mission
                    # Extra penalty if agent set a very high threshold (bad policy)
                    fa_severity  = 0.50 * max(0.0, act)
                    reward -= 2.40 + fa_severity

            # Conservative-action bonus: small reward for choosing the minimum
            # effective action (Occam's razor for threshold adaptation)
            reward += 0.03 * (0.20 - abs(act))

            # Temporal smoothness regulariser (penalises high-frequency jitter)
            reward -= 0.06 * ((act - self.last_action) ** 2)
            self.last_action = act

            delta_norm = float(np.clip(raw_anomaly / 60.0, 0.0, 1.0))
            noise_norm = float(np.clip(noise_std / 10.0, 0.0, 1.0))
            next_obs = self._build_obs(
                trust=trust, unc=unc, p_atk=p_atk,
                delta_norm=delta_norm, noise_norm=noise_norm,
            )

            terminated = (self.current_step >= self.max_steps)
            truncated  = False
            info = {
                "scenario":           self.scenario,
                "is_attacking":       is_attacking,
                "is_fault":           is_fault,
                "isolated":           isolated,
                "effective_threshold": eff_threshold,
                "trust":              trust,
                "p_attack":           p_atk,
                "raw_anomaly":        raw_anomaly,
                "ema_trust":          self._ema_trust,
                "attack_streak":      float(np.mean(self._attack_history)) if self._attack_history else 0.0,
            }
            return next_obs, reward, terminated, truncated, info

    # Backwards-compatible alias
    TrustMetaEnv = RealisticEpistemicEnv


class PPOMetaController:
    """
    PPO Meta-Controller for dynamic reroute threshold adaptation.

    Wraps a trained Stable-Baselines3 PPO policy to provide per-peer adaptive
    thresholds during D* Lite replanning. Includes an analytical heuristic
    fallback if the RL library is unavailable or no weights are loaded.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_threshold: float = 0.40,
    ):
        self.base_threshold = base_threshold
        self.model_path = model_path
        self.model: Optional[Any] = None
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> bool:
        """Load trained PPO policy weights from disk."""
        if not SB3_AVAILABLE:
            return False
        try:
            self.model = PPO.load(model_path, device="cpu")
            self.model_path = model_path
            self.is_trained = True
            return True
        except Exception as ex:
            print(f"[PPOMetaController] Failed to load model from {model_path}: {ex}")
            self.model = None
            self.is_trained = False
            return False

    def predict_threshold(
        self,
        trust: float,
        uncertainty: float,
        p_attack: float,
        anomaly_delta: float,
        noise_level: float = 0.0,
        ema_trust: Optional[float] = None,
        attack_streak: float = 0.0,
        step_progress: float = 0.5,
        last_action: float = 0.0,
    ) -> float:
        """
        Predict optimal reroute threshold given current epistemic state features.

        Parameters
        ----------
        trust          : Beta-posterior mean trust ∈ [0, 1].
        uncertainty    : Beta posterior std (raw, not normalised).
        p_attack       : DBN posterior probability of attack ∈ [0, 1].
        anomaly_delta  : Raw kinematic anomaly magnitude (metres or units).
        noise_level    : Normalised sensor noise level ∈ [0, 1].
        ema_trust      : Exponential moving average of trust (optional).
        attack_streak  : Fraction of recent steps with p_attack > 0.5 ∈ [0, 1].
        step_progress  : Episode progress ∈ [0, 1] (optional, used internally).
        last_action    : Most recent action taken ∈ [-0.20, +0.20].

        Returns
        -------
        effective_threshold : float ∈ [0.15, 0.75]
        """
        if ema_trust is None:
            ema_trust = trust  # fallback: use current trust as EMA approximation

        delta_norm = float(np.clip(
            anomaly_delta / 60.0 if anomaly_delta > 1.0 else anomaly_delta, 0.0, 1.0
        ))
        last_act_norm = float(np.clip((last_action + 0.20) / 0.40, 0.0, 1.0))

        obs = np.array([
            np.clip(trust,                   0.0, 1.0),
            np.clip(uncertainty * 4.0,       0.0, 1.0),
            np.clip(p_attack,                0.0, 1.0),
            np.clip(delta_norm,              0.0, 1.0),
            np.clip(noise_level,             0.0, 1.0),
            np.clip(ema_trust,               0.0, 1.0),
            np.clip(attack_streak,           0.0, 1.0),
            np.clip(step_progress,           0.0, 1.0),
            np.clip(last_act_norm,           0.0, 1.0),
        ], dtype=np.float32)

        if self.is_trained and self.model is not None:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                delta_act = float(action[0])
                thresh = self.base_threshold + delta_act
                return float(np.clip(thresh, 0.15, 0.75))
            except Exception:
                pass

        # Analytical heuristic fallback (no RL weights loaded)
        # Integrates EMA trust trend and attack streak for richer decision
        risk_offset  = (p_attack - 0.40) * 0.30
        trend_offset = (trust - ema_trust) * (-0.20)   # falling trust → raise threshold
        streak_offset = attack_streak * 0.15           # persistent attacks → raise threshold
        if uncertainty > 0.15:
            risk_offset *= 0.6  # high uncertainty → dampen aggressive raises
        total_offset = risk_offset + trend_offset + streak_offset
        return float(np.clip(self.base_threshold + total_offset, 0.15, 0.75))


def train_meta_controller(
    total_timesteps: int = 120000,
    save_path: str = "models/ppo_trust_controller.zip",
    n_envs: int = 4,
    seed: int = 42,
    verbose: int = 1,
) -> str:
    """
    Train advanced PPO meta-controller on vectorized RealisticEpistemicEnv.

    Parameters
    ----------
    total_timesteps : Total environment interaction steps (120,000 recommended).
    save_path       : Destination filepath for saved policy weights.
    n_envs          : Number of parallel vectorized environments.
    seed            : Random seed for reproducibility.
    verbose         : SB3 verbosity level.

    Returns
    -------
    save_path : Absolute or relative path to saved policy.
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 and gymnasium are required to train the meta-controller.")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    def make_env(rank: int):
        def _init():
            env = RealisticEpistemicEnv()
            env.reset(seed=seed + rank)
            return env
        return _init

    env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Tuned MLP policy with separate policy and value heads
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )

    if verbose > 0:
        print(f"[PPO Training] Training on {n_envs} parallel closed-loop environments ({total_timesteps} steps)...")

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)

    if verbose > 0:
        print(f"[PPO Training] Training complete! Saved to {save_path}")

    return save_path


def benchmark_policy(
    model_path: str = "models/ppo_trust_controller.zip",
    n_episodes: int = 100,
    seed: int = 100,
) -> Dict[str, Any]:
    """
    Run an empirical benchmark comparing the trained PPO Meta-Controller
    against fixed baselines (tau=0.40, tau=0.60, tau=0.20) on identical seeds.
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 required for benchmarking.")

    controller = PPOMetaController(model_path=model_path)
    policies = {
        "PPO Adaptive (Ours)": lambda s: controller.predict_threshold(s[0], s[1]/4.0, s[2], s[3]*50.0, s[4]*10.0),
        "Fixed Standard (0.40)": lambda s: 0.40,
        "Over-Cautious (0.60)": lambda s: 0.60,
        "Permissive (0.20)": lambda s: 0.20,
    }

    results = {}

    for name, policy_fn in policies.items():
        total_rewards = []
        attack_steps = 0
        attack_isolated = 0
        honest_steps = 0
        honest_false_alarms = 0

        for ep in range(n_episodes):
            env = RealisticEpistemicEnv()
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False

            while not done:
                # Calculate threshold under this policy
                tau = policy_fn(obs)
                act = np.array([tau - env.base_threshold], dtype=np.float32)
                next_obs, rew, term, trunc, info = env.step(act)
                ep_reward += rew
                done = term or trunc
                obs = next_obs

                if info["is_attacking"]:
                    attack_steps += 1
                    if info["isolated"]:
                        attack_isolated += 1
                elif not info["is_fault"]:
                    honest_steps += 1
                    if info["isolated"]:
                        honest_false_alarms += 1

            total_rewards.append(ep_reward)

        isolation_rate = (attack_isolated / attack_steps * 100.0) if attack_steps > 0 else 100.0
        false_alarm_rate = (honest_false_alarms / honest_steps * 100.0) if honest_steps > 0 else 0.0

        results[name] = {
            "mean_reward": round(float(np.mean(total_rewards)), 2),
            "std_reward": round(float(np.std(total_rewards)), 2),
            "attack_isolation_pct": round(isolation_rate, 2),
            "false_alarm_pct": round(false_alarm_rate, 2),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate DEF PPO Meta-Controller")
    parser.add_argument("--train", action="store_true", help="Train advanced PPO meta-controller")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark against fixed baselines")
    parser.add_argument("--timesteps", type=int, default=120000, help="Total timesteps to train")
    parser.add_argument("--save", type=str, default="models/ppo_trust_controller.zip", help="Path to save model")
    args = parser.parse_args()

    if args.train:
        train_meta_controller(total_timesteps=args.timesteps, save_path=args.save)
    elif args.benchmark:
        res = benchmark_policy(model_path=args.save)
        print("\n" + "=" * 76)
        print(f"{'POLICY':<24} | {'MEAN REWARD':<12} | {'ISOLATION %':<14} | {'FALSE ALARM %':<14}")
        print("-" * 76)
        for pol, data in res.items():
            print(f"{pol:<24} | {data['mean_reward']:>12.2f} | {data['attack_isolation_pct']:>13.2f}% | {data['false_alarm_pct']:>13.2f}%")
        print("=" * 76 + "\n")
    else:
        ctrl = PPOMetaController(model_path=args.save)
        print("Model loaded:", ctrl.is_trained)
        test_obs = [
            (0.85, 0.02, 0.05, 0.05, 0.05),  # Honest
            (0.35, 0.05, 0.90, 0.85, 0.10),  # Attacker
            (0.60, 0.08, 0.15, 0.40, 0.70),  # Fault/Noise
        ]
        for t, u, p, d, n in test_obs:
            tau = ctrl.predict_threshold(t, u, p, d, n)
            print(f"Obs: trust={t}, p_atk={p} -> Adapted Threshold: {tau:.3f}")
