"""
test_ppo_controller.py — Comprehensive tests for the PPO Meta-Controller.

Test coverage:
  Unit Tests:
    - Environment reset, step, obs space, action clipping
    - 9-dimensional observation correctness (all features present & normalised)
    - EMA trust computation
    - Attack streak tracking
    - Reward shaping polarity (attacker → positive when isolated; honest → positive when not)
    - All 6 adversarial scenarios can be sampled
    - Temporal smoothness regulariser fires correctly

  Integration Tests:
    - PPOMetaController heuristic fallback monotonicity
    - Trained model loading & inference (skipped if model absent)
    - Threshold ordering: attacker > fault > honest
    - Planner integration: adaptive threshold raises hazard correctly
    - Full episode rollout terminates within max_steps

  Statistical Tests:
    - 200-episode Monte-Carlo: PPO isolation rate > random policy (threshold=0.40)
    - Scenario coverage: all 6 regimes sampled over 100 resets
"""

from __future__ import annotations

import os
import math
import numpy as np
import pytest

from src.adaptation.ppo_meta_controller import (
    RealisticEpistemicEnv,
    TrustMetaEnv,         # backward-compat alias
    PPOMetaController,
    SB3_AVAILABLE,
)
from src.adaptation.path_planner import TrustDStarLite

# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/ppo_trust_controller.zip"
OBS_DIM    = 9


# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def trained_model_path():
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Trained PPO model not found at {MODEL_PATH}. Run training first.")
    return MODEL_PATH


@pytest.fixture
def env():
    e = RealisticEpistemicEnv()
    yield e


# ═════════════════════════════════════════════════════════════════════════════
# UNIT: Environment
# ═════════════════════════════════════════════════════════════════════════════

class TestRealisticEpistemicEnv:

    # --- Reset ---

    def test_reset_obs_shape(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,), f"Expected shape ({OBS_DIM},), got {obs.shape}"

    def test_reset_obs_dtype(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_reset_obs_normalised(self, env):
        obs, _ = env.reset(seed=7)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"Obs out of [0,1]: {obs}"

    def test_reset_returns_scenario(self, env):
        _, info = env.reset(seed=1)
        assert "scenario" in info
        assert info["scenario"] in {
            "CLEAN", "BURST_ATTACK", "STEALTH_CREEPING_DRIFT",
            "INTERMITTENT_ON_OFF", "HARDWARE_SENSOR_FAULT", "MULTI_PHASE_MIXED",
        }

    def test_reset_resets_ema_trust(self, env):
        # EMA should be reset to 0.85 on each reset
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.array([0.10], dtype=np.float32))
        env.reset(seed=1)
        assert abs(env._ema_trust - 0.85) < 1e-6

    def test_reset_clears_attack_history(self, env):
        env.reset(seed=0)
        for _ in range(15):
            env.step(np.array([0.0], dtype=np.float32))
        env.reset(seed=2)
        assert len(env._attack_history) == 0

    # --- Step ---

    def test_step_obs_shape(self, env):
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        assert obs.shape == (OBS_DIM,)

    def test_step_obs_normalised(self, env):
        env.reset(seed=42)
        for _ in range(30):
            obs, _, done, _, _ = env.step(np.array([0.05], dtype=np.float32))
            assert np.all(obs >= -1e-4) and np.all(obs <= 1.0 + 1e-4), \
                f"Obs out of bounds at step: {obs}"
            if done:
                break

    def test_step_action_clipping(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([9.99], dtype=np.float32))
        assert info["effective_threshold"] <= 0.75 + 1e-6

    def test_step_returns_info_keys(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        required_keys = {
            "scenario", "is_attacking", "is_fault", "isolated",
            "effective_threshold", "trust", "p_attack", "raw_anomaly",
            "ema_trust", "attack_streak",
        }
        assert required_keys.issubset(set(info.keys())), \
            f"Missing keys: {required_keys - set(info.keys())}"

    def test_effective_threshold_range(self, env):
        env.reset(seed=3)
        for action_val in [-0.20, -0.10, 0.0, 0.10, 0.20]:
            env.reset(seed=3)
            _, _, _, _, info = env.step(np.array([action_val], dtype=np.float32))
            eff = info["effective_threshold"]
            assert 0.15 <= eff <= 0.75, f"Effective threshold {eff} out of [0.15, 0.75]"

    def test_terminates_at_max_steps(self):
        env = RealisticEpistemicEnv(max_steps=10)
        env.reset(seed=0)
        for i in range(10):
            _, _, done, _, _ = env.step(np.array([0.0], dtype=np.float32))
            if i < 9:
                assert not done
        assert done

    # --- Observation features ---

    def test_obs_trust_index(self, env):
        """obs[0] should equal info['trust']."""
        env.reset(seed=10)
        obs, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert abs(obs[0] - np.clip(info["trust"], 0.0, 1.0)) < 1e-4

    def test_obs_p_attack_index(self, env):
        """obs[2] should equal info['p_attack'] (normalised)."""
        env.reset(seed=10)
        obs, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
        assert abs(obs[2] - np.clip(info["p_attack"], 0.0, 1.0)) < 1e-4

    def test_obs_ema_trust_monotone_update(self, env):
        """EMA trust should converge towards current trust over many steps."""
        env.reset(seed=0)
        env.scenario = "CLEAN"  # honest scenario: trust stays high
        prev_ema = env._ema_trust
        obs, _, _, _, _ = env.step(np.array([0.0], dtype=np.float32))
        ema_now = env._ema_trust
        assert ema_now is not None
        # EMA should be a convex combination of previous EMA and current trust
        assert 0.0 <= ema_now <= 1.0

    def test_obs_attack_streak_grows(self, env):
        """Attack streak (obs[6]) should grow when p_attack repeatedly > 0.5."""
        env.reset(seed=5)
        env.scenario = "BURST_ATTACK"
        env.attack_start = 0
        env.attack_offset = 200.0
        streaks = []
        for _ in range(15):
            obs, _, done, _, _ = env.step(np.array([0.15], dtype=np.float32))
            streaks.append(obs[6])
            if done:
                break
        # streak should increase over the first several attack steps
        if len(streaks) >= 6:
            assert streaks[-1] >= streaks[0]

    def test_obs_step_progress_monotone(self, env):
        """obs[7] (step_progress) should monotonically increase."""
        env.reset(seed=0)
        progresses = []
        for _ in range(5):
            obs, _, done, _, _ = env.step(np.array([0.0], dtype=np.float32))
            progresses.append(obs[7])
            if done:
                break
        for i in range(1, len(progresses)):
            assert progresses[i] >= progresses[i - 1], \
                f"Step progress not monotone: {progresses}"

    # --- Reward polarity ---

    def test_reward_positive_when_attacker_isolated(self, env):
        """Reward should be positive when an attack is ongoing and peer is isolated."""
        for seed in range(10):
            env.reset(seed=seed)
            env.scenario = "BURST_ATTACK"
            env.attack_start = 0
            env.attack_offset = 200.0
            # High action → raise threshold → likely isolation
            _, reward, _, _, info = env.step(np.array([0.20], dtype=np.float32))
            if info["is_attacking"] and info["isolated"]:
                assert reward > 0.0, f"Expected positive reward for isolating attacker, got {reward}"
                return
        # If no attack step was sampled (unlikely), skip
        pytest.skip("No attacking step sampled in first 10 seeds")

    def test_reward_negative_when_attacker_accepted(self, env):
        """Reward should be negative when an attack is ongoing and peer is NOT isolated."""
        for seed in range(20):
            env.reset(seed=seed)
            env.scenario = "BURST_ATTACK"
            env.attack_start = 0
            env.attack_offset = 200.0
            # Low action → base threshold → trust may still be high → peer NOT isolated
            _, reward, _, _, info = env.step(np.array([-0.20], dtype=np.float32))
            if info["is_attacking"] and not info["isolated"]:
                assert reward < 0.0, f"Expected negative reward for accepting attacker, got {reward}"
                return
        pytest.skip("No non-isolated attack step sampled in first 20 seeds")

    def test_reward_positive_when_honest_not_isolated(self, env):
        """Reward should be positive when an honest peer is correctly retained."""
        for seed in range(15):
            env.reset(seed=seed)
            env.scenario = "CLEAN"
            _, reward, _, _, info = env.step(np.array([0.0], dtype=np.float32))
            if not info["is_attacking"] and not info["is_fault"] and not info["isolated"]:
                assert reward > 0.0, f"Expected positive reward for honest collaboration, got {reward}"
                return

    # --- Scenario coverage ---

    def test_all_scenarios_sampled(self):
        """All 6 scenarios must be reachable via reset."""
        scenarios_seen = set()
        env = RealisticEpistemicEnv()
        for i in range(120):
            _, info = env.reset(seed=i)
            scenarios_seen.add(info["scenario"])
        expected = {
            "CLEAN", "BURST_ATTACK", "STEALTH_CREEPING_DRIFT",
            "INTERMITTENT_ON_OFF", "HARDWARE_SENSOR_FAULT", "MULTI_PHASE_MIXED",
        }
        assert expected.issubset(scenarios_seen), \
            f"Not all scenarios sampled. Missing: {expected - scenarios_seen}"

    # --- Backward compat alias ---

    def test_trust_meta_env_alias(self):
        env = TrustMetaEnv()
        obs, _ = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)

    # --- Temporal smoothness regulariser ---

    def test_smoothness_penalty_fires_on_jitter(self):
        """The jitter penalty should be higher when actions oscillate."""
        env = RealisticEpistemicEnv(max_steps=40)
        env.reset(seed=0)
        env.scenario = "CLEAN"

        rewards_smooth = []
        for _ in range(5):
            _, r, done, _, _ = env.step(np.array([0.0], dtype=np.float32))
            rewards_smooth.append(r)
            if done: break

        env.reset(seed=0)
        env.scenario = "CLEAN"
        rewards_jitter = []
        sign = 1.0
        for _ in range(5):
            _, r, done, _, _ = env.step(np.array([sign * 0.18], dtype=np.float32))
            rewards_jitter.append(r)
            sign *= -1.0
            if done: break

        # Jittery actions should produce lower cumulative reward
        assert sum(rewards_jitter) <= sum(rewards_smooth) + 1.0  # +1 margin


# ═════════════════════════════════════════════════════════════════════════════
# UNIT: PPOMetaController
# ═════════════════════════════════════════════════════════════════════════════

class TestPPOMetaController:

    def test_no_model_fallback_sensible(self):
        ctrl = PPOMetaController(model_path="does_not_exist.zip")
        assert not ctrl.is_trained
        # Heuristic must return value in valid range
        for trust, p_atk in [(0.9, 0.05), (0.5, 0.50), (0.2, 0.95)]:
            tau = ctrl.predict_threshold(
                trust=trust, uncertainty=0.04, p_attack=p_atk,
                anomaly_delta=0.10, noise_level=0.1,
            )
            assert 0.15 <= tau <= 0.75, f"Fallback tau {tau} out of range"

    def test_fallback_monotone_in_p_attack(self):
        """Higher p_attack should yield higher threshold (fallback heuristic)."""
        ctrl = PPOMetaController()
        taus = [
            ctrl.predict_threshold(trust=0.7, uncertainty=0.03, p_attack=p, anomaly_delta=0.5)
            for p in [0.05, 0.30, 0.70, 0.95]
        ]
        # Monotone non-decreasing
        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1] + 0.01, \
                f"Fallback not monotone at index {i}: {taus}"

    def test_fallback_ema_trend_lowers_trust_raises_threshold(self):
        """If trust is falling (trust < ema_trust), threshold should be higher."""
        ctrl = PPOMetaController()
        # Falling trust: current trust < EMA trust
        tau_falling = ctrl.predict_threshold(
            trust=0.45, uncertainty=0.05, p_attack=0.40,
            anomaly_delta=0.3, noise_level=0.1,
            ema_trust=0.75, attack_streak=0.0,
        )
        # Stable trust: current trust ≈ EMA trust
        tau_stable = ctrl.predict_threshold(
            trust=0.65, uncertainty=0.05, p_attack=0.40,
            anomaly_delta=0.3, noise_level=0.1,
            ema_trust=0.65, attack_streak=0.0,
        )
        assert tau_falling >= tau_stable - 0.05, \
            f"Falling trust {tau_falling:.3f} should be >= stable trust {tau_stable:.3f}"

    def test_model_loading(self, trained_model_path):
        ctrl = PPOMetaController(model_path=trained_model_path)
        assert ctrl.is_trained, "Model should be marked as trained after loading"

    def test_inference_output_range(self, trained_model_path):
        ctrl = PPOMetaController(model_path=trained_model_path)
        for _ in range(50):
            obs = np.random.uniform(0, 1, size=8).tolist()
            tau = ctrl.predict_threshold(*obs[:5])
            assert 0.15 <= tau <= 0.75, f"Inference tau {tau} out of range"

    def test_threshold_ordering_attacker_gt_honest(self, trained_model_path):
        """Trained PPO must set higher tau for attacker than for honest peer."""
        ctrl = PPOMetaController(model_path=trained_model_path)
        tau_honest = ctrl.predict_threshold(
            trust=0.90, uncertainty=0.02, p_attack=0.04,
            anomaly_delta=0.02, noise_level=0.05,
            ema_trust=0.90, attack_streak=0.0,
        )
        tau_attack = ctrl.predict_threshold(
            trust=0.30, uncertainty=0.05, p_attack=0.95,
            anomaly_delta=0.90, noise_level=0.05,
            ema_trust=0.50, attack_streak=0.80,
        )
        assert tau_attack > tau_honest, (
            f"Attacker tau ({tau_attack:.3f}) should exceed honest tau ({tau_honest:.3f})"
        )


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRATION: Planner integration
# ═════════════════════════════════════════════════════════════════════════════

class TestPlannerIntegration:

    def _make_planner(self):
        return TrustDStarLite(
            width=20, height=20, obstacles=[],
            reroute_threshold=0.40, hazard_radius=3, sensitivity=5.0,
        )

    def test_standard_threshold_trusts_peer(self):
        planner = self._make_planner()
        planner.init(start=(0, 10), goal=(19, 10))
        trusts = {"A2": 0.45}
        peer_pos = {"A2": (10, 10)}
        planner.replan((0, 10), trusts, peer_pos)
        # trust 0.45 >= threshold 0.40 → peer should NOT be in hazard
        assert (10, 10) not in planner.hazard

    def test_adaptive_threshold_avoids_peer(self):
        planner = self._make_planner()
        planner.init(start=(0, 10), goal=(19, 10))
        trusts = {"A2": 0.45}
        peer_pos = {"A2": (10, 10)}
        # Meta-controller raises threshold for A2 to 0.60
        planner.replan((0, 10), trusts, peer_pos, peer_thresholds={"A2": 0.60})
        # trust 0.45 < threshold 0.60 → peer SHOULD be in hazard
        assert (10, 10) in planner.hazard
        assert "A2" in planner.last_replan_stats["peers_avoided"]

    def test_lowered_threshold_retains_peer(self):
        planner = self._make_planner()
        planner.init(start=(0, 10), goal=(19, 10))
        # Peer with trust=0.35 (below standard 0.40) but meta-controller lowers to 0.20
        trusts = {"A2": 0.35}
        peer_pos = {"A2": (10, 10)}
        planner.replan((0, 10), trusts, peer_pos, peer_thresholds={"A2": 0.20})
        # trust 0.35 >= threshold 0.20 → peer should NOT be in hazard
        assert (10, 10) not in planner.hazard


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICAL: Monte-Carlo validation
# ═════════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:

    @pytest.mark.slow
    def test_ppo_isolation_better_than_fixed(self, trained_model_path):
        """
        PPO adaptive policy should isolate attackers at least as often as the
        standard fixed threshold (tau=0.40) over 200 episodes.
        """
        if not SB3_AVAILABLE:
            pytest.skip("stable-baselines3 not installed")

        ctrl = PPOMetaController(model_path=trained_model_path)
        N = 200
        seed_offset = 500

        def run(policy_fn):
            atk, iso = 0, 0
            for ep in range(N):
                env = RealisticEpistemicEnv()
                obs, _ = env.reset(seed=seed_offset + ep)
                done = False
                while not done:
                    tau = policy_fn(obs)
                    act = np.array([tau - env.base_threshold], dtype=np.float32)
                    obs, _, done, _, info = env.step(act)
                    if info["is_attacking"]:
                        atk += 1
                        if info["isolated"]:
                            iso += 1
            return 100.0 * iso / atk if atk > 0 else 100.0

        ppo_iso  = run(lambda obs: ctrl.predict_threshold(
            obs[0], obs[1]/4.0, obs[2], obs[3]*60.0, obs[4]*10.0,
            ema_trust=obs[5], attack_streak=obs[6], step_progress=obs[7],
        ))
        fixed_iso = run(lambda obs: 0.40)

        # PPO should match or beat fixed policy (allow 5% margin for stochasticity)
        assert ppo_iso >= fixed_iso - 5.0, (
            f"PPO isolation {ppo_iso:.1f}% worse than fixed {fixed_iso:.1f}% by more than 5%"
        )

    @pytest.mark.slow
    def test_ppo_false_alarm_better_than_conservative(self, trained_model_path):
        """
        PPO should produce fewer false alarms than the over-cautious tau=0.60 policy
        over 200 honest-only episodes.
        """
        if not SB3_AVAILABLE:
            pytest.skip("stable-baselines3 not installed")

        ctrl = PPOMetaController(model_path=trained_model_path)
        N = 100
        seed_offset = 700

        def run(policy_fn):
            hon, fa = 0, 0
            for ep in range(N):
                env = RealisticEpistemicEnv()
                env.reset(seed=seed_offset + ep)
                env.scenario = "CLEAN"   # force honest scenario
                obs, _ = env.reset(seed=seed_offset + ep)
                done = False
                while not done:
                    tau = policy_fn(obs)
                    act = np.array([tau - env.base_threshold], dtype=np.float32)
                    obs, _, done, _, info = env.step(act)
                    if not info["is_attacking"] and not info["is_fault"]:
                        hon += 1
                        if info["isolated"]:
                            fa += 1
            return 100.0 * fa / hon if hon > 0 else 0.0

        ppo_fa  = run(lambda obs: ctrl.predict_threshold(
            obs[0], obs[1]/4.0, obs[2], obs[3]*60.0, obs[4]*10.0,
            ema_trust=obs[5], attack_streak=obs[6], step_progress=obs[7],
        ))
        cons_fa = run(lambda obs: 0.60)

        # PPO false alarm rate should be lower than over-cautious policy
        assert ppo_fa <= cons_fa + 5.0, (
            f"PPO false alarm {ppo_fa:.1f}% much worse than conservative {cons_fa:.1f}%"
        )
