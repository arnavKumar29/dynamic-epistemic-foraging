"""
train_and_evaluate.py — Full PPO Meta-Controller Training & Evaluation Pipeline.

Pipeline stages:
  1. Curriculum-aware PPO training (400 k steps, 8 envs, two-phase LR schedule)
  2. Empirical benchmark vs. three fixed baselines (300 episodes, 6 scenarios)
  3. Per-scenario breakdown table
  4. Matplotlib training convergence + benchmark bar plots
  5. JSON results dump for dashboard integration

Usage:
    python scripts/train_and_evaluate.py
    python scripts/train_and_evaluate.py --skip-train   # reuse existing weights
"""

from __future__ import annotations

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
import math
import warnings
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptation.ppo_meta_controller import (
    RealisticEpistemicEnv,
    PPOMetaController,
    SB3_AVAILABLE,
)

if not SB3_AVAILABLE:
    raise SystemExit("stable-baselines3 + gymnasium required. pip install -r requirements.txt")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy

# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH  = str(ROOT / "models" / "ppo_trust_controller.zip")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Training
# ─────────────────────────────────────────────────────────────────────────────

class LearningCurveCallback(BaseCallback):
    """Collects per-rollout mean reward for post-hoc convergence plotting."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_rewards: List[float]   = []
        self.ep_lengths: List[int]     = []
        self.timesteps:  List[int]     = []

    def _on_step(self) -> bool:
        # SB3 stores completed episode stats in self.locals['infos']
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.timesteps.append(self.num_timesteps)
        return True


def make_env(rank: int, seed: int = 42):
    def _init():
        env = RealisticEpistemicEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def run_training(total_timesteps: int = 400_000, n_envs: int = 8, seed: int = 42) -> str:
    """
    Two-phase curriculum PPO training:
      Phase 1 (0 – 200k):  balanced 6-scenario distribution, LR=4e-4
      Phase 2 (200k – 400k): adversarial-heavy distribution, LR=2e-4
    """
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)

    print("=" * 72)
    print("  DEF PPO Meta-Controller — Advanced Training Pipeline")
    print("=" * 72)
    print(f"  Timesteps : {total_timesteps:,}")
    print(f"  Envs      : {n_envs} (DummyVecEnv)")
    print(f"  Seed      : {seed}")
    print("=" * 72)

    callback = LearningCurveCallback()

    # Phase 1 — balanced
    print("\n[Phase 1/2] Balanced 6-scenario curriculum  (200k steps, LR=4e-4)")
    env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=4e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=12,
        gamma=0.985,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.008,
        vf_coef=0.50,
        max_grad_norm=0.5,
        seed=seed,
        device="cpu",
        verbose=1,
    )
    model.learn(
        total_timesteps=total_timesteps // 2,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=False,
    )
    env.close()

    # Phase 2 — adversarial-heavy (re-use existing weights, swap envs)
    print("\n[Phase 2/2] Adversarial-heavy regime  (200k steps, LR=2e-4)")

    class AdversarialHeavyEnv(RealisticEpistemicEnv):
        """Override scenario probabilities towards attacks/faults."""
        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            # Force a heavier adversarial draw
            self.scenario = self.np_random.choice(
                [
                    "CLEAN",
                    "BURST_ATTACK",
                    "STEALTH_CREEPING_DRIFT",
                    "INTERMITTENT_ON_OFF",
                    "HARDWARE_SENSOR_FAULT",
                    "MULTI_PHASE_MIXED",
                ],
                p=[0.10, 0.30, 0.20, 0.20, 0.10, 0.10],
            )
            return obs, info

    def make_adv_env(rank: int, seed: int = 42):
        def _init():
            env = AdversarialHeavyEnv()
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init

    env2 = DummyVecEnv([make_adv_env(i, seed) for i in range(n_envs)])
    model.set_env(env2)
    model.learning_rate = 2e-4
    model.learn(
        total_timesteps=total_timesteps // 2,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    env2.close()

    model.save(MODEL_PATH)
    print(f"\n[Training] Model saved → {MODEL_PATH}")

    # Save learning curve data
    curve_path = str(RESULTS_DIR / "learning_curve.json")
    with open(curve_path, "w") as f:
        json.dump({
            "timesteps": callback.timesteps,
            "ep_rewards": callback.ep_rewards,
            "ep_lengths": callback.ep_lengths,
        }, f)
    print(f"[Training] Learning curve data → {curve_path}")

    return MODEL_PATH


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Benchmark
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    "CLEAN",
    "BURST_ATTACK",
    "STEALTH_CREEPING_DRIFT",
    "INTERMITTENT_ON_OFF",
    "HARDWARE_SENSOR_FAULT",
    "MULTI_PHASE_MIXED",
]

FIXED_POLICIES = {
    "Fixed τ=0.40 (Standard)": 0.40,
    "Fixed τ=0.60 (Conservative)": 0.60,
    "Fixed τ=0.20 (Permissive)": 0.20,
}


def _run_episodes(
    policy_fn,
    n_episodes: int = 300,
    seed: int = 200,
    per_scenario: bool = False,
) -> Dict[str, Any]:
    """Evaluate a policy over n_episodes, optionally tracking per-scenario results."""

    total_rewards: List[float] = []
    attack_steps = 0;  attack_isolated = 0
    honest_steps = 0;  honest_false_alarms = 0
    fault_steps  = 0;  fault_isolated = 0

    scenario_rewards: Dict[str, List[float]] = {s: [] for s in SCENARIOS}
    scenario_iso:     Dict[str, List[float]] = {s: [] for s in SCENARIOS}
    scenario_fa:      Dict[str, List[float]] = {s: [] for s in SCENARIOS}

    for ep in range(n_episodes):
        env = RealisticEpistemicEnv()
        obs, info = env.reset(seed=seed + ep)
        scenario = env.scenario
        ep_reward = 0.0
        ep_atk  = 0;  ep_atk_iso = 0
        ep_hon  = 0;  ep_hon_fa  = 0
        done = False

        while not done:
            tau = policy_fn(obs)
            act = np.array([tau - env.base_threshold], dtype=np.float32)
            obs, rew, term, trunc, info = env.step(act)
            ep_reward += rew
            done = term or trunc

            if info["is_attacking"]:
                attack_steps += 1; ep_atk += 1
                if info["isolated"]:
                    attack_isolated += 1; ep_atk_iso += 1
            elif info["is_fault"]:
                fault_steps += 1
                if info["isolated"]:
                    fault_isolated += 1
            else:
                honest_steps += 1; ep_hon += 1
                if info["isolated"]:
                    honest_false_alarms += 1; ep_hon_fa += 1

        total_rewards.append(ep_reward)
        scenario_rewards[scenario].append(ep_reward)
        if ep_atk  > 0: scenario_iso[scenario].append(100.0 * ep_atk_iso / ep_atk)
        if ep_hon  > 0: scenario_fa[scenario].append(100.0 * ep_hon_fa / ep_hon)

    iso_pct = (attack_isolated / attack_steps * 100.0)  if attack_steps > 0 else 100.0
    fa_pct  = (honest_false_alarms / honest_steps * 100.0) if honest_steps > 0 else 0.0
    fi_pct  = (fault_isolated / fault_steps * 100.0)   if fault_steps > 0 else 0.0

    per_sc = {}
    for sc in SCENARIOS:
        per_sc[sc] = {
            "mean_reward": round(float(np.mean(scenario_rewards[sc])) if scenario_rewards[sc] else 0.0, 2),
            "isolation_pct": round(float(np.mean(scenario_iso[sc])) if scenario_iso[sc] else 100.0, 2),
            "false_alarm_pct": round(float(np.mean(scenario_fa[sc])) if scenario_fa[sc] else 0.0, 2),
        }

    return {
        "mean_reward": round(float(np.mean(total_rewards)), 3),
        "std_reward":  round(float(np.std(total_rewards)),  3),
        "attack_isolation_pct": round(iso_pct, 2),
        "false_alarm_pct":      round(fa_pct, 2),
        "fault_isolation_pct":  round(fi_pct, 2),
        "per_scenario": per_sc,
    }


def run_benchmark(n_episodes: int = 300, seed: int = 200) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("  Benchmark: PPO vs. Fixed Baselines")
    print(f"  Episodes per policy: {n_episodes}  |  Seed: {seed}")
    print("=" * 72)

    ctrl = PPOMetaController(model_path=MODEL_PATH)
    assert ctrl.is_trained, "PPO model not loaded — run training first."

    policies = {
        "PPO Adaptive (Ours)": lambda obs: ctrl.predict_threshold(
            obs[0], obs[1] / 4.0, obs[2], obs[3] * 50.0, obs[4] * 10.0
        ),
    }
    for name, val in FIXED_POLICIES.items():
        v = val  # capture
        policies[name] = lambda obs, v=v: v

    results: Dict[str, Any] = {}
    for name, policy_fn in policies.items():
        t0 = time.time()
        print(f"  Evaluating: {name} ...", end="", flush=True)
        results[name] = _run_episodes(policy_fn, n_episodes=n_episodes, seed=seed)
        elapsed = time.time() - t0
        print(f"  done ({elapsed:.1f}s)")

    out_path = str(RESULTS_DIR / "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Benchmark] Results saved → {out_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Any]):
    W = 80
    SEP = "─" * W
    HDR = "═" * W

    print(f"\n{HDR}")
    print("  DYNAMIC EPISTEMIC FORAGING — PPO META-CONTROLLER EVALUATION REPORT")
    print(HDR)
    fmt = f"  {'Policy':<28} | {'Reward':>8} ± {'Std':>5} | {'Isolation':>10} | {'FalseAlarm':>11}"
    print(fmt)
    print(f"  {'':─<28}─┼─{'':─>8}─┼─{'':─>5}─┼─{'':─>10}─┼─{'':─>11}")
    for name, data in results.items():
        marker = " ◀" if "Ours" in name else "  "
        print(
            f"  {name:<28} | {data['mean_reward']:>8.3f} ± {data['std_reward']:>5.3f}"
            f" | {data['attack_isolation_pct']:>9.2f}%"
            f" | {data['false_alarm_pct']:>10.2f}%{marker}"
        )

    print(f"\n{SEP}")
    print("  Per-Scenario Breakdown (PPO Adaptive Only)")
    print(SEP)
    ppo_data = results.get("PPO Adaptive (Ours)", {})
    per_sc = ppo_data.get("per_scenario", {})
    print(f"  {'Scenario':<30} | {'Reward':>8} | {'Isolation':>10} | {'FalseAlarm':>10}")
    print(f"  {'':─<30}─┼─{'':─>8}─┼─{'':─>10}─┼─{'':─>10}")
    for sc, d in per_sc.items():
        print(
            f"  {sc:<30} | {d['mean_reward']:>8.2f}"
            f" | {d['isolation_pct']:>9.2f}%"
            f" | {d['false_alarm_pct']:>9.2f}%"
        )
    print(HDR + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(results: Dict[str, Any]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("[Plots] matplotlib not found — skipping plots.")
        return

    PALETTE = {
        "PPO Adaptive (Ours)":         "#4FC3F7",
        "Fixed τ=0.40 (Standard)":     "#FF8A65",
        "Fixed τ=0.60 (Conservative)": "#CE93D8",
        "Fixed τ=0.20 (Permissive)":   "#A5D6A7",
    }

    fig = plt.figure(figsize=(18, 10), facecolor="#0D1117")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    ax_curve  = fig.add_subplot(gs[0, :2])   # learning curve  (wide)
    ax_reward = fig.add_subplot(gs[0, 2])    # mean reward bar
    ax_iso    = fig.add_subplot(gs[1, 0])    # isolation rate
    ax_fa     = fig.add_subplot(gs[1, 1])    # false alarm rate
    ax_sc     = fig.add_subplot(gs[1, 2])    # per-scenario heatmap

    DARK   = "#0D1117"
    CARD   = "#161B22"
    BORDER = "#30363D"
    TEXT   = "#C9D1D9"
    DIM    = "#8B949E"

    for ax in [ax_curve, ax_reward, ax_iso, ax_fa, ax_sc]:
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

    # ── A: Learning Curve ──────────────────────────────────────────────
    curve_path = RESULTS_DIR / "learning_curve.json"
    if curve_path.exists():
        with open(curve_path) as f:
            curve = json.load(f)
        ts  = np.array(curve["timesteps"])
        rws = np.array(curve["ep_rewards"])
        # Rolling mean (window = 40 episodes)
        window = 40
        pad = np.full(window - 1, np.nan)
        padded = np.concatenate([pad, rws])
        rolling = np.array([np.nanmean(padded[i:i+window]) for i in range(len(rws))])
        ax_curve.plot(ts, rws,     color="#30363D", lw=0.6, alpha=0.5, label="Episode reward")
        ax_curve.plot(ts, rolling, color="#4FC3F7", lw=2.0, label=f"Rolling mean (w={window})")
        if len(ts) > 0:
            mid = ts[len(ts)//2]
            ax_curve.axvline(mid, color="#FF8A65", lw=1.2, ls="--", alpha=0.7, label="Phase 1 → Phase 2")
        ax_curve.set_title("PPO Training Convergence (Curriculum Two-Phase)", fontsize=11, pad=8)
        ax_curve.set_xlabel("Environment Steps")
        ax_curve.set_ylabel("Episode Return")
        ax_curve.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor=BORDER)
    else:
        ax_curve.text(0.5, 0.5, "No training data", ha="center", va="center",
                      color=DIM, transform=ax_curve.transAxes)
        ax_curve.set_title("Training Convergence", fontsize=11)

    def _bar(ax, metric_key: str, title: str, ylabel: str, fmt: str = "{:.2f}%"):
        names  = list(results.keys())
        values = [results[n][metric_key] for n in names]
        colors = [PALETTE.get(n, "#90CAF9") for n in names]
        bars   = ax.bar(range(len(names)), values, color=colors, width=0.6, zorder=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [n.split("(")[0].strip() for n in names],
            rotation=30, ha="right", fontsize=7,
        )
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", color=BORDER, lw=0.5, zorder=0)
        for bar, val in zip(bars, values):
            label = fmt.format(val)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                label, ha="center", va="bottom", color=TEXT, fontsize=7.5, fontweight="bold",
            )

    _bar(ax_reward, "mean_reward",          "Mean Episode Reward",      "Reward",       "{:.2f}")
    _bar(ax_iso,    "attack_isolation_pct", "Attack Isolation Rate",    "Isolation %",  "{:.1f}%")
    _bar(ax_fa,     "false_alarm_pct",      "False Alarm Rate (Lower↓)","False Alarm %","{:.1f}%")

    # ── E: Per-Scenario Heatmap (PPO only) ────────────────────────────
    ppo_sc = results.get("PPO Adaptive (Ours)", {}).get("per_scenario", {})
    sc_names  = list(ppo_sc.keys())
    metrics   = ["mean_reward", "isolation_pct", "false_alarm_pct"]
    mat       = np.array([[ppo_sc[s][m] for m in metrics] for s in sc_names], dtype=float)

    # Normalise each column for colour mapping
    col_mins  = mat.min(axis=0, keepdims=True)
    col_maxs  = mat.max(axis=0, keepdims=True)
    col_range = np.where((col_maxs - col_mins) < 1e-6, 1.0, col_maxs - col_mins)
    mat_norm  = (mat - col_mins) / col_range

    # False alarm: invert so green = good (low FA)
    mat_norm[:, 2] = 1.0 - mat_norm[:, 2]

    im = ax_sc.imshow(mat_norm.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_sc.set_xticks(range(len(sc_names)))
    ax_sc.set_xticklabels(
        [s.replace("_", "\n") for s in sc_names], fontsize=6.5, color=TEXT, rotation=0
    )
    ax_sc.set_yticks(range(len(metrics)))
    ax_sc.set_yticklabels(["Reward", "Isolation", "FA (inv.)"], fontsize=8, color=TEXT)
    ax_sc.set_title("Per-Scenario Performance (PPO)", fontsize=10, pad=6)
    # Annotate with raw values
    raw_labels = [
        [f"{mat[r,c]:.1f}" for c in range(len(metrics))]
        for r in range(len(sc_names))
    ]
    for r in range(len(sc_names)):
        for c in range(len(metrics)):
            ax_sc.text(r, c, raw_labels[r][c], ha="center", va="center", fontsize=6.5,
                       color="black", fontweight="bold")

    # ── Figure title & footer ─────────────────────────────────────────
    fig.text(
        0.5, 0.97,
        "Dynamic Epistemic Foraging — PPO Meta-Controller Evaluation",
        ha="center", va="top", fontsize=14, color=TEXT, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Trained on 6 adversarial regimes | Two-phase curriculum | 400k steps | DummyVecEnv×8",
        ha="center", fontsize=8, color=DIM,
    )
    fig.patch.set_facecolor(DARK)

    plot_path = str(RESULTS_DIR / "evaluation_report.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"[Plots] Evaluation report saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train",   action="store_true", help="Skip training, use existing model")
    parser.add_argument("--timesteps",    type=int, default=400_000)
    parser.add_argument("--n-envs",       type=int, default=8)
    parser.add_argument("--n-episodes",   type=int, default=300)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    # 1. Train
    if not args.skip_train:
        run_training(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
        )
    else:
        print(f"[Training] Skipped — using existing model at {MODEL_PATH}")

    # 2. Benchmark
    results = run_benchmark(n_episodes=args.n_episodes, seed=200)

    # 3. Report
    print_results_table(results)

    # 4. Plots
    generate_plots(results)

    print("\n✅  Full evaluation pipeline complete.\n")
