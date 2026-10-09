"""
main.py — Dynamic Epistemic Foraging simulation orchestrator.

Research-grade implementation — every algorithmic decision is visible
----------------------------------------------------------------------
This orchestrator wires together all three cognitive domains:

  DOMAIN 1 — NETWORK
    ECDH (SECP256R1) + AES-256-GCM + Replay Guard + MITM proxy.
    Agents broadcast encrypted position states via real UDP sockets.
    A scripted Man-in-the-Middle proxy intercepts selected packets and
    applies forgery, replay, or drop attacks according to attack_schedule.yaml.

  DOMAIN 2 — REASONING
    BDI Engine (Belief-Desire-Intention, Rao & Georgeff 1995).
    Bayesian Theory of Mind (Baker et al. 2011): Z3 SMT kinematic proof +
    Boltzmann-rational intent inference over named goals.
    Every paradox detection event is logged with the full SMT verdict,
    anomaly delta, and per-goal intent posteriors.

  DOMAIN 3 — ADAPTATION
    Feature Extractor → 2-slice Dynamic Bayesian Network (DBN) forward filter.
    Beta-Binomial trust with asymmetric recovery and grudge memory (Jøsang 2002).
    D* Lite trust-weighted incremental replanning (Koenig & Likhachev 2002).
    Continuous-space avoidance vector derived from the hazard field blended
    into FSM steering so robots move in ANY direction at any angle.

Continuous-angle omnidirectional motion
---------------------------------------
Robots are not grid-locked. steer_toward() uses math.atan2 + cos/sin so the
heading can be any value in [0, 2π).  Additionally, get_avoidance_vector()
from TrustDStarLite adds a continuous repulsion offset to the steering target
so agents physically curve away from distrusted zones rather than just planning
a grid path around them.

Rich structured logging
------------------------
Every tick:
  - TRUST_UPDATE logged with full Beta params + grudge tier.
  - EPISTEMIC_PARADOX logged with delta, SMT reason, intent posteriors.
  - DBN_STEP logged with complete forward-filter math trace.
  - REPLAN logged with cost delta and avoided peer list.
Every 100 ticks: HAZARD_SNAPSHOT logged for the D* Lite dashboard tab.

Lévy-flight foraging (Viswanathan et al. 1999)
-----------------------------------------------
Heavy-tailed power-law step distribution (α=1.5) — optimal for sparse-resource
environments. Each agent uses an independent RNG seeded from the master seed,
so paths are statistically independent while remaining reproducible.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import yaml

# Safe UTF-8 configuration for Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Domain imports ────────────────────────────────────────────────────── #
from src.network.secure_comms    import ECDHSession, SecureChannel
from src.network.mitm_proxy      import MITMProxy

from src.reasoning.bdi_engine    import BDIEngine
from src.adaptation.feature_extractor  import FeatureExtractor
from src.adaptation.bayesian_network   import DynamicBayesianClassifier
from src.adaptation.trust_manager      import BetaTrustEngine
from src.adaptation.path_planner       import TrustDStarLite
from src.adaptation.ppo_meta_controller import PPOMetaController

from src.environment.agent_state import AgentState
from src.dashboard.logger        import StructuredEventLedger

# ── Optional Pygame ───────────────────────────────────────────────────── #
try:
    from src.environment.arena import Arena
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════════ #
#  Lévy-flight helper (Mantegna 1994)                                     #
# ═══════════════════════════════════════════════════════════════════════ #

def levy_mantegna(alpha: float, rng: random.Random) -> float:
    """
    Sample from a Lévy-stable distribution using Mantegna's algorithm.

    Returns a positive float proportional to P(l) ∝ l^{-α}.
    α = 1.5 is theoretically optimal for sparse-resource foraging
    (Viswanathan et al. 1999, Nature 401:911-914).
    """
    sigma_u = (
        math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2) /
        (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))
    ) ** (1 / alpha)
    u = rng.gauss(0.0, sigma_u)
    v = abs(rng.gauss(0.0, 1.0)) + 1e-12
    return abs(u / v ** (1 / alpha))


# ═══════════════════════════════════════════════════════════════════════ #
#  Shared resource                                                         #
# ═══════════════════════════════════════════════════════════════════════ #

@dataclass
class Resource:
    x:          float
    y:          float
    radius:     float
    available:  bool = True
    respawn_at: int  = 0


# ═══════════════════════════════════════════════════════════════════════ #
#  AgentNode — full cognitive robot                                        #
# ═══════════════════════════════════════════════════════════════════════ #

class AgentNode:
    """
    Full cognitive robot agent combining all three domains.

    Cognitive loop per tick
    -----------------------
    1. FSM update → movement target
    2. steer_toward(target + avoidance_vector)  ← continuous-angle motion
    3. build_payload() + broadcast (encrypted UDP)
    4. decode_inbox() → decrypt raw bytes
    5. process_cognition() → BDI → DBN → Trust → Replan
    """

    EXPLORE  = "EXPLORE"
    APPROACH = "APPROACH"
    COLLECT  = "COLLECT"
    RETURN   = "RETURN"
    DEPOSIT  = "DEPOSIT"

    def __init__(
        self,
        agent_id:    int,
        pos:         Tuple[float, float, float],   # (x, y, heading)
        obstacles:   List[Tuple[float, float, float, float]],
        goals:       Dict[str, Tuple[float, float]],
        resources:   List[Resource],
        cfg:         dict,
        master_seed: int,
    ) -> None:
        self.agent_id  = agent_id
        self.agent_str = f"A{agent_id}"
        self.W         = cfg["environment"]["width"]
        self.H         = cfg["environment"]["height"]
        self.resources = resources
        self.goals     = goals

        # ── Per-agent independent RNG ─────────────────────────────────── #
        # XOR-hash master seed with agent id for statistical independence
        seed     = master_seed ^ (agent_id * 0x9e3779b9 & 0xFFFFFFFF)
        self.rng = random.Random(seed)

        # ── Kinematic state ───────────────────────────────────────────── #
        kin          = cfg["fleet"]["kinematic_limits"]
        self.max_vel = kin["max_velocity"]
        self.max_ang = kin["max_angular_velocity"]   # rad/tick
        self.pos_tol = kin.get("position_tolerance", 20.0)
        self.state   = AgentState(
            agent_id=agent_id,
            x=pos[0], y=pos[1], heading=pos[2],
            velocity=0.0, status="EXPLORE",
            tick=0,
        )

        # ── Foraging config ───────────────────────────────────────────── #
        forg = cfg.get("foraging", {})
        self.levy_alpha         = forg.get("levy_alpha",             1.5)
        self.sensor_noise_std   = forg.get("sensor_noise_std",       1.8)
        self.detect_radius      = forg.get("resource_detect_radius", 55.0)
        self.collect_ticks_max  = forg.get("collect_ticks",          30)
        self.deposit_ticks_max  = forg.get("deposit_ticks",          15)

        # ── FSM state ─────────────────────────────────────────────────── #
        self.fsm                  = self.EXPLORE
        self.base_pos             = (pos[0], pos[1])
        self.levy_waypoint        = self._new_levy_waypoint()
        self.levy_steps_remaining = 0
        self.target_resource_idx: Optional[int] = None
        self.collect_timer        = 0
        self.deposit_timer        = 0
        self.resources_carried    = 0
        self.score                = 0

        # ── Crypto ───────────────────────────────────────────────────── #
        self.ecdh_session    = ECDHSession()
        self.channels:       Dict[str, SecureChannel] = {}
        self._raw_inbox:     List[Tuple[int, bytes]]  = []
        self._msg_inbox:     List[dict]               = []

        # ── Reasoning ────────────────────────────────────────────────── #
        self.bdi = BDIEngine(
            agent_id=self.agent_str,
            obstacles=obstacles,
            max_velocity=self.max_vel,
            pos_tolerance=self.pos_tol,
            goals=goals,
        )

        # ── Adaptation ───────────────────────────────────────────────── #
        self.feature_extractor = FeatureExtractor()
        # Per-peer DBN classifiers (created on first contact)
        self.dbn: Dict[str, DynamicBayesianClassifier] = {}

        tcfg       = cfg.get("trust", {})
        self.trust = BetaTrustEngine(
            alpha0=tcfg.get("alpha0", 4.0),
            beta0 =tcfg.get("beta0",  1.0),
            aging =tcfg.get("aging",  0.97),
        )

        # ── RL Meta-Controller ────────────────────────────────────────── #
        meta_cfg = tcfg.get("meta_controller", {})
        if meta_cfg.get("enabled", True):
            m_path = meta_cfg.get("model_path", "models/ppo_trust_controller.zip")
            self.meta_controller: Optional[PPOMetaController] = PPOMetaController(
                model_path=m_path,
                base_threshold=tcfg.get("reroute_threshold", 0.40),
            )
        else:
            self.meta_controller = None
        self._last_dbn: Dict[str, dict] = {}

        # ── Planner ───────────────────────────────────────────────────── #
        self.replan_every = cfg["simulation"].get("replan_every_n_ticks", 5)
        obs_cells         = self._rasterize(obstacles)
        self.planner      = TrustDStarLite(
            width=self.W, height=self.H,
            obstacles=obs_cells,
            reroute_threshold=tcfg.get("reroute_threshold", 0.40),
            hazard_radius=12,
            sensitivity=tcfg.get("dstar_sensitivity", 7.0),
        )
        self.planner.init(
            (int(pos[0]), int(pos[1])),
            (int(self.levy_waypoint[0]), int(self.levy_waypoint[1])),
        )
        self._current_path: List[Tuple[int,int]] = []

        # ── Telemetry ─────────────────────────────────────────────────── #
        self.attack_detections = 0
        # Per-peer intent posteriors (most recent) for arena display
        self.intent_posteriors: Dict[str, Dict[str,float]] = {}

        # Respawn ticks set externally
        self._respawn_ticks: int = 250

    # ── Lévy flight ──────────────────────────────────────────────────── #

    def _levy_step_size(self) -> float:
        raw = levy_mantegna(self.levy_alpha, self.rng)
        return min(raw * 22.0, 280.0)

    def _new_levy_waypoint(self) -> Tuple[float, float]:
        """
        Sample a new Lévy-flight waypoint at a heavy-tailed distance in a
        uniformly random direction.  This produces the characteristic
        mix of short local steps and rare long-range jumps.
        """
        step  = self._levy_step_size()
        angle = self.rng.uniform(0.0, 2 * math.pi)
        wx    = max(20.0, min(self.W - 20.0, self.state.x + step * math.cos(angle)))
        wy    = max(20.0, min(self.H - 20.0, self.state.y + step * math.sin(angle)))
        return (wx, wy)

    # ── FSM ──────────────────────────────────────────────────────────── #

    def update_fsm(self, tick: int) -> Tuple[float, float]:
        """Advance the foraging FSM one tick; returns the movement target."""
        px, py = self.state.x, self.state.y

        if self.fsm == self.EXPLORE:
            # Check for nearby resources
            for i, res in enumerate(self.resources):
                if not res.available:
                    continue
                if math.hypot(res.x - px, res.y - py) < self.detect_radius:
                    self.target_resource_idx = i
                    self.fsm                 = self.APPROACH
                    return (res.x, res.y)
            # Lévy-flight exploration
            d = math.hypot(self.levy_waypoint[0] - px, self.levy_waypoint[1] - py)
            if d < 14.0 or self.levy_steps_remaining <= 0:
                self.levy_waypoint        = self._new_levy_waypoint()
                self.levy_steps_remaining = max(1, int(
                    math.hypot(self.levy_waypoint[0] - px,
                               self.levy_waypoint[1] - py) / self.max_vel
                ) + self.rng.randint(5, 25))
            self.levy_steps_remaining -= 1
            return self.levy_waypoint

        if self.fsm == self.APPROACH:
            if self.target_resource_idx is None:
                self.fsm = self.EXPLORE
                return self._new_levy_waypoint()
            res = self.resources[self.target_resource_idx]
            if not res.available:
                self.target_resource_idx = None
                self.fsm                 = self.EXPLORE
                return self._new_levy_waypoint()
            if math.hypot(res.x - px, res.y - py) < res.radius:
                self.fsm           = self.COLLECT
                self.collect_timer = self.collect_ticks_max
            return (res.x, res.y)

        if self.fsm == self.COLLECT:
            self.collect_timer -= 1
            if self.collect_timer <= 0:
                res               = self.resources[self.target_resource_idx]
                res.available     = False
                res.respawn_at    = tick + self._respawn_ticks
                self.resources_carried += 1
                self.fsm          = self.RETURN
            return (px, py)

        if self.fsm == self.RETURN:
            if math.hypot(self.base_pos[0] - px, self.base_pos[1] - py) < 25.0:
                self.fsm           = self.DEPOSIT
                self.deposit_timer = self.deposit_ticks_max
            return self.base_pos

        if self.fsm == self.DEPOSIT:
            self.deposit_timer -= 1
            if self.deposit_timer <= 0:
                self.score             += self.resources_carried
                self.resources_carried  = 0
                self.fsm                = self.EXPLORE
                self.levy_waypoint      = self._new_levy_waypoint()
                self.levy_steps_remaining = 0
            return self.base_pos

        return self.levy_waypoint

    # ── Continuous-angle omnidirectional movement ─────────────────────── #

    def steer_toward(
        self,
        target:           Tuple[float, float],
        avoidance_vector: Tuple[float, float] = (0.0, 0.0),
        avoidance_weight: float               = 0.35,
    ) -> None:
        """
        Steer toward target with optional trust-avoidance bias.

        Motion model:
          effective_target = target + avoidance_weight * avoidance_vector
          desired_heading  = atan2(ety - y, etx - x) + Gaussian(0, 0.012)
          heading_delta    = clamp(error, ±max_angular_velocity)
          speed            = min(max_velocity, distance_to_effective_target)

        The heading is a continuous angle in [0, 2π) so the robot can move
        in ANY direction at any angle — not grid-locked.
        """
        tx, ty = target
        # Blend in avoidance
        ax, ay  = avoidance_vector
        etx     = tx + ax * avoidance_weight
        ety     = ty + ay * avoidance_weight
        # Clamp effective target to arena bounds
        etx     = max(5.0, min(self.W - 5.0, etx))
        ety     = max(5.0, min(self.H - 5.0, ety))

        dx, dy = etx - self.state.x, ety - self.state.y
        dist   = math.hypot(dx, dy)

        if dist < 0.5:
            self.state.velocity = 0.0
            return

        # Desired heading toward effective target, with small steering noise
        desired = math.atan2(dy, dx) + self.rng.gauss(0.0, 0.012)
        # Angular error in [-π, π]
        delta   = (desired - self.state.heading + math.pi) % (2 * math.pi) - math.pi
        delta   = max(-self.max_ang, min(self.max_ang, delta))
        self.state.heading = self.state.heading + delta   # NOT clamped: full 2π

        speed         = min(self.max_vel, dist)
        self.state.x  = max(0.0, min(self.W - 1.0,
                            self.state.x + speed * math.cos(self.state.heading)))
        self.state.y  = max(0.0, min(self.H - 1.0,
                            self.state.y + speed * math.sin(self.state.heading)))
        self.state.velocity = speed

    # ── Broadcast payload ─────────────────────────────────────────────── #

    def build_payload(self, tick: int) -> dict:
        """
        Build the broadcast state payload.
        Positions include Gaussian sensor noise (σ = sensor_noise_std px).
        This means the DBN must tolerate single noisy observations without
        immediately flagging them as attacks — persistent anomalies are needed.
        """
        return {
            "sender_id": self.agent_id,
            "id":        self.agent_id,
            "x":         self.state.x + self.rng.gauss(0.0, self.sensor_noise_std),
            "y":         self.state.y + self.rng.gauss(0.0, self.sensor_noise_std),
            "heading":   self.state.heading + self.rng.gauss(0.0, 0.03),
            "velocity":  self.state.velocity,
            "status":    self.fsm,
            "tick":      tick,
        }

    # ── Decode inbox ──────────────────────────────────────────────────── #

    def decode_inbox(self) -> None:
        """Decrypt raw_inbox bytes into msg_inbox dicts."""
        for (src_id, raw) in self._raw_inbox:
            peer_str = f"A{src_id}"
            if peer_str in self.channels:
                try:
                    msg = self.channels[peer_str].decrypt_state(raw)
                    self._msg_inbox.append(msg)
                except Exception as e:
                    self._msg_inbox.append({
                        "CORRUPTED": True, "sender_id": src_id,
                        "error": str(e), "tick": -1,
                    })
            else:
                self._msg_inbox.append({
                    "CORRUPTED": True, "sender_id": src_id,
                    "error": "no channel", "tick": -1,
                })
        self._raw_inbox.clear()

    # ── Cognition loop ────────────────────────────────────────────────── #

    def process_cognition(self, tick: int, ledger: StructuredEventLedger) -> List[dict]:
        """
        Full BDI → Feature Extract → DBN → Trust Update pipeline.

        Returns list of per-event update dicts (for caller logging).
        All math traces are logged directly to the ledger from here.
        """
        if not self._msg_inbox:
            return []

        epistemic_events = self.bdi.update_beliefs(self._msg_inbox)
        self._msg_inbox.clear()
        updates = []

        for ev in epistemic_events:
            pid    = ev["peer_id"]
            delta  = ev.get("delta",         0.0)
            clash  = ev.get("contradiction", False)
            reason = ev.get("reason",        "")
            intent = ev.get("intent_post",   {})

            # Store intent posteriors for arena display
            self.intent_posteriors[pid] = intent

            # Log epistemic paradox to ledger
            if clash:
                ledger.log_epistemic_paradox(
                    tick        = tick,
                    evaluator   = self.agent_str,
                    peer_id     = pid,
                    delta       = delta,
                    reason      = reason,
                    explanation = ev.get("explanation", ""),
                    intent_posteriors = intent,
                    smt_reason  = reason,
                )

            # ── Feature extraction ────────────────────────────────────── #
            features = self.feature_extractor.extract(pid, ev)

            # ── DBN forward filter ────────────────────────────────────── #
            if pid not in self.dbn:
                self.dbn[pid] = DynamicBayesianClassifier()
            dbn_result = self.dbn[pid].classify(pid, features)
            self._last_dbn[pid] = {**dbn_result, "delta": delta}
            label      = dbn_result["label"]
            p_atk      = dbn_result["p_attack"]
            p_flt      = dbn_result["p_fault"]

            # Log full DBN math trace
            ledger.log_dbn_step(tick, self.agent_str, pid, dbn_result)

            # ── Verdict resolution ────────────────────────────────────── #
            if "CRYPTO" in reason:
                verdict, p_atk = "ATTACK", 1.0
                ledger.log_crypto_fault(tick, self.agent_str, pid, reason)
            elif label == "ATTACK":
                verdict = "ATTACK"
                self.attack_detections += 1
            elif label == "FAULT":
                verdict = "FAULT"
            else:
                verdict = "HONEST"

            # ── Beta trust update ─────────────────────────────────────── #
            upd = self.trust.update(pid, p_atk, delta, verdict)
            ledger.log_trust_update(tick, self.agent_str, upd)

            updates.append({**ev, **upd, "label": label, "dbn_result": dbn_result})

        return updates

    def replan(self, tick: int, ledger: StructuredEventLedger) -> None:
        """
        Incremental D* Lite replan.  Only runs every replan_every ticks.
        Logs cost delta and avoidance decisions to the ledger.
        """
        if tick % self.replan_every != 0:
            return
        trusts = self.trust.all_trusts()
        if not trusts:
            return
        peer_pos = {}
        for pid in trusts:
            b = self.bdi.kb.query(pid, "position")
            if b:
                peer_pos[pid] = (int(b.value[0]), int(b.value[1]))
        if peer_pos:
            peer_thresholds = {}
            if self.meta_controller:
                for pid in peer_pos:
                    t_score, unc = self.trust.get(pid)
                    last_d = self._last_dbn.get(pid, {})
                    p_atk = last_d.get("p_attack", 0.05)
                    d_val = last_d.get("delta", 0.0)
                    peer_thresholds[pid] = self.meta_controller.predict_threshold(
                        trust=t_score,
                        uncertainty=unc,
                        p_attack=p_atk,
                        anomaly_delta=d_val,
                    )

            path = self.planner.replan(
                (int(self.state.x), int(self.state.y)),
                trusts, peer_pos,
                peer_thresholds=peer_thresholds if peer_thresholds else None,
            )
            self._current_path = path
            # Log replan event
            stats = self.planner.last_replan_stats
            if stats.get("n_cells_changed", 0) > 0 or stats.get("peers_avoided"):
                ledger.log_replan(tick, self.agent_str, "TRUST_CHANGE", stats)

    # ── Util ──────────────────────────────────────────────────────────── #

    def _rasterize(
        self,
        obstacles: List[Tuple[float, float, float, float]],
        step:      int = 4,
    ) -> List[Tuple[int,int]]:
        cells = []
        for (ox, oy, ow, oh) in obstacles:
            for ix in range(int(ox), int(ox + ow), step):
                for iy in range(int(oy), int(oy + oh), step):
                    cells.append((ix, iy))
        return cells


# ═══════════════════════════════════════════════════════════════════════ #
#  ECDH key bootstrap                                                      #
# ═══════════════════════════════════════════════════════════════════════ #

def bootstrap_keys(agents: List[AgentNode]) -> None:
    """Full O(n²) pairwise ECDH key derivation (SECP256R1 + HKDF-SHA256)."""
    last_shared = None
    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i == j:
                continue
            shared = ai.ecdh_session.derive_shared(aj.ecdh_session.pub_bytes())
            ai.channels[f"A{j}"] = SecureChannel(shared)
            last_shared = shared
    if last_shared:
        print(f"[CRYPTO] ECDH keys derived (SECP256R1+HKDF). Preview: {last_shared.hex()[:16]}...")


# ═══════════════════════════════════════════════════════════════════════ #
#  Insider channel bootstrap (for MITM forge with correct AES-GCM tags)   #
# ═══════════════════════════════════════════════════════════════════════ #

def bootstrap_mitm_channels(agents: List[AgentNode]) -> Dict:
    """
    Build a (src_id, dst_id) → SecureChannel map for the MITM proxy.
    The proxy uses these to forge position packets with valid AES-GCM tags,
    simulating an insider attacker who has compromised the fleet key.
    """
    channels = {}
    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i == j:
                continue
            # The proxy shares the same derived key as agent i sending to agent j
            shared = ai.ecdh_session.derive_shared(aj.ecdh_session.pub_bytes())
            channels[(i, j)] = SecureChannel(shared)
    return channels


# ═══════════════════════════════════════════════════════════════════════ #
#  Main async simulation loop                                              #
# ═══════════════════════════════════════════════════════════════════════ #

async def run_simulation(
    config_path: str,
    no_display:  bool = False,
    max_ticks:   Optional[int] = None,
) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())

    # ── Random seed ──────────────────────────────────────────────────── #
    master_seed: Optional[int] = cfg["simulation"].get("random_seed", None)
    if master_seed is None:
        master_seed = random.randint(0, 2**32 - 1)
    random.seed(master_seed)
    print(f"[SIM] Random seed: {master_seed}  "
          f"(set random_seed in scenario.yaml to reproduce)")

    # ── Shared resources ─────────────────────────────────────────────── #
    respawn_ticks = cfg.get("foraging", {}).get("resource_respawn_ticks", 250)
    resources: List[Resource] = [
        Resource(x=r[0], y=r[1], radius=r[2])
        for r in cfg["environment"].get("resources", [])
    ]

    # ── Environment ──────────────────────────────────────────────────── #
    raw_obs   = cfg["environment"].get("obstacles", [])
    obstacles = [(o[0], o[1], o[2], o[3]) for o in raw_obs]
    raw_goals = cfg["environment"].get("goals", {})
    goals     = {k: tuple(v) for k, v in raw_goals.items()}

    # ── Agents ───────────────────────────────────────────────────────── #
    agents: List[AgentNode] = []
    for i in range(cfg["fleet"]["agent_count"]):
        sp = cfg["fleet"]["start_positions"][i]
        a  = AgentNode(
            agent_id    = i,
            pos         = (sp[0], sp[1], sp[2]),
            obstacles   = obstacles,
            goals       = goals,
            resources   = resources,
            cfg         = cfg,
            master_seed = master_seed,
        )
        a._respawn_ticks = respawn_ticks
        agents.append(a)
    print(f"[SIM] {len(agents)} agents initialized.")

    bootstrap_keys(agents)
    mitm_channels = bootstrap_mitm_channels(agents)

    # ── MITM proxy ───────────────────────────────────────────────────── #
    atk_cfg = yaml.safe_load(
        (ROOT / "config" / "attack_schedule.yaml").read_text()
    )
    proxy = MITMProxy(
        listen_port     = 9999,
        forward_ports   = {i: 9000 + i for i in range(len(agents))},
        attack_schedule = atk_cfg.get("attacks", []),
        channels        = mitm_channels,
    )
    await proxy.start()
    print("[MITM] Proxy started on port 9999")

    # ── UDP transport per agent ───────────────────────────────────────── #
    import socket as _socket

    transports: List = []
    for agent in agents:
        loop = asyncio.get_running_loop()
        sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        sock.setblocking(False)
        sock.bind(("127.0.0.1", 9000 + agent.agent_id))

        class _Proto(asyncio.DatagramProtocol):
            def __init__(self, a: AgentNode):
                self._agent = a
            def datagram_received(self, data: bytes, addr):
                try:
                    env    = json.loads(data.decode("utf-8"))
                    src_id = int(env.get("sender_id", -1))
                    raw    = base64.b64decode(env.get("payload", ""))
                    if src_id >= 0 and src_id != self._agent.agent_id:
                        self._agent._raw_inbox.append((src_id, raw))
                except Exception:
                    pass
            def error_received(self, exc): pass

        transport, _ = await loop.create_datagram_endpoint(
            lambda a=agent: _Proto(a), sock=sock
        )
        transports.append(transport)

    # ── Logger ───────────────────────────────────────────────────────── #
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ledger  = StructuredEventLedger(str(log_dir / "execution_ledger.jsonl"))
    ledger.log_simulation_start(
        tick=0, n_agents=len(agents), max_ticks=cfg["simulation"].get("ticks", 2000),
        use_ecdh=cfg.get("security_profile", {}).get("use_ecdh", True),
        seed=master_seed,
        cfg_summary={
            "ticks_per_second":   cfg["simulation"].get("ticks_per_second", 20),
            "levy_alpha":         cfg.get("foraging", {}).get("levy_alpha", 1.5),
            "sensor_noise_std":   cfg.get("foraging", {}).get("sensor_noise_std", 1.8),
            "n_resources":        len(resources),
        },
    )

    # ── Optional renderer ─────────────────────────────────────────────── #
    renderer = None
    if not no_display and PYGAME_OK:
        try:
            renderer = Arena(
                bounds    = (cfg["environment"]["width"], cfg["environment"]["height"]),
                obstacles = obstacles,
                resources = [[r.x, r.y, r.radius] for r in resources],
            )
        except Exception as e:
            print(f"[DISPLAY] Pygame init failed: {e}. Running headless.")

    # ── Attack-window lookup (ground truth) ───────────────────────────── #
    attack_windows = [
        (a["start_tick"], a["start_tick"] + a["duration"],
         a.get("attacker_id", -1), a.get("target_id", -1), a.get("mode", ""))
        for a in atk_cfg.get("attacks", [])
    ]

    def in_attack_window(tick: int):
        """Returns (is_active, attacker_id, target_id, mode) or (False, -1, -1, '')"""
        for s, e, att, tgt, mode in attack_windows:
            if s <= tick < e:
                return True, att, tgt, mode
        return False, -1, -1, ""

    # ── Main simulation loop ──────────────────────────────────────────── #
    total_ticks  = max_ticks if max_ticks is not None else cfg["simulation"].get("ticks", 2000)
    print_every  = max(50, total_ticks // 40)
    hazard_every = 100   # log hazard snapshot every N ticks

    for tick in range(total_ticks):
        proxy.set_tick(tick)

        # Resource respawn
        for res in resources:
            if not res.available and tick >= res.respawn_at:
                res.available = True

        # Log attack window ground truth (for dashboard overlay)
        atk_active, atk_id, tgt_id, atk_mode = in_attack_window(tick)
        if atk_active and tick % 10 == 0:
            ledger.log_attack_window(tick, atk_id, tgt_id, atk_mode)

        # ── Move agents (FSM + continuous-angle steering) ─────────────── #
        for agent in agents:
            target = agent.update_fsm(tick)
            # Continuous-space avoidance vector from trust hazard field
            avoidance = agent.planner.get_avoidance_vector((agent.state.x, agent.state.y))
            agent.steer_toward(target, avoidance_vector=avoidance)
            agent.state.status = agent.fsm
            agent.state.tick   = tick

        # ── Broadcast (encrypt → proxy) ───────────────────────────────── #
        for sender in agents:
            payload = sender.build_payload(tick)
            for receiver in agents:
                if receiver.agent_id == sender.agent_id:
                    continue
                peer_str = f"A{receiver.agent_id}"
                if peer_str in sender.channels:
                    try:
                        raw      = sender.channels[peer_str].encrypt_state(payload)
                        envelope = json.dumps({
                            "sender_id": sender.agent_id,
                            "dest_id":   receiver.agent_id,
                            "payload":   base64.b64encode(raw).decode("utf-8"),
                        }).encode("utf-8")
                        transports[sender.agent_id].sendto(
                            envelope, ("127.0.0.1", 9999)
                        )
                    except Exception:
                        pass

        # Brief yield for UDP callbacks to fire
        await asyncio.sleep(0.0)

        # ── Cognition ────────────────────────────────────────────────── #
        for agent in agents:
            agent.decode_inbox()
            agent.process_cognition(tick, ledger)
            agent.replan(tick, ledger)

        # ── Periodic hazard snapshot ──────────────────────────────────── #
        if tick % hazard_every == 0:
            for agent in agents:
                snap = agent.planner.get_hazard_snapshot()
                if snap:
                    ledger.log_hazard_snapshot(tick, agent.agent_str, snap)

        # ── Render ───────────────────────────────────────────────────── #
        if renderer:
            agent_states = {a.agent_id: a.state for a in agents}
            trust_map_flat = agents[0].trust.all_trusts()
            trust_map_flat[agents[0].agent_str] = 1.0

            still_open = renderer.render(
                agents            = agent_states,
                trust_map         = trust_map_flat,
                tick              = tick,
                attack_active     = atk_active,
                paths             = {a.agent_id: a._current_path for a in agents},
                hazard_fields     = {agents[0].agent_id: agents[0].planner.get_hazard_snapshot()},
                intent_posteriors = {a.agent_id: a.intent_posteriors for a in agents},
                goals             = goals,
                mitm_stats        = proxy.stats,
                scores            = {a.agent_id: a.score for a in agents},
                fsm_states        = {a.agent_id: a.fsm   for a in agents},
                resource_states   = [r.available for r in resources],
            )
            if not still_open:
                break

        # ── Heartbeat ────────────────────────────────────────────────── #
        if tick % print_every == 0:
            t_snap = {
                f"A{j}": f"{agents[0].trust.get_trust(f'A{j}'):.2f}"
                for j in range(1, len(agents))
            }
            scores  = {f"A{a.agent_id}": a.score for a in agents}
            flag    = "[!] ATTACK" if atk_active else "          "
            avail   = sum(1 for r in resources if r.available)
            print(
                f"[SIM] Tick {tick:04d}  {flag}  "
                f"MITM: {proxy.stats}  "
                f"A0->trust: {t_snap}  "
                f"scores: {scores}  "
                f"resources: {avail}/{len(resources)}"
            )

    # ── Shutdown ─────────────────────────────────────────────────────── #
    for t in transports:
        t.close()
    proxy.stop()
    if renderer:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass

    # ── Final summary ─────────────────────────────────────────────────── #
    print(f"\n{'='*72}")
    print(f"[SIM] Finished {total_ticks} ticks   Seed: {master_seed}")
    print(f"[SIM] MITM final: {proxy.stats}")
    print(f"\n[SCORES] Resources collected:")
    for a in agents:
        print(f"         A{a.agent_id}: {a.score:2d} resources  "
              f"| attack detections: {a.attack_detections}")
    print(f"\n[TRUST]  Final trust matrix (A0's perspective):")
    for j in range(1, len(agents)):
        state = agents[0].trust.full_state().get(f"A{j}", {})
        print(
            f"         A{j}: trust={state.get('trust', 0):.3f}  "
            f"unc={state.get('uncertainty', 0):.4f}  "
            f"alpha={state.get('alpha', 0):.2f}  beta={state.get('beta', 0):.2f}  "
            f"min_ever={state.get('min_trust_ever', 0):.3f}  "
            f"tier={state.get('grudge_tier', 'N/A')}"
        )
    print(f"{'='*72}\n")

    summary = {
        "total_ticks":  total_ticks,
        "seed":         master_seed,
        "mitm_stats":   proxy.stats,
        "scores":       {f"A{a.agent_id}": a.score for a in agents},
        "detections":   {f"A{a.agent_id}": a.attack_detections for a in agents},
        "final_trust":  agents[0].trust.all_trusts(),
    }
    ledger.log_simulation_end(total_ticks, summary)
    ledger.close()


# ═══════════════════════════════════════════════════════════════════════ #
#  Entry point                                                             #
# ═══════════════════════════════════════════════════════════════════════ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic Epistemic Foraging — Research Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                          # Pygame arena (default)
  python -m src.main --no-display             # Headless (faster)
  python -m src.main --config config/scenario.yaml
  streamlit run src/dashboard/app.py          # Live research dashboard
        """,
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "scenario.yaml"),
        help="Path to scenario YAML configuration file.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless (no Pygame window) — faster for batch experiments.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=None,
        help="Override total simulation ticks.",
    )
    args = parser.parse_args()
    asyncio.run(run_simulation(
        args.config,
        no_display=args.no_display,
        max_ticks=args.max_ticks,
    ))


if __name__ == "__main__":
    main()
