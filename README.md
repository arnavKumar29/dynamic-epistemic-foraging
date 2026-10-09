# Dynamic Epistemic Foraging

**Multi-agent cognitive architecture for trust-aware navigation under active network attacks.**

A simulation where four autonomous robots must navigate a 2D arena while:
- Broadcasting encrypted position states over a shared network.
- Detecting Man-in-the-Middle attacks using **formal SMT verification** and **Bayesian Theory of Mind**.
- Updating **epistemic trust scores** using a Beta-Binomial Dirichlet reputation system.
- Replanning paths in real-time using **D\* Lite** to route around distrusted agents.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DOMAIN 1: NETWORK                     │
│  ECDH (SECP256R1) + AES-256-GCM + Replay Guard + MITM   │
└────────────────────────────┬────────────────────────────┘
                             │ Encrypted UDP broadcasts
┌────────────────────────────▼────────────────────────────┐
│                   DOMAIN 2: REASONING                     │
│  Belief → BDI Engine → Bayesian ToM + Z3 SMT Checker     │
│                 (Epistemic Clash Detection)               │
└────────────────────────────┬────────────────────────────┘
                             │ EpistemicEvent{contradiction, delta, intent}
┌────────────────────────────▼────────────────────────────┐
│                  DOMAIN 3: ADAPTATION                     │
│  Feature Extractor → 2-slice DBN → Beta Trust Engine      │
│              → D* Lite Trust-Weighted Planner             │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies
```bash
cd dynamic-epistemic-foraging
pip install -r requirements.txt
```

### 2. Run the simulation (Pygame window)
```bash
python -m src.main
```

### 3. Run headless (no display)
```bash
python -m src.main --no-display
```

### 4. Start the live Streamlit dashboard (in a second terminal)
```bash
streamlit run src/dashboard/app.py
```
Then open http://localhost:8501 in your browser.

### 5. Run the test suite
```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
dynamic-epistemic-foraging/
├── config/
│   ├── scenario.yaml          # Arena, agent, trust, crypto configuration
│   └── attack_schedule.yaml   # Scripted MITM attack windows
├── src/
│   ├── main.py                # ← Simulation entry point
│   ├── environment/
│   │   ├── agent_state.py     # Agent dataclass with serialization
│   │   └── arena.py           # Pygame renderer (trust-colored agents)
│   ├── network/
│   │   ├── sdes.py            # S-DES (educational cipher)
│   │   ├── secure_comms.py    # ECDH + AES-256-GCM + replay guard
│   │   ├── secure_broadcast.py# AsyncIO UDP broadcast
│   │   └── mitm_proxy.py      # Scripted MITM (forge / replay / drop)
│   ├── reasoning/
│   │   ├── belief.py          # Belief dataclass + AGM knowledge base
│   │   ├── smt_checker.py     # Z3 SMT kinematic proof
│   │   ├── theory_of_mind.py  # Bayesian ToM + Boltzmann intent inference
│   │   └── bdi_engine.py      # BDI wiring (beliefs → ToM → events)
│   ├── adaptation/
│   │   ├── feature_extractor.py # Discretize anomaly observations
│   │   ├── bayesian_network.py  # 2-slice Dynamic Bayesian Network
│   │   ├── trust_manager.py     # Beta-Binomial trust with decay
│   │   └── path_planner.py      # D* Lite + trust hazard field
│   └── dashboard/
│       ├── logger.py          # JSONL structured event ledger
│       └── app.py             # Streamlit live dashboard
└── tests/
    ├── test_sdes.py
    ├── test_smt.py
    ├── test_network.py
    ├── test_reasoning.py
    ├── test_trust.py
    └── test_planner.py
```

---

## Key Design Decisions (Rev. 02 vs Rev. 01)

| Component | Rev. 01 | Rev. 02 (This Build) |
|---|---|---|
| Key Distribution | Hardcoded AES key | ECDH SECP256R1 + HKDF + replay guard |
| Contradiction detection | if dist > v*dt threshold | **Z3 SMT formal proof** |
| Theory of Mind | Physics check only | **Bayesian ToM + Boltzmann intent inference** |
| Fault vs. Attack | Static 1-slice Bayes | **Dynamic 2-slice DBN (temporal memory)** |
| Trust update | Scalar delta rule | **Beta-Binomial + temporal decay (Josang)** |
| Path planning | Static A* | **D\* Lite (incremental repair on trust changes)** |
| Explainability | Print statement | **JSONL ledger + Streamlit live dashboard** |

---

## The Six-Step Cognitive Loop

Each tick, every agent runs:

1. **Sense** — Register own position from sensors into knowledge base.
2. **Receive** — Decrypt and authenticate incoming UDP packets from peers.
3. **Translate** — Convert raw messages into symbolic `Belief` atoms.
4. **Verify (Epistemic Clash)** — Run `BayesianToM` (Z3 SMT + intent inference) on position claims.
5. **Classify** — Pass clash features through `DynamicBayesianClassifier` (DBN forward filter).
6. **Adapt** — Update `BetaTrustEngine` → Replan via `TrustDStarLite`.

---

## Key Observable Behaviors

Running the simulation for 500 ticks:

1. **Ticks 0–39** — All agents trusted. Straight-line navigation.
2. **Ticks 40–160** — Attack 1 active. Agent A2 forges A0's position (+160px offset).
   - Z3 SMT detects kinematic violation within 1-2 ticks of the attack.
   - DBN posterior P(Attack|A2) rises from 0.05 → 0.70+ over ~5 ticks.
   - Trust(A2) drops from 0.80 → below 0.40 by tick 55.
   - Other agents visually reroute: red hazard aura appears around A2's zone.
3. **Ticks 200–280** — Attack 2 active (replay from A1 targeting A3).
   - Replay attack blocked by monotonic tick guard → `CRYPTO_AUTHENTICATION_FAILURE` event.
   - Trust(A1) drops immediately on first crypto failure.
4. **Dashboard** — Live trust timeline shows trust recovery rates.
   Recovery is slow (Beta distribution: forgiveness is slow), punishments fast.

---

## Configuration

### `config/scenario.yaml`
Controls arena dimensions, obstacle positions, agent starting positions, kinematic limits,
ECDH/AES settings, and trust hyperparameters (α₀, β₀, aging factor).

### `config/attack_schedule.yaml`
Defines scripted MITM attacks:
- **position_forge**: Add offset to victim's reported position.
- **replay**: Buffer and re-emit packets after N ticks.
- **drop**: Silently discard packets (silent DoS).

---

## Cryptographic Architecture

```
Session setup (once at startup):
  For each agent pair (i, j):
    ecdh_i = ECDHSession()
    ecdh_j = ECDHSession()
    shared_key = HKDF-SHA256(ecdh_i.ECDH(ecdh_j.pub))

Per-message (AES-256-GCM):
  nonce = os.urandom(12)
  ciphertext || tag = AES-GCM-Encrypt(key, nonce, payload)
  packet = nonce || ciphertext || tag

Replay guard (SecureChannel):
  Seen nonces → reject duplicates
  Monotonic tick per sender → reject out-of-order or repeated ticks
```

---

## Academic Defense Notes

**Q: Why not just threshold the kinematic velocity?**
> Thresholds are ad-hoc — any attacker who knows the threshold can stay just below it.
> Z3 SMT provides a **formally verified** proof that the claimed position is physically
> unreachable given the agent's rated velocity and the time elapsed.

**Q: Why DBN instead of a static Bayesian network?**
> A static Bayes net has no memory. A single packet loss (random glitch) would
> spike the attack posterior to 0.6+, causing false positives. The 2-slice DBN
> remembers that attacks **persist** across ticks (P(A_t=1|A_{t-1}=1)=0.90),
> while glitches don't (P(F_t=1|F_{t-1}=0)=0.08). Only sustained contradictions
> convince the DBN this is a real attack.

**Q: Why Beta-Binomial trust instead of a simple counter?**
> Beta-Binomial gives **uncertainty quantification** (the variance of the Beta
> distribution tells you how confident you are in the trust score), **temporal
> decay** (old evidence ages out), and an asymmetric update (fast to punish,
> slow to forgive — behaviorally realistic).

**Q: What does D\* Lite buy you over A\*?**
> When trust changes (which happens every tick), A\* requires a full O(n log n)
> re-search of the entire grid. D\* Lite only re-expands vertices whose edge
> costs changed — typically O(k) where k is the small number of hazard cells
> that changed. At 20 ticks/second with 4 agents, this is the difference between
> 50ms/tick and 0.5ms/tick.

---

## Dependencies

- `cryptography` — ECDH, AES-256-GCM
- `z3-solver` — SMT formal verification
- `pgmpy` — Bayesian network (available for validation; forward filter is native)
- `networkx` — Graph utilities
- `pygame` — Visualization
- `streamlit` + `plotly` — Live dashboard
- `pyyaml` — Configuration

- `stable-baselines3`, `torch`, `gymnasium` — PPO reinforcement learning meta-controller for dynamic trust reroute threshold adaptation

### PPO Meta-Controller Training

To retrain the PPO policy weights:
```bash
python -m src.adaptation.ppo_meta_controller --train --timesteps 40000 --save models/ppo_trust_controller.zip
```
Model weights are saved to `models/ppo_trust_controller.zip` and automatically loaded at runtime by the fleet agents.
