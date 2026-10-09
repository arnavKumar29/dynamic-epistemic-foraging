# Dynamic Epistemic Foraging (DEF): Architectural, Mathematical, and Algorithmic Specification

---

## 1. Executive Summary & Research Motivation

Multi-agent autonomous systems (e.g., drone swarms, automated warehouse rovers, robotic search-and-rescue teams) critically rely on continuous inter-agent telemetry to coordinate spatial dispersion, avoid mutual collisions, and allocate shared environmental targets. Standard multi-robot frameworks operate under two fragile assumptions:
1. **Sensory Honesty**: Communicated telemetry is assumed to reflect genuine physical state measurements.
2. **Binary Fault Models**: Anomalies are treated as either standard Gaussian noise or binary network dropouts.

In adversarial or degraded real-world environments, these assumptions fail catastrophically:
- **Active Adversaries**: Compromised or rogue agents (insiders possessing valid cryptographic keys) or Man-in-the-Middle (MITM) attackers can inject falsified telemetry (e.g., phantom obstacle claims, spoofed self-positions, delayed replays).
- **Silent Degradation**: Hardware failures (e.g., encoder slippage, degraded IMUs, motor under-voltage) create kinematic deviations that mimic malicious deception.
- **Explainability Deficit in Machine Learning**: Naive deep learning or statistical outlier detectors (e.g., Isolation Forests, Mahalanobis distance, autoencoders) flag that a data point is unusual, but cannot explain **why** it occurred, **who** caused it, or **what intent** underlies the deviation.

**Dynamic Epistemic Foraging (DEF)** resolves this fundamental gap by formulating multi-robot coordination as an **Epistemic Reasoning and Causal Attribution Problem**. Agents do not simply track coordinates; they maintain formal symbolic beliefs, deduce the physical plausibility of peer reports via First-Order Satisfiability Modulo Theories (SMT), infer peer intentions through Bayesian Theory of Mind (ToM), causally isolate attacks from hardware faults using a Dynamic Bayesian Network (DBN), maintain evidence-grounded Dirichlet trust distributions with asymmetric punishment and ratchet grudge memory, and dynamically deform their continuous navigation trajectories away from distrusted spatial zones via trust-modulated D* Lite incremental replanning.

---

## 2. Global Execution Loop & Information Flow

The DEF architecture is divided into three interconnected cognitive domains executing over discrete simulation ticks $t \in \{0, 1, 2, \dots, T\}$ at frequency $f_{\text{sim}} = 20\text{ Hz}$ ($\Delta t = 0.05\text{ s}$):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DOMAIN 1: NETWORK LAYER                                │
│                                                                                        │
│   Agent i State  ───────►  ECDH + HKDF  ───────►  AES-256-GCM  ───────►  UDP Socket   │
│   (x, y, θ, v, tick)        (SECP256R1)            (12B IV, 16B Tag)       (Port 9000+i) │
│                                                          ▲                             │
│                                                          │ Intercept & Corrupt         │
│                                                    MITM Proxy                          │
│                                             (Forge, Replay, Drop)                      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Encrypted Datagrams
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                DOMAIN 2: REASONING LAYER                               │
│                                                                                        │
│   Replay Guard (Δtick > 0)  ───────►  AGM Belief Revision  ───────►  Z3 SMT Checker    │
│                                       (KnowledgeBase)             (Kinematic Proof)    │
│                                                                            │           │
│                                                                            ▼           │
│   Epistemic Event  ◄────────────────────────────────────────  Bayesian Theory of Mind  │
│   {anomaly, delta, intent_posteriors}                          (Boltzmann Rationality) │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Structured Evidence Vector
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                                DOMAIN 3: ADAPTATION LAYER                              │
│                                                                                        │
│   Feature Extractor  ───────►  2-Slice DBN Filter  ───────►  Beta-Binomial Trust       │
│   [KA, Var, Loss]             (P(Attack), P(Fault))           (Asymmetric & Grudge)    │
│                                                                            │           │
│                                                                            ▼           │
│   Continuous Steering  ◄──────────────────────────────────  D* Lite Incremental        │
│   (atan2 + Repulsion Vector)                                (Dynamic Hazard Field)     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### The Six-Stage Tick Cycle
For every agent $i$ at tick $t$:
1. **Broadcast & Encryption**: Agent $i$ encapsulates its physical state $\mathbf{s}_i(t) = [x_i, y_i, \theta_i, v_i, \text{status}_i, t]^T$, encrypts it using pairwise derived keys, and transmits over UDP.
2. **Channel Interception**: The MITM proxy intercepts the datagram, consults the attack schedule, and either forwards, injects spatial offsets, replays stale packets, or drops the packet.
3. **Decryption & Epistemic Ingestion**: Agent $j$ decrypts the packet, verifies the monotonic tick condition, updates its symbolic Knowledge Base via AGM belief revision, and stores the observed state.
4. **SMT Kinematic Verification & Theory of Mind**: Agent $j$ queries Z3 SMT to verify whether the transition $\mathbf{s}_i(t-1) \to \mathbf{s}_i(t)$ obeys physical velocity bounds, acceleration limits, and obstacle constraints. Concurrently, Bayesian ToM evaluates the Boltzmann likelihood of this move relative to environmental goals.
5. **Causal Attribution via DBN**: If an epistemic clash occurs, the continuous anomaly $\delta$ and rolling statistics are mapped into a discrete feature vector $\mathbf{e}_t$. The 2-slice DBN executes an exact forward-filter recursion to calculate marginal posteriors $P(A_t \mid \mathbf{e}_{1:t})$ and $P(F_t \mid \mathbf{e}_{1:t})$.
6. **Trust Modulation & Continuous Path Adaptation**: The causal verdict updates agent $j$'s Beta trust parameters $(\alpha_i, \beta_i)$ for peer $i$. When trust drops below threshold $\tau_{\text{reroute}}$, D* Lite updates its hazard cost field, and continuous steering blends attractive goal vectors with repulsive hazard vectors.

---

## 3. Mathematical Formulations & Component Specifications

### 3.1 Network Layer & Cryptographic Mechanics

#### 3.1.1 Elliptic Curve Diffie-Hellman (ECDH) Key Exchange
Before communication begins, agents perform non-interactive pairwise key derivation over the NIST SECP256R1 curve:
- Base point $G \in E(\mathbb{F}_p)$ with order $n$.
- Private key $d_i \in_R [1, n-1]$.
- Public key $Q_i = d_i \cdot G$.
- Pairwise shared secret:
  $$S_{ij} = d_i \cdot Q_j = d_i \cdot (d_j \cdot G) = d_j \cdot (d_i \cdot G) = S_{ji}$$
- Cryptographic key derivation via HKDF-SHA256:
  $$K_{ij} = \text{HKDF-Extract}(\text{salt}=\emptyset, \text{IKM}=S_{ij.x})$$
  $$\text{Key}_{ij} = \text{HKDF-Expand}(K_{ij}, \text{info}=\text{"def-fleet-comms-v1"}, \text{len}=32\text{ bytes})$$

#### 3.1.2 Authenticated Encryption (AES-256-GCM)
Payload serialization: $\mathbf{m} = \text{JSON}(\mathbf{s}_i(t))$.
- Initialization Vector: $\text{IV} \in_R \{0, 1\}^{96}$ (12 bytes generated uniformly at random per transmission).
- Encryption:
  $$C, \mathbf{tag} = \text{AES-GCM-Encrypt}_{\text{Key}_{ij}}(\text{IV}, \mathbf{m}, \text{AAD}=\text{Header})$$
- The 16-byte authentication tag guarantees ciphertext integrity and authenticity. An external attacker altering ciphertext bytes causes polynomial Galois field evaluation mismatch, rejected with `CRYPTO_AUTHENTICATION_FAILURE`.

#### 3.1.3 Monotonic Replay Guard
To defeat replay attacks where an attacker captures valid ciphertexts and re-emits them later:
$$\text{Guard}(t_{\text{recv}}, t_{\text{last}}) = \begin{cases} \text{Accept}, & \text{if } t_{\text{recv}} > t_{\text{last}} \\ \text{Reject (Replay Attack)}, & \text{if } t_{\text{recv}} \le t_{\text{last}} \end{cases}$$
The receiver drops the packet immediately without passing it to the reasoning layer, incrementing the crypto replay fault counter.

#### 3.1.4 Simplified DES (S-DES) Educational Fallback
For resource-constrained or pedagogical execution modes, an exact 8-bit block / 10-bit key S-DES cipher is implemented:
- Key Generation: 10-bit raw key $\to P_{10} \to$ Circular left shift (LS-1) $\to P_8 \to K_1$; LS-2 $\to P_8 \to K_2$.
- Feistel Round Function:
  $$f_K(R, K) = P_4\Big(S_0(EP(R) \oplus K)_{L} \,\|\, S_1(EP(R) \oplus K)_{R}\Big)$$
  where $EP$ is the 8-bit expansion permutation, $S_0, S_1$ are $4 \times 4$ substitution matrices, and $P_4$ is the 4-bit output permutation.

---

### 3.2 Reasoning Layer: AGM Belief Revision & Theory of Mind

#### 3.2.1 AGM Belief Revision (Alchourrón, Gärdenfors, Makinson 1985)
The internal state of agent $j$'s mind maintains an epistemic belief base $K$. When new telemetry $\phi = (\text{peer}_i, \text{pos}, \mathbf{x}, t)$ arrives:
- **Success Postulate**: $\phi \in K * \phi$.
- **Consistency Postulate**: If $\phi$ is consistent, $K * \phi$ is consistent.
- **Revision Operator**:
  $$K * \phi = (K \setminus \text{Stale}(\text{peer}_i)) \cup \{\phi\}$$
  Historical beliefs are retained in an immutable temporal trace $\mathcal{H}_i = [\mathbf{x}_i(t_0), \mathbf{x}_i(t_1), \dots, \mathbf{x}_i(t_k)]$ for derivative kinematic and statistical calculation.

#### 3.2.2 Formal SMT Kinematic Verification (Z3 Theorem Prover)
Let the previous confirmed position of peer $i$ be $\mathbf{p}_{t-1} = (x_{t-1}, y_{t-1})$ at tick $t-1$, and the claimed position be $\mathbf{p}_t = (x_t, y_t)$ at tick $t$.
Agent $j$ constructs a first-order logic sentence over the real arithmetic theory $\mathcal{T}_{\text{NRA}}$ (Nonlinear Real Arithmetic):

$$\Phi \equiv \Phi_{\text{temporal}} \land \Phi_{\text{kinematic}} \land \Phi_{\text{collision}}$$

1. **Temporal Strict Monotonicity**:
   $$\Phi_{\text{temporal}} \equiv (t - t_{\text{prev}} > 0)$$
2. **Kinematic Reachability Bound**:
   The maximum theoretical Euclidean displacement over time interval $\Delta t_{\text{ticks}} = t - t_{\text{prev}}$ is bounded by maximum vehicle velocity $v_{\text{max}} = 6.0\text{ px/tick}$, plus a sensor noise tolerance margin $\epsilon = 12.0\text{ px}$:
   $$D_{\text{euclid}}^2 = (x_t - x_{t-1})^2 + (y_t - y_{t-1})^2$$
   $$D_{\text{max}} = v_{\text{max}} \cdot \Delta t_{\text{ticks}} + \epsilon$$
   $$\Phi_{\text{kinematic}} \equiv \left(D_{\text{euclid}}^2 \le D_{\text{max}}^2\right)$$
3. **Obstacle Non-Intersection**:
   For each rectangular static obstacle $O_k = [x_{\text{min}}^k, x_{\text{max}}^k, y_{\text{min}}^k, y_{\text{max}}^k]$, the claimed destination point $\mathbf{p}_t$ must not lie inside the occupied volume:
   $$\Phi_{\text{collision}} \equiv \bigwedge_{k} \neg \left( x_{\text{min}}^k \le x_t \le x_{\text{max}}^k \land y_{\text{min}}^k \le y_t \le y_{\text{max}}^k \right)$$

If Z3 evaluates $\text{Check}(\Phi) \to \text{UNSAT}$, an **Epistemic Paradox** is asserted:
$$\text{Contradiction} = \text{True}, \quad \delta = \max\left(0, \sqrt{D_{\text{euclid}}^2} - D_{\text{max}}\right)$$

#### 3.2.3 Bayesian Theory of Mind & Boltzmann Intent Inference
Human observers and sophisticated cognitive agents do not interpret spatial motions in a vacuum; they assume teammates act **rationally** to fulfill unobserved intentions (Baker, Saxe, & Tenenbaum 2011).

Let $\mathcal{G} = \{g_1, g_2, \dots, g_M\}$ be the discrete set of environmental goals (resource deposits, foraging waypoints).
- **Boltzmann Rationality Action Model**:
  The likelihood that agent $i$ executes action $\mathbf{a}_t = \mathbf{p}_t - \mathbf{p}_{t-1}$ given candidate goal $g_m$ is governed by the exponentiated negative geodesic cost differential:
  $$Q(s, \mathbf{a}; g_m) = -\|\mathbf{p}_t - g_m\|_2$$
  $$P(\mathbf{a}_t \mid g_m, \mathbf{p}_{t-1}) = \frac{\exp\Big(\beta_{\text{rational}} \cdot Q(s, \mathbf{a}; g_m)\Big)}{\sum_{\mathbf{a}'} \exp\Big(\beta_{\text{rational}} \cdot Q(s, \mathbf{a}'; g_m)\Big)}$$
  In our continuous formulation, this simplifies to the directional cosine alignment between the displacement vector $\mathbf{d} = \mathbf{p}_t - \mathbf{p}_{t-1}$ and the goal target vector $\mathbf{u}_m = g_m - \mathbf{p}_{t-1}$:
  $$\cos(\theta_m) = \frac{\mathbf{d} \cdot \mathbf{u}_m}{\|\mathbf{d}\|_2 \|\mathbf{u}_m\|_2}$$
  $$P(\mathbf{a}_t \mid g_m) \propto \exp\Big(\beta_{\text{rational}} \cdot \cos(\theta_m)\Big)$$
  where $\beta_{\text{rational}} = 2.5$ represents the inverse temperature (rationality index).

- **Intent Posterior Recursive Update**:
  Applying Bayes' rule across candidate goals:
  $$P(g_m \mid \mathbf{a}_{1:t}) = \frac{P(\mathbf{a}_t \mid g_m) \cdot P(g_m \mid \mathbf{a}_{1:t-1})}{\sum_{k=1}^M P(\mathbf{a}_t \mid g_k) \cdot P(g_k \mid \mathbf{a}_{1:t-1})}$$
  If an adversary injects random or malicious spatial teleportation, the directional alignment with all genuine goals drops simultaneously, triggering an **Intent Entropy Spike**:
  $$H(\mathcal{G}) = -\sum_{m=1}^M P(g_m) \log_2 P(g_m)$$

---

### 3.3 Adaptation Layer: Causal DBN & Trust Replanning

#### 3.3.1 Feature Vector Extraction & Discretization
Continuous observations from the reasoning layer are transformed into discrete random variables over a sliding temporal window $W = 12\text{ ticks}$:
1. **Kinematic Anomaly ($K_t \in \{0, 1, 2\}$)**:
   $$K_t = \begin{cases} 0\text{ (Low)}, & \delta < 15.0\text{ px} \\ 1\text{ (Medium)}, & 15.0 \le \delta < 80.0\text{ px} \\ 2\text{ (High)}, & \delta \ge 80.0\text{ px} \end{cases}$$
2. **Rolling Delta Variance ($V_t \in \{0, 1\}$)**:
   $$\sigma_\delta^2 = \frac{1}{|W|} \sum_{\tau \in W} (\delta_\tau - \bar{\delta})^2$$
   $$V_t = \begin{cases} 0\text{ (Low)}, & \sigma_\delta^2 < 200.0\text{ px}^2 \\ 1\text{ (High)}, & \sigma_\delta^2 \ge 200.0\text{ px}^2 \end{cases}$$
3. **Estimated Packet Loss ($L_t \in \{0, 1\}$)**:
   Calculated from paradox occurrence frequency over window $W$:
   $$L_t = \begin{cases} 1\text{ (High Loss)}, & \frac{1}{|W|}\sum_{\tau \in W} \mathbb{I}(\text{anomaly}_\tau) \ge 0.30 \\ 0\text{ (Low Loss)}, & \text{otherwise} \end{cases}$$

#### 3.3.2 Two-Slice Dynamic Bayesian Network (DBN)
A standard Bayesian classifier assumes independent and identically distributed (i.i.d.) observations, failing to capture the temporal persistence of active cyber attacks. DEF employs a 2-slice DBN forward filter:

- **State Space**:
  - Hidden variables: $A_t \in \{0, 1\}$ (Under Active Attack), $F_t \in \{0, 1\}$ (Hardware Fault).
  - Observed variables: $K_t \in \{0, 1, 2\}$, $V_t \in \{0, 1\}$, $L_t \in \{0, 1\}$.

- **First-Order Markov Transition Model**:
  Attacks and hardware faults exhibit temporal autocorrelation:
  $$P(A_t = 1 \mid A_{t-1} = 1) = 0.90, \quad P(A_t = 1 \mid A_{t-1} = 0) = 0.05$$
  $$P(F_t = 1 \mid F_{t-1} = 1) = 0.80, \quad P(F_t = 1 \mid F_{t-1} = 0) = 0.08$$

- **Observation Emission Likelihoods ($CPT_K$ and $CPT_V$)**:
  - Conditional probability table for Kinematic Anomaly $K_t$ given $(A_t, F_t)$:
    $$\begin{array}{cc|ccc}
    A_t & F_t & P(K_t=0) & P(K_t=1) & P(K_t=2) \\
    \hline
    0 & 0 & 0.92 & 0.07 & 0.01 \\
    0 & 1 & 0.15 & 0.65 & 0.20 \\
    1 & 0 & 0.03 & 0.17 & 0.80 \\
    1 & 1 & 0.01 & 0.14 & 0.85 \\
    \end{array}$$
  - Conditional probability table for Variance $V_t$ given $(F_t, L_t)$:
    $$\begin{array}{cc|cc}
    F_t & L_t & P(V_t=0) & P(V_t=1) \\
    \hline
    0 & 0 & 0.90 & 0.10 \\
    0 & 1 & 0.40 & 0.60 \\
    1 & 0 & 0.20 & 0.80 \\
    1 & 1 & 0.10 & 0.90 \\
    \end{array}$$

- **Exact Closed-Form Forward Filter Update**:
  Let $\mathbf{P}_{t-1}(A, F)$ be the joint posterior probability at $t-1$.
  1. *Time Prediction Step*:
     $$\text{Prior}_t(A_t = a, F_t = f) = \sum_{a' \in \{0,1\}} \sum_{f' \in \{0,1\}} P(A_t=a \mid A_{t-1}=a') \cdot P(F_t=f \mid F_{t-1}=f') \cdot \mathbf{P}_{t-1}(a', f')$$
  2. *Measurement Likelihood Conditioning*:
     $$\tilde{P}_t(A_t = a, F_t = f) = P(K_t = k \mid A_t = a, F_t = f) \cdot P(V_t = v \mid F_t = f, L_t = l) \cdot \text{Prior}_t(a, f)$$
  3. *Marginal Normalization*:
     $$Z_t = \sum_{a \in \{0,1\}} \sum_{f \in \{0,1\}} \tilde{P}_t(a, f)$$
     $$\mathbf{P}_t(A_t = a, F_t = f) = \frac{\tilde{P}_t(a, f)}{Z_t}$$
  4. *Marginal Inference*:
     $$P(A_t = 1) = \mathbf{P}_t(1, 0) + \mathbf{P}_t(1, 1)$$
     $$P(F_t = 1) = \mathbf{P}_t(0, 1) + \mathbf{P}_t(1, 1)$$

- **Causal Verdict Decision Boundary**:
  $$\text{Verdict} = \begin{cases} \text{ATTACK}, & \text{if } P(A_t = 1) \ge 0.50 \land P(A_t = 1) > P(F_t = 1) \\ \text{HARDWARE\_FAULT}, & \text{if } P(F_t = 1) \ge 0.50 \land P(F_t = 1) \ge P(A_t = 1) \\ \text{HONEST}, & \text{otherwise} \end{cases}$$

#### 3.3.3 Dirichlet/Beta Trust Engine with Ratchet Grudge Memory
Trust in peer $j$ is modeled as a subjective probability density following a Beta distribution $\text{Beta}(\alpha_j, \beta_j)$:
$$f(p; \alpha_j, \beta_j) = \frac{1}{\text{B}(\alpha_j, \beta_j)} p^{\alpha_j - 1} (1 - p)^{\beta_j - 1}$$
- **Expected Trust**:
  $$\mathbb{E}[T_j] = \frac{\alpha_j}{\alpha_j + \beta_j}$$
- **Epistemic Uncertainty (Variance)**:
  $$\text{Var}[T_j] = \frac{\alpha_j \beta_j}{(\alpha_j + \beta_j)^2 (\alpha_j + \beta_j + 1)}$$

##### Asymmetric Update Rules & Temporal Aging
Real-world reputation exhibits severe asymmetry: trust is destroyed rapidly through transgression, but rebuilt painstakingly through sustained cooperation.
When an event occurs:
1. *Temporal Aging (Multiplicative Decay)*:
   $$\alpha_j \leftarrow \alpha_0 + \lambda \cdot (\alpha_j - \alpha_0), \quad \beta_j \leftarrow \beta_0 + \lambda \cdot (\beta_j - \beta_0)$$
   where $\lambda = 0.97$ per update tick, preventing ancient historical benevolence from masking recent malicious actions.
2. *Asymmetric Evidence Accumulation*:
   - If Verdict == `ATTACK`:
     $$\Delta \beta = 2.50 \cdot P(A_t=1), \quad \Delta \alpha = 0.0$$
   - If Verdict == `HARDWARE_FAULT`:
     $$\Delta \beta = 0.80 \cdot P(F_t=1), \quad \Delta \alpha = 0.0$$
   - If Verdict == `HONEST`:
     $$\Delta \alpha = 0.35 \cdot \mu_{\text{grudge}}, \quad \Delta \beta = 0.0$$
3. *One-Way Ratchet Grudge Memory*:
   Agent $i$ continuously records the historical minimum trust:
   $$\tau_{\text{min}}^j = \min\Big(\tau_{\text{min}}^j, \mathbb{E}[T_j]\Big)$$
   The recovery multiplier $\mu_{\text{grudge}}$ is determined by a strict step function:
   $$\mu_{\text{grudge}} = \begin{cases} 1.00\text{ (NORMAL)}, & \tau_{\text{min}}^j \ge 0.60 \\ 0.40\text{ (MODERATE)}, & 0.40 \le \tau_{\text{min}}^j < 0.60 \\ 0.12\text{ (SLOW)}, & 0.20 \le \tau_{\text{min}}^j < 0.40 \\ 0.03\text{ (FROZEN)}, & \tau_{\text{min}}^j < 0.20 \end{cases}$$
   Once an agent has been caught executing a blatant spatial injection that drives its trust below $0.20$, it can virtually never regain full trusted status during the mission lifetime.

#### 3.3.4 D* Lite Trust-Modulated Path Planning & Continuous Steering
Standard path planning executes over a static binary occupancy grid. DEF introduces a **dynamic trust hazard potential field**.

- **Trust Hazard Field Construction**:
  For every peer $j$ whose expected trust $\mathbb{E}[T_j] < \tau_{\text{reroute}} = 0.40$:
  Let peer $j$ be at coordinates $\mathbf{p}_j = (x_j, y_j)$. For every grid cell $c = (u, v)$ within hazard radius $R_{\text{hazard}} = 12\text{ cells}$:
  $$d(c, \mathbf{p}_j) = \|c - \mathbf{p}_j\|_2$$
  $$\text{Hazard}(c) \leftarrow \text{Hazard}(c) + \kappa \cdot \big(1 - \mathbb{E}[T_j]\big) \cdot \left(1 - \frac{d(c, \mathbf{p}_j)}{R_{\text{hazard}}}\right)$$
  where $\kappa = 4.0$ is the hazard sensitivity coefficient.

- **Incremental Edge Cost**:
  The traversal cost between adjacent grid vertices $u$ and $v$ is inflated:
  $$c(u, v) = \|u - v\|_2 \cdot \Big(1.0 + \text{Hazard}(v)\Big)$$
  D* Lite maintains consistency between the estimated distance to goal $g(s)$ and the one-step lookahead cost $rhs(s)$:
  $$rhs(u) = \min_{s' \in \text{Succ}(u)} \Big(c(u, s') + g(s')\Big)$$
  When trust updates alter cell hazards, only the affected local vertices are updated in the priority queue, achieving $O(k)$ graph repair rather than full $O(N \log N)$ A* re-computation.

- **Continuous-Angle Omnidirectional Repulsion Vector**:
  Robots do not jump from grid cell to grid cell. In `src/adaptation/path_planner.py`, `get_avoidance_vector(pos)` calculates a continuous physical repulsion force:
  $$\vec{F}_{\text{repulse}}(\mathbf{p}_i) = \sum_{j: \mathbb{E}[T_j] < 0.40} \left(1 - \frac{\|\mathbf{p}_i - \mathbf{p}_j\|_2}{R_{\text{hazard}}}\right) \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\|\mathbf{p}_i - \mathbf{p}_j\|_2} \cdot \big(1 - \mathbb{E}[T_j]\big)$$
  In `src/main.py`, the robot's physical target steering combines the attractive unit vector toward the goal $\mathbf{u}_{\text{target}}$ and the repulsion vector:
  $$\mathbf{v}_{\text{desired}} = \mathbf{u}_{\text{target}} + 1.8 \cdot \vec{F}_{\text{repulse}}$$
  $$\theta_{\text{desired}} = \text{atan2}(v_{\text{desired}.y}, v_{\text{desired}.x})$$
  $$\theta_t = \theta_{t-1} + \text{clamp}\Big(\text{angle\_diff}(\theta_{\text{desired}}, \theta_{t-1}), -\Delta \theta_{\text{max}}, \Delta \theta_{\text{max}}\Big)$$
  $$x_t = x_{t-1} + v \cdot \cos(\theta_t) \cdot \Delta t, \quad y_t = y_{t-1} + v \cdot \sin(\theta_t) \cdot \Delta t$$
  This allows continuous heading across all angles $[0, 2\pi)$ without discretization artifacts.

---

### 3.4 Foraging Dynamics & Lévy-Flight Mobility

When an agent is not actively tracking a known resource, it executes a heavy-tailed random search strategy. Standard Brownian motion (Gaussian random walks) is statistically sub-optimal for locating sparse resources, as it repeatedly covers previously searched space.

DEF implements a **Lévy-Flight Foraging Pattern** using Mantegna's algorithm (Mantegna 1994, Viswanathan et al. 1999):
- Step length distribution:
  $$P(l) \propto l^{-\alpha_{\text{levy}}}, \quad \alpha_{\text{levy}} = 1.5$$
- Mantegna step generator:
  $$u \sim \mathcal{N}(0, \sigma_u^2), \quad v \sim \mathcal{N}(0, \sigma_v^2)$$
  $$\sigma_u = \left( \frac{\Gamma(1 + \alpha) \cdot \sin(\pi \alpha / 2)}{\Gamma((1 + \alpha) / 2) \cdot \alpha \cdot 2^{(\alpha - 1)/2}} \right)^{1/\alpha}, \quad \sigma_v = 1.0$$
  $$\text{step} = \frac{u}{|v|^{1/\alpha}}$$
This produces a combination of intensive local foraging clusters punctuated by occasional long-distance ballistic exploratory sprints.

---

## 4. Architectural Component Registry & Codebase Mapping

| Component | Source File | Key Class / Method | Mathematical / Algorithmic Function |
|---|---|---|---|
| **Dataclass State** | `src/environment/agent_state.py` | `AgentState` | Continuous coordinate tracking $(x, y, \theta, v)$, elastic boundary reflection |
| **Pygame Arena** | `src/environment/arena.py` | `Arena.render()` | Multi-layer visualizer: hazard heatmap, intent arrows, path polylines, trust auras |
| **ECDH Comms** | `src/network/secure_comms.py` | `ECDHSession`, `SecureChannel` | Curve SECP256R1, HKDF-SHA256, AES-256-GCM encryption, monotonic replay check |
| **MITM Proxy** | `src/network/mitm_proxy.py` | `MITMProxy` | Intercepts UDP port 9999; implements position forgery, replay delay, silent drops |
| **S-DES Cipher** | `src/network/sdes.py` | `SDES` | 8-bit block / 10-bit key educational Feistel cipher |
| **Knowledge Base** | `src/reasoning/belief.py` | `KnowledgeBase`, `Belief` | AGM belief revision, historical trajectory retention, topic queries |
| **SMT Verification** | `src/reasoning/smt_checker.py` | `SMTChecker.check_transition()` | Z3 formal kinematic boundary, temporal progression, and obstacle collision proofs |
| **Theory of Mind** | `src/reasoning/theory_of_mind.py`| `BayesianToM.infer_intent()` | Boltzmann rationality intent posteriors over named environmental targets |
| **BDI Engine** | `src/reasoning/bdi_engine.py` | `BDIEngine.update_beliefs()` | Orchestrates incoming messages $\to$ AGM update $\to$ SMT proof $\to$ ToM inference |
| **Feature Extractor**| `src/adaptation/feature_extractor.py`| `FeatureExtractor.extract()` | Discretizes continuous anomaly $\delta$, rolling variance, and packet loss rate |
| **Dynamic Bayes Net**| `src/adaptation/bayesian_network.py` | `DynamicBayesianClassifier.classify()`| 2-slice exact forward filter computing joint $P(A_t, F_t \mid \mathbf{e}_{1:t})$ and math trace |
| **Trust Engine** | `src/adaptation/trust_manager.py` | `BetaTrustEngine.update()` | Beta-Binomial Dirichlet updates, asymmetric decay, one-way ratchet grudge tiers |
| **Path Planner** | `src/adaptation/path_planner.py` | `TrustDStarLite` | Incremental D* Lite search, trust hazard cost field, continuous repulsion vectors |
| **Simulation Core** | `src/main.py` | `run_simulation()` | AsyncIO UDP event loop, Lévy flight foraging, continuous vector blending |
| **Event Ledger** | `src/dashboard/logger.py` | `StructuredEventLedger` | Append-only JSON Lines logging of all internal mathematical states |
| **Telemetry UI** | `src/dashboard/app.py` | Streamlit Dashboard | 6-tab research console: DBN math trace, Beta distributions, D* Lite maps, SMT logs |

---

## 5. Telemetry Dashboard (Research Console) Structure

The Streamlit dashboard (`src/dashboard/app.py`) is designed as a **transparent evaluation workbench** allowing examiners to inspect every intermediate calculation:

1. **Tab 1: Mission Overview**: High-level key performance indicators (total events, attack detection recall/precision, trust equilibrium), per-agent collected resource counts, and an attack timeline Gantt chart aligning scheduled attack waves with agent detection flags.
2. **Tab 2: Bayesian Network Inspector**: Real-time forward-filter trace viewer. Displays the starting prior, transition propagation, $CPT_K / CPT_V$ lookups, unnormalized $2 \times 2$ joint distribution heatmap, normalization constant $Z$, and final marginal probabilities $P(A_t)$ and $P(F_t)$ alongside categorical explanations.
3. **Tab 3: Beta Trust Engine**: Live probability density function (PDF) plots of $\text{Beta}(\alpha, \beta)$ for all peers, showing how epistemic variance shrinks with observations and how the grudge ratchet irreversibly scales down recovery speed.
4. **Tab 4: D* Lite Planner**: Dynamic spatial heatmap of cell hazard penalties $\text{Hazard}(c)$, replan frequency logs, and cost comparisons between unconstrained A* paths versus trust-constrained D* Lite paths.
5. **Tab 5: Epistemic Events & SMT**: Comprehensive tabular ledger of every paradox event, displaying the Z3 SMT evaluation result (kinematic bound vs. obstacle collision vs. temporal anomaly) and Boltzmann intent posteriors across all goals.
6. **Tab 6: Network & Cryptography**: Real-time MITM proxy metrics (forwarded vs. forged vs. replayed vs. dropped counts), cryptographic MAC failure occurrences, and ground-truth attack validation overlays.

---

## 6. Critical Analysis & System Limitations

While the current architecture represents a complete, mathematically grounded research implementation, an objective scientific defense requires acknowledging its present theoretical and practical limitations:

1. **First-Order Theory of Mind Only ($ToM_1$)**:
   Agents currently maintain beliefs about their peers' intentions ($ToM_1$: "I believe peer $j$ intends to reach Goal 1"). However, they do not model recursive beliefs ($ToM_2$: "Peer $j$ knows that I suspect it of lying, and is deliberately moving normally to rebuild trust").
2. **Synchronous SMT Execution Overhead**:
   Calling the Z3 SMT solver synchronously within the agent cognition loop introduces worst-case exponential complexity $\mathcal{O}(2^n)$ for general non-linear real arithmetic. While our specific formulation is small and solves in $< 1\text{ ms}$, scaling to 100+ swarm agents with dynamic obstacle environments would bottleneck real-time performance.
3. **Passive Epistemic Observation**:
   Currently, agents evaluate telemetry **passively** as datagrams arrive. They do not actively perform *epistemic probing* (e.g., executing a deliberate verification maneuver or transmitting a challenge beacon to test whether an anomalous peer responds consistently).
4. **Grid-Decoupled Hybrid Approximation**:
   Although steering is continuous via trigonometric blending and potential repulsion vectors, the underlying global path planning still computes over a discretized grid. In dense obstacle environments, potential field local minima (trap states) can still occur.
5. **Static DBN Parameters**:
   The transition probabilities $P(A_t \mid A_{t-1})$ and emission likelihood tables ($CPT_K, CPT_V$) are currently fixed based on expert domain heuristics rather than learned online via Expectation-Maximization (Baum-Welch algorithm).

---

## 7. Comprehensive Improvement Roadmap

The following prioritized roadmap outlines concrete theoretical, algorithmic, and engineering enhancements to elevate this work to top-tier publications (e.g., IEEE Transactions on Robotics, Autonomous Agents and Multi-Agent Systems (AAMAS), or IROS):

### Roadmap Phase 1: Algorithmic & Theoretical Enhancements

#### 1. Recursive / Second-Order Theory of Mind ($ToM_2$)
- **Concept**: Extend the BDI engine to support nested epistemic models:
  $$\text{Belief}_i\Big(\text{Belief}_j(\mathbf{s}_i)\Big)$$
- **Impact**: Enables detection of sophisticated adversaries who execute deceptive feints (e.g., behaving cooperatively while in proximity to witness nodes, but attacking isolated agents).

#### 2. Active Epistemic Probing via POMDPs
- **Concept**: Formulate the agent's decision process as a Partially Observable Markov Decision Process (POMDP) where the reward function balances:
  $$R(\mathbf{s}, \mathbf{a}) = R_{\text{forage}}(\mathbf{s}, \mathbf{a}) + \gamma_{\text{info}} \cdot \mathcal{I}\Big(A_j; \text{Observation} \mid \mathbf{a}\Big)$$
  where $\mathcal{I}$ is the mutual information gain regarding peer $j$'s hidden state.
- **Impact**: An agent uncertain about whether a peer has suffered motor failure or is under attack can actively maneuver to gain an unoccluded line-of-sight sensor measurement to confirm ground truth.

#### 3. Continuous-Curvature Control Barrier Functions (CBFs)
- **Concept**: Replace the hybrid grid-repulsion model with formal **Control Barrier Functions (CBFs)** integrated into a Quadratic Program (QP):
  $$\min_{\mathbf{u}} \|\mathbf{u} - \mathbf{u}_{\text{nominal}}\|^2 \quad \text{s.t.} \quad \dot{h}(\mathbf{x}, \mathbf{u}) \ge -\gamma \cdot h(\mathbf{x})$$
  where the safety barrier $h(\mathbf{x})$ dynamically expands its safety radius as a function of peer distrust:
  $$h_{ij}(\mathbf{x}) = \|\mathbf{p}_i - \mathbf{p}_j\|^2 - \Big(R_{\text{safe}} + \lambda_{\text{trust}} \cdot (1 - \mathbb{E}[T_j])\Big)^2$$
- **Impact**: Provides mathematically guaranteed collision avoidance with smooth continuous curvature without grid artifacts or potential field trapping.

#### 4. Online Expectation-Maximization (EM) for DBN Adaptation
- **Concept**: Implement online Baum-Welch parameter learning. As agents observe environmental noise levels and packet drop rates across differing operational terrains, the DBN automatically tunes its $CPT_K$ and $CPT_V$ emission tables to prevent false-positive alarms under extreme environmental conditions.

---

### Roadmap Phase 2: Experimental & Architectural Extensions

#### 5. Multi-Agent Reinforcement Learning (MARL) Adversary (Stackelberg Game)
- **Concept**: Replace the scripted attack schedule with an adversarial RL agent (e.g., PPO or MADDPG) trained to maximize resource denial or fleet collision rates while minimizing detection probability by DEF's Bayesian observer.
- **Impact**: Demonstrates system resilience against adaptive, learning adversaries in a formal game-theoretic equilibrium.

#### 6. ROS 2 & Gazebo Hardware-in-the-Loop (HIL) Integration
- **Concept**: Wrap the existing Python domain modules into standard ROS 2 nodes communicating over DDS (Data Distribution Service):
  - `Node_Comms`: Handles encrypted topics and network packet simulation.
  - `Node_Cognition`: Runs the BDI engine, Z3 SMT verifier, and Bayesian ToM.
  - `Node_Navigation`: Nav2 lifecycle node with custom costmap plugin driven by `TrustDStarLite`.
- **Impact**: Enables seamless deployment onto physical mobile robot platforms (e.g., TurtleBot4, Clearpath Jackal) or high-fidelity physics simulators (Gazebo / Webots).

---

## 8. Summary for Thesis & Evaluation Presentations

When presenting this architecture to research advisors, project guides, or conference reviewers, emphasize the following three core distinctions:

1. **Causal Attribution over Anomaly Detection**:
   DEF does not simply flag that "coordinate $(x, y)$ is an outlier." It determines whether the coordinate violates kinematic laws of motion (SMT), contradicts goal-directed intentional rationality (ToM), and probabilistically stems from malicious cyber injection versus sensory hardware fault (DBN).
2. **Asymmetric Cognitive Trust**:
   Unlike symmetric filtering (e.g., standard Kalman consensus or moving-average reputation), DEF implements cognitive asymmetric trust decay with permanent ratchet grudge tiers, mirroring biological trust dynamics.
3. **End-to-End Closed Loop**:
   The architecture forms a closed loop: from low-level AES-GCM encrypted UDP sockets $\to$ symbolic epistemic reasoning $\to$ Bayesian filtering $\to$ trust estimation $\to$ real-time continuous trajectory replanning.
