# Dynamic Epistemic Foraging

## Opening (30 seconds, memorize this)

"My project is called Dynamic Epistemic Foraging. It's a simulated multi-robot system where each agent doesn't just follow a path, it holds beliefs about the world, models what its teammates believe and intend, and continuously decides who among its peers it can still trust, because some of the network traffic between agents is being actively attacked. The core idea is: instead of hardcoding what a 'failure' looks like, the agents reason about failures the way a person would, by asking 'is this peer lying to me, broken, or telling the truth?' and updating their trust accordingly."

That's it. That's the whole project in one breath. Everything else is elaboration.

## The walkthrough, in the order they'll probably want it

### 1. The problem you're solving
"Most multi-robot systems assume clean sensor data and honest teammates. That's unrealistic. Drone swarms and robot fleets share wireless links that can be intercepted or spoofed. Existing systems either hardcode a list of known failures, or they use pure statistics to flag outliers. Neither approach tells you why something went wrong, and neither adapts to something it hasn't seen before."

### 2. What you actually built (the three-domain structure)
Walk it left to right, like the architecture diagram:

- "Domain 1 is the environment. A 2D simulator where agents move around, and they broadcast their position and status to each other over real encrypted sockets, AES or S-DES. I also built a scripted attacker that does man-in-the-middle interception and false data injection, so some of that broadcast traffic is corrupted on purpose."
- "Domain 2 is the reasoning layer. Each agent runs a BDI engine, Belief-Desire-Intention, which is a classic cognitive architecture from Rao and Georgeff. It turns raw numbers into symbolic beliefs, like 'Agent 3 is heading north.' On top of that sits a Theory of Mind module: each agent simulates what its peer should believe given the map, and if a peer's claimed position or intention contradicts that, it flags a logical paradox."
- "Domain 3 is the adaptation layer. When a paradox is flagged, a Dynamic Bayesian Network calculates the probability that it's a hardware fault versus an actual attack. That verdict updates a Beta-Binomial trust score with asymmetric decay and grudge memory. A trained PPO reinforcement learning meta-controller observes epistemic uncertainty and dynamically adapts the reroute threshold, and D* Lite routes around agents with low trust."
- "Then the loop closes: the corrected path gets executed, and the fleet keeps exploring."

### 3. Why this is novel, not just a combination of existing techniques
"Individually, BDI, Theory of Mind, Bayesian fault detection, and trust-based MARL all exist in the literature. What doesn't exist is a single closed pipeline that goes from encrypted network telemetry all the way to trust-weighted replanning, where the trust score comes from a causal, explainable reasoning process instead of a black-box anomaly score. Most security papers treat an attack as 'this data point looks weird, flag it.' We're asking 'why does this data point look weird, and what does that imply about this agent's intentions.'"

## Questions to be ready for

**"Is this actually cognitive robotics if there's no physical robot?"**
Say what we discussed: "Cognitive robotics as a field is defined by the reasoning architecture, not the chassis. Simulated agents are a completely standard testbed, that's how most BDI and Theory-of-Mind research is validated, including the papers I'm building on. My scope explicitly treats physical hardware as future work, not something I'm claiming to have done."

**"What's your evaluation plan? How do you know it works?"**
Have this ready even if it's not fully built yet: "I'll measure trajectory accuracy, precision and recall of the hardware-versus-attack classification, and replanning latency, and I'll scale the attack intensity to see where the system breaks down."

**"Why BDI instead of just training an end-to-end neural network to do all this?"**
"Because I want the agent's decisions to be explainable, I can point to a specific belief and a specific paradox and say 'this is why it distrusted that peer.' An end-to-end black box wouldn't give me that, and it wouldn't let me reason about deception explicitly the way the Sarkadi paper does."

**"What's the hardest part you expect to struggle with?"**
Honest answer, professors respect this: "Probably tuning the Bayesian thresholds so the system doesn't get either paranoid (flagging every sensor glitch as an attack) or naive (missing real attacks). That's the classic precision-recall tradeoff and I expect it'll take real experimentation."

## One tip for the actual conversation

Don't recite Domain 1/2/3 like a script if they interrupt with a question, follow their question. Professors are usually testing whether you understand the reasoning, not whether you memorized the deck. If you can explain the six-step execution loop (explore, anomaly, symbolic translation, epistemic clash, causal calculation, adaptation) in your own words without looking at anything, you're in good shape.
