# Dopamine & Temporal-Difference Learning
## 5-Slide Project Presentation
**Presented by:**
- Shaik Juveria Tabassum (`1005237290091`)
- Dasari John Wesley (`100523729083`)
- Kandula Naveen (`100523729089`)

---

## Slide 1: Motivation & Problem Statement

### Background & Neuro-AI Convergence
- **The Core Problem:** How do intelligent biological and artificial systems learn to anticipate future rewards across temporal delays?
- **Neuroscience Finding (Schultz et al., 1997):** Primate midbrain dopamine neurons fire in phasic bursts and pauses that reflect **reward prediction errors (RPE)** rather than reward delivery itself.
- **Reinforcement Learning Connection:** Sutton & Barto's **Temporal-Difference (TD) learning** provides the exact normative mathematical framework that explains this biological signal.

### Project Objective
- Build a lightweight, reproducible computational model of Pavlovian conditioning.
- Demonstrate how tabular TD(0) updates produce:
  1. Positive error on unexpected reward.
  2. Decay to zero on expected reward + transfer of error to predictive cue.
  3. Negative error dip upon reward omission.
- Empirically benchmark against non-learning control baselines.

---

## Slide 2: Methodology & Theoretical Formulation

### Problem Formulation (Complete Serial Compound - CSC)
- **State Space ($\mathcal{S}$):** Discrete time steps within trial: $s \in \{0, 1, 2, 3, 4, 5\}$.
  - $s_0$: Conditioned Stimulus (CS Cue onset at $t=0$)
  - $s_1, s_2, s_3$: Delay interval (Inter-Stimulus Interval, ISI)
  - $s_4$: Unconditioned Stimulus (US Reward delivery at $t=4$)
  - $s_5$: Terminal absorbing state
- **Action Space ($\mathcal{A}$):** Passive observation/orienting ($a_{\text{wait}}$).
- **Reward Function ($\mathcal{R}$):** $r_4 = 1.0$ (standard), $r_4 = 0.0$ (omission), $r_4 = 2.0$ (boosted).

### The Canonical TD(0) Update Equations
$$\text{TD Error (RPE):} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\text{Value Update:} \quad V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$
- **Hyperparameters:** Learning rate $\alpha = 0.15$, Discount factor $\gamma = 0.95$.
- **Interpretation of $\delta_t$:**
  - $\delta > 0$: Outcome better than expected $\rightarrow$ Dopamine burst ($\uparrow$).
  - $\delta \approx 0$: Outcome fully expected $\rightarrow$ Baseline tonic firing.
  - $\delta < 0$: Outcome worse than expected $\rightarrow$ Dopamine pause/depression ($\downarrow$).

---

## Slide 3: Experimental Results & Findings

### Verified Experimental Outcomes (Executed Suite)
- **Exp 1 — Unexpected Reward:** Naive agent starts at $V=0$. Delivers $\delta_{\text{reward}} = \mathbf{+1.0000}$ (large positive burst).
- **Exp 2 — Expected Reward (Conditioning):** 
  - Rapid convergence within **20 trials** ($|\delta_{\text{reward}}| < 0.05$).
  - Reward error decays to zero: $\delta_{\text{reward}} \to \mathbf{0.0000}$.
  - Anticipatory error emerges at cue: $\delta_{\text{cue}} = \mathbf{+0.8145}$ (matches theoretical $\gamma^4 R = 0.8145$, MAE = $5.49 \times 10^{-6}$).
- **Exp 3 — Reward Omission:** Reward unexpectedly withheld ($r_4 = 0$). Produces sharp negative dip: $\delta_{\text{reward}} = \mathbf{-1.0000}$ (dopamine depression).
- **Exp 4 — Magnitude Variation:** Boosted reward ($R=2.0$) yields $\delta = \mathbf{+1.0000}$; Reduced reward ($R=0.5$) yields $\delta = \mathbf{-0.5000}$.

### Baseline Comparison
- **TD(0) Learning Agent:** Mean MSPE = **$0.0300$** $\rightarrow$ Final MSPE = **$1.03 \times 10^{-16}$** (100% error reduction).
- **Fixed Baseline ($V=0.5$):** Static MSPE = **$0.2500$** (cannot adapt).
- **No-Learning Baseline ($V=0.0$):** Static MSPE = **$1.0000$** (no cue anticipation).

---

## Slide 4: Limitations & Model Scope

### Computational Assumptions
- **Discrete Time Bins:** The Complete Serial Compound assumes rigid, discrete steps, whereas real animal brains process continuous, noisy temporal intervals.
- **Simplified Determinism:** Environmental transitions and delays are modeled as deterministic; real biological environments exhibit temporal jitter.
- **Passive Paradigm:** Evaluates pure value prediction (policy evaluation) without instrumental action-selection or exploration-exploitation trade-offs.

### Biological Distinctions
- **Simulated vs. In-Vivo Data:** Validated against published findings, not direct microelectrode spike trains.
- **Circuit Complexity:** Dopamine neurons operate within complex basal ganglia-cortex loops with diverse receptor subtypes ($D_1$ vs. $D_2$), volume transmission, and co-transmitters.
- **Scalar Simplification:** A single scalar $\delta_t$ abstracts away multi-dimensional homeostatic needs and population-level distributional coding.

---

## 5. Slide 5: Ethics & Conclusions

### Scientific Ethics & Responsible Interpretation
- **Model vs. Reality:** Computational alignment does **not** prove how the human/animal brain physically operates; it is an abstract normative framework.
- **Animal Welfare Context:** Acknowledges historical primate recordings; validates computational modeling as an ethical alternative for hypothesis testing.
- **Addiction Implications:** RPE models explain chemical vulnerability in substance abuse (drugs hijack $\delta_t$). Highlights ethical responsibility in avoiding gamified behavioral exploitation.

### Key Conclusions
- **Mathematical Validation:** Confirmed that tabular TD(0) autonomously replicates all canonical RPE phenomena without ad-hoc parameter tuning.
- **Reproducibility:** 100% reproducible execution via a single command (`python main.py`).
- **Future Directions:** Incorporate Actor-Critic architectures, continuous state representations, and Distributional RL (quantile dopamine coding).

