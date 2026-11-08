# Project 22: Problem Formulation
## Dopamine and Temporal-Difference Learning: A Reinforcement Learning Simulation of Reward Prediction Error

---

### Course & Student Information
- **Course:** Reinforcement Learning
- **Student Name:** `[STUDENT NAME PLACEHOLDER]`
- **Roll Number:** `[ROLL NUMBER PLACEHOLDER]`
- **Project Track:** Project 22 — Dopamine and Temporal-Difference Learning

---

### 1. Problem Description
In biological nervous systems, midbrain dopamine neurons emit phasic firing signals that quantitatively mirror temporal-difference (TD) reward prediction errors (RPE). The objective of this project is to model a discrete temporal Pavlovian conditioning task where an agent encounters discrete temporal states separating a predictive conditioned stimulus (CS) from an unconditioned reward stimulus (US), learning the discounted future return via TD(0) bootstrapping.

---

### 2. Formal Problem Formulation

#### 2.1 State Space ($\mathcal{S}$)
The task utilizes the **Complete Serial Compound (CSC)** temporal representation (Sutton & Barto, 1990; Schultz et al., 1997). Each trial consists of $T = 6$ discrete time steps $t \in \{0, 1, 2, 3, 4, 5\}$:
$$\mathcal{S} = \{s_0, s_1, s_2, s_3, s_4, s_5\}$$
- **State $s_0$ (Cue Onset):** Presentation of the predictive sensory stimulus (CS) at $t = 0$.
- **States $s_1, s_2, s_3$ (Delay Interval):** Inter-stimulus interval (ISI) during which the sensory stimulus trace is maintained.
- **State $s_4$ (Reward Step):** Delivery of the unconditioned reward stimulus (US) at $t = 4$.
- **State $s_5$ (Terminal State):** Absorbing state marking trial completion and transition to the inter-trial interval (ITI).

#### 2.2 Action Space ($\mathcal{A}$)
Classical Pavlovian conditioning measures passive associative learning rather than instrumental motor control. Thus, the action space is singleton:
$$\mathcal{A} = \{a_{\text{wait}}\}$$
The policy is stationary and fixed: $\pi(a_{\text{wait}} \mid s) = 1.0 \quad \forall s \in \mathcal{S}$. The task evaluates policy value prediction.

#### 2.3 Reward Function ($\mathcal{R}$)
Reward feedback $r_t \in \mathbb{R}$ is delivered conditionally at step $t=4$:
$$r_t = \begin{cases}
R & \text{if } s_t = s_4 \text{ (reward delivery step)}, \\
0 & \text{otherwise}.
\end{cases}$$
- **Standard Conditioning:** $R = 1.0$ (Conditioned expectation)
- **Reward Omission:** $R = 0.0$ (Extinction probe)
- **Magnitude Shift:** $R = 2.0$ (Boosted reward) or $R = 0.5$ (Reduced reward)

#### 2.4 Environment Transition Dynamics ($\mathcal{P}$)
Transitions are deterministic within each trial:
$$P(s_{t+1} = s_t + 1 \mid s_t, a_{\text{wait}}) = 1.0 \quad \text{for } t < 5$$
Upon reaching absorbing state $s_5$, the episode terminates (`done = True`) and resets to $s_0$.

#### 2.5 Value Function ($V$) and Discount Factor ($\gamma$)
The state value function $V^\pi(s)$ defines the expected cumulative discounted future reward from state $s$ with discount factor $\gamma = 0.95$:
$$V^*(s_t) = \mathbb{E}\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; s_t \right] = \begin{cases}
\gamma^{4 - t} R & \text{for } 0 \le t \le 4, \\
0 & \text{for } t \ge 5.
\end{cases}$$
Analytic values for $R=1.0$: $V^*(s_4) = 1.0000$, $V^*(s_3) = 0.9500$, $V^*(s_2) = 0.9025$, $V^*(s_1) = 0.8574$, $V^*(s_0) = 0.8145$.

---

### 3. Temporal-Difference Learning Algorithm
The agent computes the TD prediction error $\delta_t$ and updates state values incrementally via:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$
where $\alpha = 0.15$ is the learning rate.
- $\delta_t > 0$: Outcome is better than expected (Phasic dopamine burst).
- $\delta_t \approx 0$: Outcome matches expectation (Tonic baseline firing).
- $\delta_t < 0$: Outcome is worse than expected (Dopamine firing pause/depression).

