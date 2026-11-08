# Project 22: Dopamine and Temporal-Difference Learning
## Faculty Viva Examination Preparation Guide

This document contains 15 high-frequency oral examination (viva voce) questions, accompanied by concise, technically accurate, and memorable answers tailored for an undergraduate/postgraduate Reinforcement Learning evaluation.

---

### Q1: What is reinforcement learning?
**Answer:**  
Reinforcement learning (RL) is a computational framework where an autonomous agent learns to make optimal sequential decisions by interacting with an environment. Unlike supervised learning, which relies on labeled target outputs, the agent learns by trial and error using scalar feedback signals called rewards, aiming to maximize cumulative discounted return over time.

---

### Q2: What is temporal-difference (TD) learning?
**Answer:**  
Temporal-difference (TD) learning is a central class of model-free RL algorithms that learns directly from raw experience without requiring a transition model. TD methods combine the core ideas of Monte Carlo methods (learning from sample episodes) and Dynamic Programming (bootstrapping). Specifically, TD updates a state's value estimate based on immediate reward and the *estimated value of the next state*, without waiting for the episode to terminate.

---

### Q3: What is the TD error?
**Answer:**  
The TD error, denoted by $\delta_t$, is the difference between the updated estimate of future return (the TD target) and the agent's current value estimate before observing the transition:
$$\delta_t = \text{TD Target} - \text{Current Estimate} = (r_t + \gamma V(s_{t+1})) - V(s_t)$$
It quantifies the "surprise" or prediction discrepancy at each discrete time step.

---

### Q4: Explain the TD equation and its components.
**Answer:**  
The TD(0) update rule consists of two equations:
1. **Prediction Error Calculation:**
   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
   - $r_t$: Immediate scalar reward received upon transitioning from $s_t$ to $s_{t+1}$.
   - $\gamma$: Temporal discount factor ($0 \le \gamma \le 1$), weighting the present value of future rewards.
   - $V(s_{t+1})$: Estimated value of the successor state. If $s_{t+1}$ is terminal, $V(s_{t+1}) = 0$.
   - $V(s_t)$: Current expected return from state $s_t$.
2. **Value Update Rule:**
   $$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$
   - $\alpha$: Learning rate (step-size parameter) governing how rapidly new information overwrites prior estimates.

---

### Q5: What does alpha ($\alpha$) represent?
**Answer:**  
Alpha ($\alpha$) is the learning rate or step size ($0 < \alpha \le 1$). It determines the proportion of the prediction error that is used to update the current value estimate. A higher $\alpha$ makes the agent adapt quickly to recent outcomes but risks instability, while a lower $\alpha$ ensures smooth, noise-resilient convergence. In biological terms, $\alpha$ corresponds to synaptic plasticity or the rate of Long-Term Potentiation (LTP).

---

### Q6: What does gamma ($\gamma$) represent?
**Answer:**  
Gamma ($\gamma$) is the discount factor ($0 \le \gamma < 1$). It models the subjective time-preference of the agent, determining how much future rewards are valued relative to immediate rewards. A value of $\gamma = 0$ makes the agent myopic (caring only about immediate reward), while $\gamma \to 1$ makes the agent farsighted. In our simulation, $\gamma = 0.95$ causes value to attenuate smoothly as the temporal distance from the reward increases.

---

### Q7: Why does the TD error become close to zero after repeated trials?
**Answer:**  
As the agent experiences repeated trials, the value estimates $V(s_t)$ converge to the true expected discounted returns through iterative updates ($V(s_t) \to \mathbb{E}[r_t + \gamma V(s_{t+1})]$). Once the expectation matches the reality, the TD target $r_t + \gamma V(s_{t+1})$ equals $V(s_t)$, causing their difference $\delta_t$ to decay toward zero. In our simulation, $\delta_{\text{reward}}$ converged from $1.0000$ down to $3.99 \times 10^{-9}$ by Trial 120.

---

### Q8: What happens when an unexpected reward is received?
**Answer:**  
When an unexpected reward is delivered to a naive agent, the agent has no prior anticipation ($V(s_{\text{reward}}) = 0$). Consequently, the TD error is strongly positive:
$$\delta = r - V = 1.0 - 0.0 = +1.0$$
In computational neuroscience, this positive error corresponds directly to the phasic burst of action potentials observed in midbrain dopamine neurons.

---

### Q9: What happens when an expected reward is omitted?
**Answer:**  
When the predictive cue has conditioned the agent to expect a reward ($V(s_{\text{reward}}) \approx 1.0$), but the reward is unexpectedly withheld ($r = 0.0$), the TD error is negative:
$$\delta = 0.0 - 1.0 = -1.0$$
This negative prediction error corresponds to the transient pause or depression in the tonic firing rate of dopamine neurons below baseline (from $\sim 4$ Hz down to $\sim 0$ Hz).

---

### Q10: Why is the TD error related to the biological reward prediction error (RPE)?
**Answer:**  
In 1997, Wolfram Schultz, Peter Dayan, and Read Montague discovered that dopamine neurons in the primate substantia nigra and VTA do not encode sensory stimulus properties or absolute reward amounts; instead, their firing rates quantitatively mirror the mathematical properties of the TD error $\delta_t$:
- Bursting when reward is better than expected ($\delta > 0$).
- Baseline firing when reward is fully predicted ($\delta \approx 0$).
- Pausing when reward is worse than expected ($\delta < 0$).
- Migrating from reward delivery backward to the earliest predictive conditioned stimulus (CS).

---

### Q11: Why did you choose this environment representation?
**Answer:**  
We chose a discrete-trial Complete Serial Compound (CSC) Markov environment because it is the canonical, peer-reviewed standard used in computational neuroscience literature (Sutton & Barto, 1990; Schultz et al., 1997). By representing successive time steps between the cue ($t=0$) and reward delivery ($t=4$) as sequential states, the CSC representation allows TD(0) to explicitly model the temporal delay and observe the step-by-step backward propagation of value and prediction error.

---

### Q12: What baseline did you implement for comparison?
**Answer:**  
We implemented two clearly justified non-learning control baselines:
1. **Fixed Prediction Baseline ($V=0.5$):** Maintains a constant, non-adaptive expectation across all states and trials without updating ($\alpha = 0$).
2. **No-Learning Baseline ($V=0.0$):** Maintains zero expectation permanently, acting as an unconditioned naive agent.

---

### Q13: Why are the baselines weaker than the TD-learning agent?
**Answer:**  
The baselines cannot learn from prediction errors because they lack an update rule ($\alpha = 0$). As a result:
- Their Mean Squared Prediction Error (MSPE) remains statically elevated at $0.2500$ (Fixed) and $1.0000$ (No-Learning), whereas the TD agent reduces MSPE to $1.03 \times 10^{-16}$.
- They cannot transfer anticipation backward to the conditioned stimulus.
- They fail to adapt to reward omission or magnitude shifts, remaining blind to environmental contingencies.

---

### Q14: What are the key limitations of your model?
**Answer:**  
The model has four primary limitations:
1. It uses rigid discrete time bins rather than continuous, noisy temporal dynamics.
2. It assumes deterministic delays and rewards, whereas biological environments are stochastic.
3. It models a passive Pavlovian evaluation task without instrumental action selection or exploration.
4. It abstracts complex biological neural circuitry, neuromodulators, and diverse dopamine receptor subtypes ($D_1$ vs. $D_2$) into a single scalar variable $\delta_t$.

---

### Q15: Does this project prove how dopamine works in the human brain?
**Answer:**  
**No, absolutely not.** It is critical to state that this simulation is a normative *computational model*, not direct biological evidence. While TD learning provides an elegant theoretical framework that matches observed firing rate trends in primate recordings, actual biological brains employ vastly more complex recurrent neural networks, distributed population coding, and intracellular biochemical cascades. We must never claim that an algorithm "proves" human brain function.

