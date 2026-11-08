# Project 22: Dopamine and Temporal-Difference Learning

A clean, rigorous, and reproducible Reinforcement Learning simulation reproducing classic temporal-difference (TD) error experiments and comparing learning curves with known neurobiological Reward Prediction Error (RPE) findings (Schultz, Dayan & Montague, 1997).

---

## 1. Project Overview

In reinforcement learning, the temporal-difference (TD) error $\delta_t$ quantifies the discrepancy between received/discounted future rewards and an agent's current expectation:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$$

In biological neuroscience, Wolfram Schultz and colleagues (1997) discovered that midbrain dopamine neurons fire in close quantitative correspondence to this TD prediction error.

This project implements a complete, lightweight, CPU-friendly simulation in Python demonstrating:
1. **Experiment 1 (Unexpected Reward):** Naive agent produces large positive prediction error at reward delivery ($\delta = +1.0000$).
2. **Experiment 2 (Expected Reward):** Repeated conditioning drives prediction error at reward delivery to zero ($\delta \to 3.99 \times 10^{-9}$), while transferring an anticipatory prediction error to the conditioned cue ($\delta_{\text{cue}} = +0.8145 = \gamma^4 R$).
3. **Experiment 3 (Reward Omission):** Unexpectedly withholding expected reward evokes a negative prediction error ($\delta = -1.0000$), modeling dopamine firing depression.
4. **Experiment 4 (Reward Magnitude Shifts):** Evaluates linear scaling of prediction error under boosted ($R=2.0 \implies \delta = +1.0$) and reduced ($R=0.5 \implies \delta = -0.5$) reward conditions.
5. **Baseline Benchmarks:** Compares TD learning against non-learning Fixed Prediction ($V=0.5$) and No-Learning ($V=0.0$) baselines, demonstrating a 100% reduction in final Mean Squared Prediction Error (MSPE).

---

## 2. Directory Structure

```text
rl_td_dopamine_project/
├── README.md               # Complete project overview & instructions
├── requirements.txt         # Package dependencies (numpy, matplotlib, pandas, scipy)
├── python.cmd              # Windows execution wrapper ensuring clean environment launch
├── main.py                 # Main orchestrator script running all experiments
├── config.py               # Centralized hyperparameters & directory configuration
├── VIVA_QUESTIONS.md       # 15 faculty examination questions with answers
│
├── src/
│   ├── __init__.py         # Package init
│   ├── environment.py      # PavlovianRewardEnv (Complete Serial Compound representation)
│   ├── td_agent.py         # Tabular TD(0) learning agent
│   ├── baseline.py         # FixedPredictionBaseline and NoLearningBaseline
│   ├── experiments.py      # Orchestrator for Experiments 1-4 and baselines
│   ├── evaluation.py       # Metrics calculation, CSV/JSON export, convergence check
│   └── visualization.py    # Publication-quality plotting routines (300 DPI)
│
├── results/
│   ├── figures/            # High-resolution generated PNG figures
│   │   ├── fig1_reward_vs_trial.png
│   │   ├── fig2_predicted_value_vs_trial.png
│   │   ├── fig3_td_error_vs_trial.png
│   │   ├── fig4_td_vs_baseline.png
│   │   ├── fig5_rpe_profiles_dopamine.png
│   │   └── fig_architecture.png
│   └── metrics/            # Generated metrics, logs, and CSV tables
│       ├── metrics_summary.json
│       ├── model_comparison.csv
│       ├── training_trials.csv
│       └── omission_trials.csv
│
├── report/
│   └── RL_TD_Dopamine_Project_Report.md  # Comprehensive academic project report
│
└── presentation/
    └── RL_TD_Dopamine_5_Slides.md        # Exactly 5 concise, presentation-ready slides
```

---

## 3. Installation & Setup

This project uses standard scientific Python packages and requires no GPU.

### Prerequisites
- Python 3.10 or higher.

### Install Dependencies
```bash
pip install -r requirements.txt
```

*(Or use the installed environment packages: `numpy`, `matplotlib`, `pandas`, `scipy`)*

---

## 4. Execution

To run the entire pipeline, execute the main script from the project root directory:

```bash
python main.py
```
*(On Windows systems where the Python launcher alias is active, you can also run `.\python.cmd main.py`)*

### Optional Command-Line Arguments
```bash
python main.py --alpha 0.15 --gamma 0.95 --trials 120 --seed 42
```

When execution completes, it prints a clean terminal summary of all numerical results and saves all figures and metric logs automatically.

---

## 5. Verified Experimental Results

Execution of `python main.py` produces the following verified metrics:

| Metric | Measured Value | Biological / Computational Significance |
| :--- | :---: | :--- |
| **Initial Reward Error ($\delta_{\text{reward}}$)** | `+1.0000` | Unpredicted reward evokes strong positive dopamine burst |
| **Final Reward Error ($\delta_{\text{reward}}$)** | `+0.0000` | Expected reward evokes zero response; fully predicted |
| **Final Cue Error ($\delta_{\text{cue}}$)** | `+0.8145` | Prediction error migrates backward to predictive cue |
| **Theoretical Cue Value ($\gamma^4 R$)** | `0.8145` | Value estimation matches analytical optimum (MAE = $5.49 \times 10^{-6}$) |
| **Convergence Trial ($|\delta| < 0.05$)** | **Trial 20** | Rapid learning within 20 trials |
| **Reward Omission Error** | `-1.0000` | Withheld reward causes negative prediction error (firing pause) |
| **Magnitude Discrepancy ($R=2.0$)** | `+1.0000` | Linear scaling of prediction error on unexpected excess |
| **TD Agent Mean MSPE** | `0.0300` | Progressive convergence to minimal prediction error |
| **TD Agent Final MSPE** | `1.03e-16` | 100% error reduction |
| **Fixed Baseline MSPE ($V=0.5$)** | `0.2500` | Non-adaptive static error |
| **No-Learning Baseline MSPE ($V=0$)** | `1.0000` | Static failure to anticipate reward or cue |

---

## 6. Generated Figures

The simulation generates six high-resolution (300 DPI) figures saved in `results/figures/`:
1. **`fig1_reward_vs_trial.png`**: Step-plot of delivered reward across conditioning ($R=1.0$) and omission ($R=0.0$) phases.
2. **`fig2_predicted_value_vs_trial.png`**: Evolution of state values $V(s_{\text{reward}})$ and $V(s_{\text{cue}})$, highlighting theoretical asymptotic convergence.
3. **`fig3_td_error_vs_trial.png`**: Trial-by-trial TD error dynamics, showing error decay at reward, emergence at cue, and negative omission dip.
4. **`fig4_td_vs_baseline.png`**: Two-panel comparison of absolute error and running MSPE between TD(0), Fixed Baseline, and No-Learning Baseline.
5. **`fig5_rpe_profiles_dopamine.png`**: Four-panel intra-trial peri-stimulus firing rate profile replicating the classic Wolfram Schultz (1997) findings.
6. **`fig_architecture.png`**: Programmatic flowchart depicting the RL-Dopamine TD feedback cycle.

---

## 7. Submission Checklist

When submitting this course assignment, provide:
1. **Academic Report:** `report/RL_TD_Dopamine_Project_Report.md` (or exported PDF).
2. **Presentation Slides:** `presentation/RL_TD_Dopamine_5_Slides.md`.
3. **Viva Examination Guide:** `VIVA_QUESTIONS.md`.
4. **Source Code & Data:** `main.py`, `config.py`, `src/`, `results/figures/`, `results/metrics/`.
5. **Instructions:** This `README.md` and `requirements.txt`.

