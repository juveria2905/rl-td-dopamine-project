"""
src/visualization.py - Publication-quality plotting routines for Dopamine and TD Learning.
Generates:
  Figure 1: Reward Received vs Trial
  Figure 2: Predicted Value vs Trial
  Figure 3: TD Error vs Trial
  Figure 4: TD-Learning Agent vs Baseline
  Figure 5: Positive, Zero, and Negative Prediction-Error Examples (Peri-stimulus dynamics)
  Figure Architecture: Reinforcement Learning & Dopamine TD Cycle Diagram
"""

from typing import Dict, Any
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from config import Config


def set_academic_style():
    """Apply clean, academic styling to matplotlib figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def plot_fig1_reward_vs_trial(results: Dict[str, Any], cfg: Config) -> str:
    """
    Figure 1: Reward received vs trial across experimental phases.
    """
    set_academic_style()
    df_combined = results["df_combined"]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    trials = df_combined["trial_number"].values
    rewards = df_combined["total_reward"].values

    # Plot continuous reward line
    ax.step(trials, rewards, where="mid", color="#1f77b4", linewidth=2.2, label=r"Delivered Reward ($r_t$)")

    # Add phase boundary
    omission_start = cfg.num_training_trials + 1
    ax.axvline(x=omission_start - 0.5, color="#d62728", linestyle=":", linewidth=2.0, label="Phase Shift: Omission")

    # Highlight phases with soft background spans
    ax.axvspan(1, cfg.num_training_trials, alpha=0.08, color="#2ca02c", label="Conditioning Phase (Standard US)")
    ax.axvspan(omission_start, trials[-1], alpha=0.08, color="#d62728", label="Extinction Phase (Omitted US)")

    ax.set_xlabel("Trial Number", fontweight="bold")
    ax.set_ylabel(r"Reward Received ($r_t$)", fontweight="bold")
    ax.set_title("Figure 1: Reward Received Across Experimental Trials", fontweight="bold", pad=12)
    ax.set_ylim(-0.15, 1.25)
    ax.set_xlim(1, trials[-1])
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    out_path = os.path.join(cfg.figures_dir, "fig1_reward_vs_trial.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig2_predicted_value_vs_trial(results: Dict[str, Any], cfg: Config) -> str:
    """
    Figure 2: Predicted value vs trial (Cue onset V(s_0) and Reward step V(s_4)).
    """
    set_academic_style()
    df_combined = results["df_combined"]

    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)

    trials = df_combined["trial_number"].values
    cue_vals = df_combined["cue_value"].values
    rew_vals = df_combined["reward_value"].values

    theoretical_cue = (cfg.gamma ** (cfg.reward_time - cfg.cue_time)) * cfg.standard_reward
    theoretical_rew = cfg.standard_reward

    ax.plot(trials, rew_vals, color="#ff7f0e", linewidth=2.4, label=r"Value at Reward Time Step $V(s_{reward})$")
    ax.plot(trials, cue_vals, color="#2ca02c", linewidth=2.4, label=r"Value at Cue Onset $V(s_{cue})$")

    # Theoretical asymptotic value reference lines
    ax.axhline(theoretical_rew, color="#ff7f0e", linestyle="--", alpha=0.65, label=f"Theoretical Reward Value ($R={theoretical_rew:.1f}$)")
    ax.axhline(theoretical_cue, color="#2ca02c", linestyle="--", alpha=0.65, label=rf"Theoretical Cue Value ($\gamma^4 R={theoretical_cue:.4f}$)")

    # Phase transition
    omission_start = cfg.num_training_trials + 1
    ax.axvline(x=omission_start - 0.5, color="#d62728", linestyle=":", linewidth=2.0)
    ax.text(cfg.num_training_trials / 2, 0.15, "Conditioning Phase\n(Asymptotic Value Learning)", 
            ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#2ca02c"))
    ax.text(omission_start + (len(trials) - omission_start) / 2, 0.70, "Omission Phase\n(Extinction Decay)", 
            ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#d62728"))

    ax.set_xlabel("Trial Number", fontweight="bold")
    ax.set_ylabel(r"Estimated State Value $V(s)$", fontweight="bold")
    ax.set_title("Figure 2: Evolution of Estimated State Values Across Trials", fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(1, trials[-1])
    ax.legend(loc="center right", frameon=True, framealpha=0.92)
    plt.tight_layout()

    out_path = os.path.join(cfg.figures_dir, "fig2_predicted_value_vs_trial.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig3_td_error_vs_trial(results: Dict[str, Any], cfg: Config) -> str:
    """
    Figure 3: Temporal-Difference Error delta_t vs trial at reward delivery and cue onset.
    """
    set_academic_style()
    df_combined = results["df_combined"]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

    trials = df_combined["trial_number"].values
    rew_deltas = df_combined["reward_delta"].values
    cue_deltas = df_combined["cue_delta"].values

    ax.plot(trials, rew_deltas, color="#d62728", linewidth=2.2, label=r"TD Error at Reward Delivery ($\delta_{reward}$)")
    ax.plot(trials, cue_deltas, color="#1f77b4", linewidth=2.2, label=r"TD Error at Cue Onset ($\delta_{cue}$)")

    # Reference zero line
    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)

    # Phase boundary
    omission_start = cfg.num_training_trials + 1
    ax.axvline(x=omission_start - 0.5, color="#7f7f7f", linestyle=":", linewidth=2.0)

    # Annotate key biological phenomena
    ax.annotate(
        "Exp 1: Unexpected Reward\nLarge positive error at US",
        xy=(1, rew_deltas[0]),
        xytext=(15, 0.90),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffebee", edgecolor="#d62728"),
    )

    conv_trial = 90
    ax.annotate(
        r"Exp 2: Expected Reward" + "\n" + r"$\delta_{reward} \to 0$ (Predicted US)" + "\n" + r"$\delta_{cue} > 0$ (Transfers to CS)",
        xy=(conv_trial, rew_deltas[conv_trial - 1]),
        xytext=(50, 0.35),
        arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f4f8", edgecolor="#1f77b4"),
    )

    ax.annotate(
        r"Exp 3: Reward Omission" + "\n" + r"Negative RPE ($\delta < 0$)" + "\n" + "Dopamine depression",
        xy=(omission_start, rew_deltas[omission_start - 1]),
        xytext=(omission_start - 25, -0.75),
        arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffebee", edgecolor="#d62728"),
    )

    ax.set_xlabel("Trial Number", fontweight="bold")
    ax.set_ylabel(r"Temporal-Difference Prediction Error ($\delta_t$)", fontweight="bold")
    ax.set_title("Figure 3: Temporal-Difference Error Evolution and RPE Dynamics", fontweight="bold", pad=12)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlim(1, trials[-1])
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    plt.tight_layout()

    out_path = os.path.join(cfg.figures_dir, "fig3_td_error_vs_trial.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig4_td_vs_baseline(results: Dict[str, Any], cfg: Config) -> str:
    """
    Figure 4: TD-learning agent vs baseline models (prediction error and running MSPE).
    """
    set_academic_style()
    df_training = results["df_training"]
    df_fixed = results["df_baseline_fixed"]
    df_none = results["df_baseline_none"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    trials = df_training["trial_number"].values

    # Panel A: Instantaneous Reward Prediction Error |delta_reward|
    ax1.plot(trials, np.abs(df_training["reward_delta"]), color="#2ca02c", linewidth=2.3, label="TD(0) Learning Agent")
    ax1.plot(trials, np.abs(df_fixed["reward_delta"]), color="#ff7f0e", linestyle="--", linewidth=2.0, label="Fixed Baseline ($V=0.5$)")
    ax1.plot(trials, np.abs(df_none["reward_delta"]), color="#d62728", linestyle=":", linewidth=2.0, label="No-Learning Baseline ($V=0.0$)")

    ax1.set_xlabel("Trial Number", fontweight="bold")
    ax1.set_ylabel(r"Absolute Prediction Error $|\delta_{reward}|$", fontweight="bold")
    ax1.set_title("A: Absolute Prediction Error at Reward Delivery", fontweight="bold")
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(loc="center right", frameon=True)

    # Panel B: Cumulative / Running Mean Squared Prediction Error (MSPE)
    td_mspe = np.cumsum(df_training["reward_delta"] ** 2) / np.arange(1, len(trials) + 1)
    fixed_mspe = np.cumsum(df_fixed["reward_delta"] ** 2) / np.arange(1, len(trials) + 1)
    none_mspe = np.cumsum(df_none["reward_delta"] ** 2) / np.arange(1, len(trials) + 1)

    ax2.plot(trials, td_mspe, color="#2ca02c", linewidth=2.3, label="TD(0) Learning Agent")
    ax2.plot(trials, fixed_mspe, color="#ff7f0e", linestyle="--", linewidth=2.0, label="Fixed Baseline ($V=0.5$)")
    ax2.plot(trials, none_mspe, color="#d62728", linestyle=":", linewidth=2.0, label="No-Learning Baseline ($V=0.0$)")

    ax2.set_xlabel("Trial Number", fontweight="bold")
    ax2.set_ylabel("Running Mean Squared Prediction Error (MSPE)", fontweight="bold")
    ax2.set_title("B: Cumulative Learning Curve Comparison (MSPE)", fontweight="bold")
    ax2.set_ylim(-0.05, 1.15)
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle("Figure 4: Performance Comparison — TD Learning vs Non-Learning Baselines", fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(cfg.figures_dir, "fig4_td_vs_baseline.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig5_rpe_profiles_dopamine(results: Dict[str, Any], cfg: Config) -> str:
    """
    Figure 5: Canonical Schultz-style peri-stimulus time step profiles.
    Shows the step-by-step TD error delta_t within a trial across four conditions:
      Panel A: Unexpected Reward (Early naive agent)
      Panel B: Expected Reward (Conditioned agent)
      Panel C: Reward Omission (Expected reward withheld)
      Panel D: Boosted Reward (Unexpected double reward)
    """
    set_academic_style()
    exemplars = results["exemplar_trajectories"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300, sharex=True, sharey=True)
    axes_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    keys = ["unexpected", "expected", "omission", "boosted"]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    time_steps = np.arange(cfg.trial_length)

    for idx, (key, ax, color) in enumerate(zip(keys, axes_list, colors)):
        data = exemplars[key]
        deltas = data["deltas"]
        plot_deltas = deltas[:cfg.trial_length] if len(deltas) >= cfg.trial_length else deltas + [0.0] * (cfg.trial_length - len(deltas))

        # Bar chart representing phasic neuronal firing rate analogy
        bar_colors = ["#2ca02c" if d > 0.05 else ("#d62728" if d < -0.05 else "#7f7f7f") for d in plot_deltas]
        ax.bar(time_steps[:len(plot_deltas)], plot_deltas, width=0.55, color=bar_colors, edgecolor="black", alpha=0.85)

        # Baseline zero line
        ax.axhline(0.0, color="black", linestyle="-", linewidth=1.2)

        # Highlight CS and US time steps
        ax.axvline(cfg.cue_time, color="blue", linestyle=":", alpha=0.6, label="CS (Cue)" if idx == 0 else None)
        ax.axvline(cfg.reward_time, color="orange", linestyle=":", alpha=0.6, label="US (Reward)" if idx == 0 else None)

        ax.set_title(data["label"], fontweight="bold", fontsize=11)
        ax.set_ylabel(r"TD Error $\delta_t$ (Dopamine Analogy)", fontweight="bold", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(-1.25, 1.35)

        # Annotate specific key step values
        for t_step, d_val in enumerate(plot_deltas):
            if abs(d_val) > 0.05:
                va = "bottom" if d_val > 0 else "top"
                y_offset = 0.06 if d_val > 0 else -0.06
                ax.text(t_step, d_val + y_offset, f"{d_val:+.2f}", ha="center", va=va, fontsize=8.5, fontweight="bold")

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlabel("Time Step within Trial ($t$)", fontweight="bold")
        ax.set_xticks(time_steps)
        ax.set_xticklabels([f"t={t}\n(CS)" if t == cfg.cue_time else (f"t={t}\n(US)" if t == cfg.reward_time else f"t={t}") for t in time_steps])

    axes[0, 0].legend(loc="upper right", frameon=True, fontsize=9)
    fig.suptitle("Figure 5: Positive, Zero, and Negative Temporal-Difference Prediction Errors\n"
                 "(Computational Reproduction of Wolfram Schultz's 1997 Dopamine Findings)", 
                 fontweight="bold", fontsize=13, y=0.99)
    plt.tight_layout()

    out_path = os.path.join(cfg.figures_dir, "fig5_rpe_profiles_dopamine.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_architecture_diagram(cfg: Config) -> str:
    """
    Generate a clean visual architecture diagram illustrating the RL-Dopamine TD Cycle:
    Environment -> State -> Agent -> Action -> Reward -> TD Error (delta) -> Value Update -> Next State
    """
    set_academic_style()
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Define box styles
    box_env = dict(boxstyle="round,pad=0.6", facecolor="#e1f5fe", edgecolor="#0288d1", linewidth=2.2)
    box_agent = dict(boxstyle="round,pad=0.6", facecolor="#e8f5e9", edgecolor="#388e3c", linewidth=2.2)
    box_dopamine = dict(boxstyle="round,pad=0.6", facecolor="#fff3e0", edgecolor="#f57c00", linewidth=2.2)
    box_value = dict(boxstyle="round,pad=0.6", facecolor="#f3e5f5", edgecolor="#7b1fa2", linewidth=2.2)

    # Boxes
    ax.text(2.0, 5.0, "ENVIRONMENT\nPavlovian Conditioning\n(CS Cue, Delay, US Reward)", ha="center", va="center", bbox=box_env, fontweight="bold", fontsize=10.5)
    ax.text(8.5, 5.0, "RL AGENT\nTemporal-Difference Learner\nState Value Table $V(s)$", ha="center", va="center", bbox=box_agent, fontweight="bold", fontsize=10.5)
    ax.text(8.5, 1.8, r"VALUE UPDATE" + "\n" + r"$V(s_t) \leftarrow V(s_t) + \alpha \delta_t$" + "\nBiological Plasticity", ha="center", va="center", bbox=box_value, fontweight="bold", fontsize=10.5)
    ax.text(3.5, 1.8, r"TD ERROR / DOPAMINE SIGNAL" + "\n" + r"$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$" + "\nReward Prediction Error (RPE)", ha="center", va="center", bbox=box_dopamine, fontweight="bold", fontsize=10.5)

    # Arrows
    arrow_kw = dict(arrowstyle="->", lw=2.2, color="#37474f", mutation_scale=18)
    
    # Env -> Agent: State s_t
    ax.annotate("", xy=(6.5, 5.5), xytext=(3.8, 5.5), arrowprops=arrow_kw)
    ax.text(5.15, 5.75, r"State $s_t$ (Temporal CSC)", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0277bd")

    # Agent -> Env: Action a_t (passive orienting)
    ax.annotate("", xy=(3.8, 4.5), xytext=(6.5, 4.5), arrowprops=arrow_kw)
    ax.text(5.15, 4.15, r"Action $a_t$ (Passive Observe)", ha="center", va="top", fontsize=9.5, color="#555555")

    # Env -> Dopamine: Reward r_t & Next State s_{t+1}
    ax.annotate("", xy=(2.8, 2.7), xytext=(2.0, 4.0), arrowprops=arrow_kw)
    ax.text(1.3, 3.2, r"Reward $r_t$" + "\n" + r"Next State $s_{t+1}$", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#d84315")

    # Agent -> Dopamine: Predictions V(s_t) and V(s_{t+1})
    ax.annotate("", xy=(5.5, 2.2), xytext=(7.5, 4.1), arrowprops=arrow_kw)
    ax.text(7.2, 3.0, r"Predictions" + "\n" + r"$V(s_t), V(s_{t+1})$", ha="center", va="center", fontsize=9.5, color="#388e3c")

    # Dopamine -> Value Update: Prediction error delta_t
    ax.annotate("", xy=(6.8, 1.8), xytext=(5.6, 1.8), arrowprops=arrow_kw)
    ax.text(6.2, 2.15, r"TD Error $\delta_t$" + "\n(Phasic Burst/Dip)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#e65100")

    # Value Update -> Agent: Updated Table V
    ax.annotate("", xy=(8.5, 3.8), xytext=(8.5, 2.7), arrowprops=arrow_kw)
    ax.text(9.3, 3.25, r"Value Sync" + "\n" + r"$V \leftarrow V + \alpha \delta$", ha="center", va="center", fontsize=9.5, color="#7b1fa2")

    ax.set_title("System Architecture: Temporal-Difference Learning and Dopamine RPE Feedback Loop", fontweight="bold", fontsize=12.5, pad=15)
    plt.tight_layout()

    out_path = os.path.join(cfg.figures_dir, "fig_architecture.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_all_figures(results: Dict[str, Any], cfg: Config) -> Dict[str, str]:
    """Generate all required publication figures and return dict of paths."""
    cfg.ensure_directories()
    paths = {
        "fig1": plot_fig1_reward_vs_trial(results, cfg),
        "fig2": plot_fig2_predicted_value_vs_trial(results, cfg),
        "fig3": plot_fig3_td_error_vs_trial(results, cfg),
        "fig4": plot_fig4_td_vs_baseline(results, cfg),
        "fig5": plot_fig5_rpe_profiles_dopamine(results, cfg),
        "fig_arch": plot_architecture_diagram(cfg),
    }
    return paths
