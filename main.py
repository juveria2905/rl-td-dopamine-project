"""
main.py - Main executable runner for Project 22: Dopamine and Temporal-Difference Learning.
Reproduces classic TD-error experiments and compares learning curves with known RPE findings.

Usage:
    python main.py
"""

import sys
import os
import argparse
from config import Config, DEFAULT_CONFIG
from src.experiments import ExperimentSuite
from src.evaluation import evaluate_and_save_metrics
from src.visualization import generate_all_figures


def run_pipeline(cfg: Config):
    """Run full end-to-end simulation pipeline."""
    print("=" * 78)
    print(" PROJECT 22: DOPAMINE AND TEMPORAL-DIFFERENCE LEARNING ")
    print(" Simulation of Reward Prediction Error (RPE) & Dopamine Dynamics ")
    print("=" * 78)
    print(f"[*] Initializing with Seed={cfg.seed}, Alpha={cfg.alpha}, Gamma={cfg.gamma}")
    print(f"[*] Trial Structure: Length={cfg.trial_length}, CS (Cue) t={cfg.cue_time}, US (Reward) t={cfg.reward_time}")
    print(f"[*] Training Trials={cfg.num_training_trials}, Omission Trials={cfg.num_omission_trials}")

    # 1. Run Experiments
    print("\n[1/3] Running Experiments (Unexpected Reward, Expected, Omission, Magnitude)...")
    suite = ExperimentSuite(cfg)
    results = suite.run_all()
    print("      -> Experiments 1 to 4 and Baseline benchmarks completed successfully.")

    # 2. Evaluate & Save Metrics
    print("\n[2/3] Computing Statistical Metrics and Saving to Disk...")
    metrics = evaluate_and_save_metrics(results, cfg)
    print(f"      -> Metrics saved to: {cfg.metrics_dir}")

    # 3. Generate Visualizations
    print("\n[3/3] Generating High-Resolution Publication Figures...")
    fig_paths = generate_all_figures(results, cfg)
    for fig_id, path in fig_paths.items():
        print(f"      -> Saved {fig_id}: {path}")

    # 4. Print Summary Report
    print("\n" + "=" * 78)
    print(" EXPERIMENT RESULTS & DOPAMINE RPE SUMMARY ")
    print("=" * 78)

    exp1 = metrics["experiment_1_unexpected_reward"]
    exp2 = metrics["experiment_2_expected_reward"]
    exp3 = metrics["experiment_3_reward_omission"]
    exp4 = metrics["experiment_4_magnitude_variation"]
    conv = metrics["convergence"]

    print("\n[EXP 1: UNEXPECTED REWARD (Naive Agent, Trial 1)]")
    print(f"  * Prediction Error at Reward Delivery (delta_4): {exp1['initial_reward_delta']:+.4f} (Large Positive burst)")
    print(f"  * Prediction Error at Cue Onset (delta_0):        {exp1['initial_cue_delta']:+.4f} (No prior expectation)")
    print(f"  * Value Estimate at Reward Time V(s_4):          {exp1['initial_reward_value']:.4f}")

    print("\n[EXP 2: EXPECTED REWARD (Conditioned Agent, Trial 120)]")
    print(f"  * Prediction Error at Reward Delivery (delta_4): {exp2['final_reward_delta']:+.4f} (~0.0, Fully predicted!)")
    print(f"  * Prediction Error at Cue Onset (delta_0):        {exp2['final_cue_delta']:+.4f} (Shifted to CS cue!)")
    print(f"  * Value Estimate at Reward Time V(s_4):          {exp2['final_reward_value']:.4f} (Approached R=1.0)")
    print(f"  * Value Estimate at Cue Onset V(s_0):            {exp2['final_cue_value']:.4f} (Theory gamma^4*R = {exp2['theoretical_cue_value']:.4f})")
    print(f"  * Convergence Trial (|delta_reward| < 0.05):     Trial {conv['convergence_trial_threshold_0_05']}")

    print("\n[EXP 3: REWARD OMISSION (Extinction Probe, Trial 121)]")
    print(f"  * Prediction Error at Cue Onset (delta_0):        {exp3['first_omission_cue_delta']:+.4f} (Anticipation maintained)")
    print(f"  * Prediction Error at Omitted Reward (delta_4):  {exp3['first_omission_reward_delta']:+.4f} (NEGATIVE RPE / Depression!)")

    print("\n[EXP 4: REWARD MAGNITUDE VARIATION]")
    print(f"  * Condition A (Expected R=1.0): delta_reward = {exp4['condition_A_reward_delta']:+.4f} (Equilibrium)")
    print(f"  * Condition B (Boosted R=2.0):  delta_reward = {exp4['condition_B_reward_delta']:+.4f} (Positive RPE spike)")
    print(f"  * Condition C (Reduced R=0.5):  delta_reward = {exp4['condition_C_reward_delta']:+.4f} (Negative RPE dip)")

    print("\n[BASELINE COMPARISON (Mean Squared Prediction Error)]")
    base = metrics["baseline_comparison"]
    print(f"  * TD(0) Learning Agent:         Mean MSPE = {base['td_mean_mspe']:.4f} | Final MSPE = {base['td_final_mspe']:.4f}")
    print(f"  * Fixed Baseline (V=0.5):       Mean MSPE = {base['fixed_mean_mspe']:.4f} | Final MSPE = {base['fixed_final_mspe']:.4f}")
    print(f"  * No-Learning Baseline (V=0.0): Mean MSPE = {base['none_mean_mspe']:.4f} | Final MSPE = {base['none_final_mspe']:.4f}")

    print("\n" + "=" * 78)
    print(" EXECUTION COMPLETED SUCCESSFULLY ")
    print("=" * 78)
    print(f"Output figures saved to:  {cfg.figures_dir}")
    print(f"Metrics and tables saved: {cfg.metrics_dir}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TD Dopamine Simulation")
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha, help="Learning rate alpha")
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG.gamma, help="Discount factor gamma")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="Random seed")
    parser.add_argument("--trials", type=int, default=DEFAULT_CONFIG.num_training_trials, help="Number of training trials")
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        alpha=args.alpha,
        gamma=args.gamma,
        num_training_trials=args.trials,
    )
    run_pipeline(config)

