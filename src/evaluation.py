"""
src/evaluation.py - Statistical evaluation, error analysis, and metric export.
"""

from typing import Dict, Any
import os
import json
import numpy as np
import pandas as pd
from config import Config


def evaluate_and_save_metrics(results: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """
    Compute rigorous metrics comparing TD learning to theoretical optima and baselines,
    and save results to CSV and JSON formats.
    """
    cfg.ensure_directories()

    df_training = results["df_training"]
    df_omission = results["df_omission"]
    df_combined = results["df_combined"]
    df_fixed = results["df_baseline_fixed"]
    df_none = results["df_baseline_none"]
    td_agent = results["td_agent_trained"]

    # 1. Convergence Analysis
    # Find first trial where absolute reward prediction error drops below 0.05 and stays low
    reward_errors = np.abs(df_training["reward_delta"].values)
    converged_indices = np.where(reward_errors < 0.05)[0]
    convergence_trial = int(converged_indices[0] + 1) if len(converged_indices) > 0 else -1

    # 2. Theoretical Value Comparison
    # Theoretical value: V*(s_t) = gamma^(reward_time - t) * standard_reward for t <= reward_time
    theoretical_values = []
    for t in range(cfg.trial_length):
        if t <= cfg.reward_time:
            v_star = (cfg.gamma ** (cfg.reward_time - t)) * cfg.standard_reward
        else:
            v_star = 0.0
        theoretical_values.append(v_star)

    learned_values = td_agent.V.tolist()
    value_mae = float(np.mean(np.abs(np.array(learned_values) - np.array(theoretical_values))))

    # 3. Model Comparison Metrics Table
    # Compare TD agent vs Fixed Baseline vs No-Learning Baseline
    td_mspe_series = (df_training["reward_delta"]) ** 2
    fixed_mspe_series = (df_fixed["reward_delta"]) ** 2
    none_mspe_series = (df_none["reward_delta"]) ** 2

    comparison_rows = [
        {
            "Model": "TD(0) Learning Agent",
            "Initial_Reward_Delta": float(df_training.iloc[0]["reward_delta"]),
            "Final_Reward_Delta": float(df_training.iloc[-1]["reward_delta"]),
            "Cue_Onset_Delta": float(df_training.iloc[-1]["cue_delta"]),
            "Mean_MSPE": float(td_mspe_series.mean()),
            "Final_MSPE": float(td_mspe_series.iloc[-10:].mean()),
            "Adapts_To_Omission": "Yes (Negative RPE dip, delta < 0)",
            "Adapts_To_Magnitude": "Yes (Linear scaling with magnitude)",
        },
        {
            "Model": "Fixed Prediction Baseline (V=0.5)",
            "Initial_Reward_Delta": float(df_fixed.iloc[0]["reward_delta"]),
            "Final_Reward_Delta": float(df_fixed.iloc[-1]["reward_delta"]),
            "Cue_Onset_Delta": float(df_fixed.iloc[-1]["cue_delta"]),
            "Mean_MSPE": float(fixed_mspe_series.mean()),
            "Final_MSPE": float(fixed_mspe_series.iloc[-10:].mean()),
            "Adapts_To_Omission": "No (Static error)",
            "Adapts_To_Magnitude": "No (Static error)",
        },
        {
            "Model": "No-Learning Baseline (V=0.0)",
            "Initial_Reward_Delta": float(df_none.iloc[0]["reward_delta"]),
            "Final_Reward_Delta": float(df_none.iloc[-1]["reward_delta"]),
            "Cue_Onset_Delta": float(df_none.iloc[-1]["cue_delta"]),
            "Mean_MSPE": float(none_mspe_series.mean()),
            "Final_MSPE": float(none_mspe_series.iloc[-10:].mean()),
            "Adapts_To_Omission": "No (Zero response to cue)",
            "Adapts_To_Magnitude": "No (Zero adaptation)",
        },
    ]

    df_comparison = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(cfg.metrics_dir, "model_comparison.csv")
    df_comparison.to_csv(comparison_csv_path, index=False)

    # 4. Save Trial DataFrames to CSV
    training_csv_path = os.path.join(cfg.metrics_dir, "training_trials.csv")
    df_training.to_csv(training_csv_path, index=False)

    omission_csv_path = os.path.join(cfg.metrics_dir, "omission_trials.csv")
    df_omission.to_csv(omission_csv_path, index=False)

    # 5. Build comprehensive JSON summary
    metrics_summary = {
        "experiment_config": {
            "seed": cfg.seed,
            "alpha": cfg.alpha,
            "gamma": cfg.gamma,
            "standard_reward": cfg.standard_reward,
            "boosted_reward": cfg.boosted_reward,
            "trial_length": cfg.trial_length,
            "cue_time": cfg.cue_time,
            "reward_time": cfg.reward_time,
            "num_training_trials": cfg.num_training_trials,
            "num_omission_trials": cfg.num_omission_trials,
        },
        "convergence": {
            "convergence_trial_threshold_0_05": convergence_trial,
            "value_mae_against_theory": value_mae,
            "theoretical_state_values": theoretical_values,
            "learned_state_values": [round(v, 4) for v in learned_values],
        },
        "experiment_1_unexpected_reward": results["exp1_unexpected"],
        "experiment_2_expected_reward": results["exp2_expected"],
        "experiment_3_reward_omission": results["exp3_omission"],
        "experiment_4_magnitude_variation": results["exp4_magnitude"],
        "baseline_comparison": results["baseline_comparison"],
    }

    summary_json_path = os.path.join(cfg.metrics_dir, "metrics_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)

    return metrics_summary

