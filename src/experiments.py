"""
src/experiments.py - Implementation of the 4 classic Dopamine & TD Learning Experiments:
  1. Unexpected Reward (early naive trials, large positive TD error)
  2. Expected Reward (conditioning convergence, TD error -> 0 at reward, shifts to cue)
  3. Reward Omission (unexpected omission after conditioning, negative TD error < 0)
  4. Reward Magnitude Variation (Condition A: R=1 vs Condition B: R=2)
Plus systematic baseline comparisons (TD vs Fixed and No-learning baselines).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from config import Config
from src.environment import PavlovianRewardEnv
from src.td_agent import TDAgent
from src.baseline import FixedPredictionBaseline, NoLearningBaseline


def run_single_trial(
    env: PavlovianRewardEnv,
    agent: Any,
    trial_type: str = "PAIRED",
    custom_reward: float = None,
    learn: bool = True,
) -> Dict[str, Any]:
    """
    Execute one full trial (episode) in the environment.
    
    Returns:
        Dictionary containing trial trajectory and summary metrics.
    """
    state, info = env.reset(trial_type=trial_type, custom_reward=custom_reward)
    states = [state]
    rewards = []
    deltas = []
    values_before = []

    # Cue prediction error: instantaneous value burst upon presentation of the predictive CS
    # If trial has a cue, the response to stimulus onset is V(s_0); if no cue (unexpected), 0.0
    val_at_cue_onset = agent.get_value(state)
    cue_delta = val_at_cue_onset if trial_type != "UNEXPECTED" else 0.0

    done = False
    while not done:
        val = agent.get_value(state)
        values_before.append(val)

        next_state, reward, done, step_info = env.step(action=0)
        states.append(next_state)
        rewards.append(reward)

        # Compute TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = agent.compute_td_error(state, reward, next_state, done)
        deltas.append(delta)

        # Update value if agent supports learning
        agent.update(state, delta, learn=learn)

        state = next_state

    total_reward = sum(rewards)
    agent.record_trial(deltas, total_reward)

    reward_idx = env.reward_time

    # Construct the canonical peri-stimulus time profile [t=0 to t=5]
    # At t=0: response to CS cue presentation
    # At t=1..3: delay transitions
    # At t=4: response to US reward delivery
    # At t=5: post-trial baseline
    peri_stimulus_deltas = [cue_delta]
    for step_idx in range(len(deltas)):
        if step_idx < env.trial_length - 1:
            if step_idx == 0:
                continue
            peri_stimulus_deltas.append(deltas[step_idx])
    while len(peri_stimulus_deltas) < env.trial_length:
        peri_stimulus_deltas.append(0.0)

    return {
        "trial_type": trial_type,
        "states": states,
        "rewards": rewards,
        "deltas": deltas,
        "values_before": values_before,
        "total_reward": total_reward,
        "cue_delta": float(cue_delta),
        "reward_delta": float(deltas[reward_idx]) if reward_idx < len(deltas) else 0.0,
        "cue_value": float(val_at_cue_onset),
        "reward_value": float(values_before[reward_idx]) if reward_idx < len(values_before) else 0.0,
        "peri_stimulus_deltas": peri_stimulus_deltas,
    }


class ExperimentSuite:
    """
    Orchestrates Experiments 1-4 and baseline comparison with exact reproducibility.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        np.random.seed(cfg.seed)

    def run_all(self) -> Dict[str, Any]:
        """
        Run the complete experimental pipeline and return all results in structured tables.
        """
        results = {}

        # -----------------------------------------------------------------
        # Phase 1 & 2: Conditioning (Covers Exp 1: Unexpected & Exp 2: Expected)
        # -----------------------------------------------------------------
        env = PavlovianRewardEnv(
            trial_length=self.cfg.trial_length,
            cue_time=self.cfg.cue_time,
            reward_time=self.cfg.reward_time,
            standard_reward=self.cfg.standard_reward,
            boosted_reward=self.cfg.boosted_reward,
        )
        td_agent = TDAgent(
            num_states=env.num_states,
            alpha=self.cfg.alpha,
            gamma=self.cfg.gamma,
            initial_value=self.cfg.initial_value,
        )

        training_trials_data = []
        for trial_idx in range(self.cfg.num_training_trials):
            record = run_single_trial(
                env=env,
                agent=td_agent,
                trial_type="PAIRED",
                learn=True,
            )
            record["trial_number"] = trial_idx + 1
            record["phase"] = "Conditioning"
            training_trials_data.append(record)

        df_training = pd.DataFrame(training_trials_data)
        results["df_training"] = df_training
        results["td_agent_trained"] = td_agent

        # Extract Experiment 1 specific metrics (Early trials: unexpected reward)
        exp1_early = df_training.iloc[:10]
        results["exp1_unexpected"] = {
            "initial_reward_delta": float(df_training.iloc[0]["reward_delta"]),
            "initial_cue_delta": float(df_training.iloc[0]["cue_delta"]),
            "initial_reward_value": float(df_training.iloc[0]["reward_value"]),
            "early_mean_reward_delta": float(exp1_early["reward_delta"].mean()),
        }

        # Extract Experiment 2 specific metrics (Asymptotic trained trials)
        exp2_late = df_training.iloc[-20:]
        theoretical_cue_value = (self.cfg.gamma ** (self.cfg.reward_time - self.cfg.cue_time)) * self.cfg.standard_reward
        results["exp2_expected"] = {
            "final_reward_delta": float(df_training.iloc[-1]["reward_delta"]),
            "final_cue_delta": float(df_training.iloc[-1]["cue_delta"]),
            "final_reward_value": float(df_training.iloc[-1]["reward_value"]),
            "final_cue_value": float(df_training.iloc[-1]["cue_value"]),
            "theoretical_cue_value": float(theoretical_cue_value),
            "late_mean_reward_delta": float(exp2_late["reward_delta"].mean()),
            "late_mean_cue_delta": float(exp2_late["cue_delta"].mean()),
        }

        # -----------------------------------------------------------------
        # Phase 3: Experiment 3 - Reward Omission (Negative Prediction Error)
        # -----------------------------------------------------------------
        omission_agent = TDAgent(
            num_states=env.num_states,
            alpha=self.cfg.alpha,
            gamma=self.cfg.gamma,
        )
        omission_agent.V = td_agent.V.copy()

        omission_trials_data = []
        for trial_idx in range(self.cfg.num_omission_trials):
            record = run_single_trial(
                env=env,
                agent=omission_agent,
                trial_type="OMISSION",
                learn=True,  # Extinction learning
            )
            record["trial_number"] = self.cfg.num_training_trials + trial_idx + 1
            record["phase"] = "Omission"
            omission_trials_data.append(record)

        df_omission = pd.DataFrame(omission_trials_data)
        results["df_omission"] = df_omission
        results["exp3_omission"] = {
            "first_omission_reward_delta": float(df_omission.iloc[0]["reward_delta"]),
            "first_omission_cue_delta": float(df_omission.iloc[0]["cue_delta"]),
            "first_omission_reward_value": float(df_omission.iloc[0]["reward_value"]),
            "final_extinction_reward_delta": float(df_omission.iloc[-1]["reward_delta"]),
        }

        # -----------------------------------------------------------------
        # Phase 4: Experiment 4 - Reward Magnitude Variation
        # -----------------------------------------------------------------
        probe_agent = TDAgent(
            num_states=env.num_states,
            alpha=self.cfg.alpha,
            gamma=self.cfg.gamma,
        )
        probe_agent.V = td_agent.V.copy()

        rec_cond_a = run_single_trial(env, probe_agent, trial_type="PAIRED", custom_reward=1.0, learn=False)
        rec_cond_b = run_single_trial(env, probe_agent, trial_type="BOOSTED", custom_reward=2.0, learn=False)
        rec_cond_c = run_single_trial(env, probe_agent, trial_type="PAIRED", custom_reward=0.5, learn=False)

        results["exp4_magnitude"] = {
            "condition_A_reward": 1.0,
            "condition_A_reward_delta": float(rec_cond_a["reward_delta"]),
            "condition_A_cue_delta": float(rec_cond_a["cue_delta"]),
            "condition_B_reward": 2.0,
            "condition_B_reward_delta": float(rec_cond_b["reward_delta"]),
            "condition_B_cue_delta": float(rec_cond_b["cue_delta"]),
            "condition_C_reward": 0.5,
            "condition_C_reward_delta": float(rec_cond_c["reward_delta"]),
            "condition_C_cue_delta": float(rec_cond_c["cue_delta"]),
            "learned_expectation_at_reward": float(rec_cond_a["reward_value"]),
        }

        # -----------------------------------------------------------------
        # Phase 5: Baseline Comparisons (TD vs Fixed vs No-Learning)
        # -----------------------------------------------------------------
        fixed_baseline = FixedPredictionBaseline(
            num_states=env.num_states,
            fixed_value=self.cfg.fixed_baseline_value,
            gamma=self.cfg.gamma,
        )
        no_learning_baseline = NoLearningBaseline(
            num_states=env.num_states,
            gamma=self.cfg.gamma,
        )

        baseline_fixed_data = []
        baseline_none_data = []

        for trial_idx in range(self.cfg.num_training_trials):
            rec_fixed = run_single_trial(env, fixed_baseline, trial_type="PAIRED", learn=False)
            rec_fixed["trial_number"] = trial_idx + 1
            baseline_fixed_data.append(rec_fixed)

            rec_none = run_single_trial(env, no_learning_baseline, trial_type="PAIRED", learn=False)
            rec_none["trial_number"] = trial_idx + 1
            baseline_none_data.append(rec_none)

        df_fixed = pd.DataFrame(baseline_fixed_data)
        df_none = pd.DataFrame(baseline_none_data)

        td_sq_errors = (df_training["reward_delta"]) ** 2
        fixed_sq_errors = (df_fixed["reward_delta"]) ** 2
        none_sq_errors = (df_none["reward_delta"]) ** 2

        results["df_baseline_fixed"] = df_fixed
        results["df_baseline_none"] = df_none
        results["baseline_comparison"] = {
            "td_mean_mspe": float(td_sq_errors.mean()),
            "td_final_mspe": float(td_sq_errors.iloc[-10:].mean()),
            "fixed_mean_mspe": float(fixed_sq_errors.mean()),
            "fixed_final_mspe": float(fixed_sq_errors.iloc[-10:].mean()),
            "none_mean_mspe": float(none_sq_errors.mean()),
            "none_final_mspe": float(none_sq_errors.iloc[-10:].mean()),
        }

        df_combined = pd.concat([df_training, df_omission], ignore_index=True)
        results["df_combined"] = df_combined

        # Save exemplar Schultz-style peri-stimulus trajectories:
        # 1. Naive trial (Trial 1: unexpected reward)
        # 2. Fully learned trial (Trial 120: expected reward)
        # 3. Omission probe trial (Trial 121: unexpected omission)
        # 4. Magnitude boost trial (Condition B: unexpected double reward)
        results["exemplar_trajectories"] = {
            "unexpected": {
                "label": "A. Unexpected Reward (Naive Agent)",
                "deltas": training_trials_data[0]["peri_stimulus_deltas"],
                "values": training_trials_data[0]["values_before"],
                "rewards": training_trials_data[0]["rewards"],
            },
            "expected": {
                "label": "B. Expected Reward (Conditioned Agent)",
                "deltas": training_trials_data[-1]["peri_stimulus_deltas"],
                "values": training_trials_data[-1]["values_before"],
                "rewards": training_trials_data[-1]["rewards"],
            },
            "omission": {
                "label": "C. Reward Omission (Extinction Probe)",
                "deltas": omission_trials_data[0]["peri_stimulus_deltas"],
                "values": omission_trials_data[0]["values_before"],
                "rewards": omission_trials_data[0]["rewards"],
            },
            "boosted": {
                "label": "D. Boosted Reward (Unexpected R=2.0)",
                "deltas": rec_cond_b["peri_stimulus_deltas"],
                "values": rec_cond_b["values_before"],
                "rewards": rec_cond_b["rewards"],
            },
        }

        return results
