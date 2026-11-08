"""
config.py - Centralized configuration for Dopamine and TD Learning Simulation.
Project 22: Dopamine and Temporal-Difference Learning
"""

import os
from dataclasses import dataclass

@dataclass
class Config:
    # -----------------------------
    # Reproducibility
    # -----------------------------
    seed: int = 42

    # -----------------------------
    # Reinforcement Learning Params
    # -----------------------------
    alpha: float = 0.15          # Learning rate (step size for value updates)
    gamma: float = 0.95          # Discount factor for future rewards
    initial_value: float = 0.0   # Initial state value estimates

    # -----------------------------
    # Pavlovian Trial Structure
    # -----------------------------
    # In each trial:
    #   t = 0: Cue presentation (Conditioned Stimulus, CS)
    #   t = 1, 2, 3: Delay interval (Inter-Stimulus Interval, ISI)
    #   t = 4: Reward delivery time (Unconditioned Stimulus, US)
    #   t = 5: Trial termination / post-reward state
    trial_length: int = 6        # Total time steps per trial [0, 1, 2, 3, 4, 5]
    cue_time: int = 0            # Time step when CS is presented
    reward_time: int = 4         # Time step when US/reward is delivered
    terminal_time: int = 5       # Time step when trial terminates

    # -----------------------------
    # Reward Magnitudes
    # -----------------------------
    standard_reward: float = 1.0 # Standard reward magnitude (US)
    boosted_reward: float = 2.0  # Boosted reward magnitude (Condition B)
    omitted_reward: float = 0.0  # Reward during omission test trials

    # -----------------------------
    # Experiment Trial Counts
    # -----------------------------
    num_training_trials: int = 120   # Conditioning trials (Exp 1 & 2)
    num_omission_trials: int = 25    # Reward omission trials (Exp 3)
    num_magnitude_trials: int = 25   # Boosted reward trials (Exp 4)

    # -----------------------------
    # Baseline Parameters
    # -----------------------------
    fixed_baseline_value: float = 0.5  # Fixed value estimate for static baseline

    # -----------------------------
    # File Paths & Directories
    # -----------------------------
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    results_dir: str = os.path.join(base_dir, "results")
    figures_dir: str = os.path.join(results_dir, "figures")
    metrics_dir: str = os.path.join(results_dir, "metrics")
    report_dir: str = os.path.join(base_dir, "report")
    presentation_dir: str = os.path.join(base_dir, "presentation")

    def ensure_directories(self):
        """Create necessary output directories if they do not exist."""
        for d in [self.results_dir, self.figures_dir, self.metrics_dir, 
                  self.report_dir, self.presentation_dir]:
            os.makedirs(d, exist_ok=True)


# Global default configuration instance
DEFAULT_CONFIG = Config()
