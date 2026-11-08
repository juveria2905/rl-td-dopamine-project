"""
src/baseline.py - Baseline models for comparison with the TD-learning agent.
Provides non-learning benchmarks:
  1. FixedPredictionBaseline: Maintains a constant non-adaptive prediction V(s) = c.
  2. NoLearningBaseline: Maintains a zero prediction V(s) = 0.
"""

from typing import List, Dict, Any
import numpy as np


class FixedPredictionBaseline:
    """
    Fixed Prediction Baseline.
    Always maintains a static value estimate V(s) = fixed_value for all states.
    It does not perform any learning updates (alpha = 0).
    Computes TD error delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) using its fixed value.
    """

    def __init__(
        self,
        num_states: int,
        fixed_value: float = 0.5,
        gamma: float = 0.95,
    ):
        self.num_states = num_states
        self.fixed_value = fixed_value
        self.gamma = gamma

        # Value is constant across all states
        self.V = np.full(num_states, fixed_value, dtype=np.float64)

        # Histories for evaluation
        self.value_history: List[np.ndarray] = [self.V.copy()]
        self.delta_history: List[List[float]] = []
        self.trial_rewards: List[float] = []

    def get_value(self, state: int) -> float:
        """Return the fixed value estimate."""
        return float(self.fixed_value)

    def compute_td_error(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Compute TD prediction error based on static value estimates:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        v_current = self.fixed_value
        v_next = 0.0 if done else self.fixed_value
        delta = reward + (self.gamma * v_next) - v_current
        return float(delta)

    def update(self, state: int, delta: float, learn: bool = False) -> float:
        """Baseline does not update; value remains static."""
        return float(self.fixed_value)

    def record_trial(self, deltas: List[float], total_reward: float):
        """Record trial results."""
        self.delta_history.append(deltas.copy())
        self.value_history.append(self.V.copy())
        self.trial_rewards.append(total_reward)


class NoLearningBaseline:
    """
    No-Learning (Zero Prediction) Baseline.
    Maintains V(s) = 0.0 indefinitely.
    At reward time, delta_t = r_t - 0 = r_t. It never anticipates reward.
    """

    def __init__(
        self,
        num_states: int,
        gamma: float = 0.95,
    ):
        self.num_states = num_states
        self.gamma = gamma
        self.V = np.zeros(num_states, dtype=np.float64)

        self.value_history: List[np.ndarray] = [self.V.copy()]
        self.delta_history: List[List[float]] = []
        self.trial_rewards: List[float] = []

    def get_value(self, state: int) -> float:
        return 0.0

    def compute_td_error(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        return float(reward)

    def update(self, state: int, delta: float, learn: bool = False) -> float:
        return 0.0

    def record_trial(self, deltas: List[float], total_reward: float):
        self.delta_history.append(deltas.copy())
        self.value_history.append(self.V.copy())
        self.trial_rewards.append(total_reward)

