"""
src/td_agent.py - Temporal-Difference (TD(0)) Learning Agent.
Implements TD error calculation and value function updates:
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    V(s_t)  <- V(s_t) + alpha * delta_t
"""

from typing import List, Dict, Any, Optional
import numpy as np


class TDAgent:
    """
    Tabular TD(0) learning agent representing value prediction and dopamine-like RPE.
    """

    def __init__(
        self,
        num_states: int,
        alpha: float = 0.15,
        gamma: float = 0.95,
        initial_value: float = 0.0,
    ):
        """
        Initialize the TD agent.
        
        Args:
            num_states: Number of discrete states in the state space.
            alpha: Learning rate (0 < alpha <= 1).
            gamma: Discount factor (0 <= gamma <= 1).
            initial_value: Initial estimate for all state values.
        """
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.initial_value = initial_value

        # Value table V(s)
        self.V = np.full(num_states, initial_value, dtype=np.float64)

        # Tracking histories
        self.value_history: List[np.ndarray] = [self.V.copy()]
        self.delta_history: List[List[float]] = []  # List of [delta_0, delta_1, ...] per trial
        self.trial_rewards: List[float] = []

    def reset_values(self):
        """Reset the value function to initial state."""
        self.V = np.full(self.num_states, self.initial_value, dtype=np.float64)
        self.value_history = [self.V.copy()]
        self.delta_history = []
        self.trial_rewards = []

    def get_value(self, state: int) -> float:
        """Return the current value estimate for a given state."""
        if 0 <= state < self.num_states:
            return float(self.V[state])
        return 0.0

    def compute_td_error(
        self,
        state: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Compute the Temporal-Difference error:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        
        If next_state is terminal (done == True), V(s_{t+1}) = 0.
        """
        v_current = self.V[state]
        v_next = 0.0 if done else self.V[next_state]
        delta = reward + (self.gamma * v_next) - v_current
        return float(delta)

    def update(
        self,
        state: int,
        delta: float,
        learn: bool = True,
    ) -> float:
        """
        Update the state value estimate using the computed TD error:
            V(s_t) <- V(s_t) + alpha * delta_t
            
        Args:
            state: Current state index.
            delta: TD prediction error delta_t.
            learn: If False, skip update (used for fixed/frozen evaluation).
            
        Returns:
            Updated value V(s_t).
        """
        if learn and (0 <= state < self.num_states):
            self.V[state] += self.alpha * delta
        return float(self.V[state])

    def record_trial(
        self,
        deltas: List[float],
        total_reward: float,
    ):
        """Record trial trajectories for post-hoc analysis and plotting."""
        self.delta_history.append(deltas.copy())
        self.value_history.append(self.V.copy())
        self.trial_rewards.append(total_reward)

