"""
src/environment.py - Pavlovian Reward Conditioning Environment.
Models discrete-trial reward prediction with Complete Serial Compound (CSC) states.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


class PavlovianRewardEnv:
    """
    Pavlovian conditioning environment reproducing classic reward-prediction paradigms.
    
    Structure of each trial (episode):
      - State 0: Cue presentation (CS presented at t=0).
      - State 1, 2, 3: Delay interval (Inter-Stimulus Interval, ISI).
      - State 4: Reward delivery time (Unconditioned Stimulus, US).
      - State 5: Trial termination / absorbing state.
      
    This environment supports four experimental trial types:
      1. 'UNEXPECTED': Reward delivered without predictive cue (unpaired/unpredicted).
      2. 'PAIRED': Cue presented at t=0, standard reward at t=reward_time.
      3. 'OMISSION': Cue presented at t=0, reward is omitted (r=0) at t=reward_time.
      4. 'BOOSTED': Cue presented at t=0, doubled reward (r=2.0) at t=reward_time.
    """

    def __init__(
        self,
        trial_length: int = 6,
        cue_time: int = 0,
        reward_time: int = 4,
        standard_reward: float = 1.0,
        boosted_reward: float = 2.0,
    ):
        self.trial_length = trial_length
        self.cue_time = cue_time
        self.reward_time = reward_time
        self.standard_reward = standard_reward
        self.boosted_reward = boosted_reward

        self.num_states = trial_length
        self.num_actions = 1

        self.current_step = 0
        self.current_trial_type = "PAIRED"
        self.custom_reward: Optional[float] = None
        self.done = False

    def reset(
        self,
        trial_type: str = "PAIRED",
        custom_reward: Optional[float] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment for a new trial.
        
        Args:
            trial_type: 'PAIRED', 'UNEXPECTED', 'OMISSION', or 'BOOSTED'.
            custom_reward: Optional override for the reward magnitude.
            
        Returns:
            initial_state: Integer state s_0 = 0.
            info: Dictionary containing trial metadata.
        """
        self.current_step = 0
        self.current_trial_type = trial_type
        self.custom_reward = custom_reward
        self.done = False

        info = {
            "time_step": self.current_step,
            "trial_type": self.current_trial_type,
            "cue_present": (trial_type != "UNEXPECTED"),
        }
        return self.current_step, info

    def step(self, action: int = 0) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one discrete time step in the trial.
        
        Args:
            action: Passive observation action (default 0).
            
        Returns:
            next_state: Next discrete state index s_{t+1}.
            reward: Scalar reward r_t received at this transition.
            done: Boolean flag indicating if the trial has terminated.
            info: Metadata dictionary.
        """
        if self.done:
            raise RuntimeError("Trial has finished. Call reset() before stepping again.")

        reward = 0.0

        # Check for reward delivery at reward_time
        if self.current_step == self.reward_time:
            if self.custom_reward is not None:
                reward = float(self.custom_reward)
            elif self.current_trial_type == "PAIRED":
                reward = self.standard_reward
            elif self.current_trial_type == "UNEXPECTED":
                reward = self.standard_reward
            elif self.current_trial_type == "OMISSION":
                reward = 0.0
            elif self.current_trial_type == "BOOSTED":
                reward = self.boosted_reward

        # Advance time step
        self.current_step += 1
        next_state = self.current_step

        # Termination condition (when entering absorbing state 5)
        if self.current_step >= self.trial_length - 1:
            self.done = True

        info = {
            "time_step": self.current_step,
            "trial_type": self.current_trial_type,
            "reward_received": reward,
            "is_reward_step": (self.current_step - 1 == self.reward_time),
            "is_cue_step": (self.current_step - 1 == self.cue_time and self.current_trial_type != "UNEXPECTED"),
        }

        return next_state, reward, self.done, info
