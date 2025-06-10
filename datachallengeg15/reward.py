import numpy as np
from abc import ABC, abstractmethod
from env import Environment    
from collections import deque


class BaseRewardFunction(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def __call__(self, env: Environment, done: bool, position_history: deque, **kwargs) -> float:
        pass


class RewardFunction(BaseRewardFunction):
    """Reward function with improved shaping and anti-loop mechanism for maze navigation."""
    
    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.map_diagonal = self.config.get('map_diagonal', 46)  # Default for efficiency
        self.success_reward = self.config.get('success_reward', 100.0)
        self.distance_penalty_coef = self.config.get('distance_penalty_coef', 0.1)
        self.step_penalty = self.config.get('step_penalty', 0.1)
        self.loop_penalty = self.config.get('loop_penalty', 0.5)
        self.loop_threshold = self.config.get('loop_threshold', 0.2)
        self.min_history_for_loop = self.config.get('min_history_for_loop', 3)
    
    def __call__(self, env: Environment, done: bool, position_history: deque, **kwargs) -> float:
        """Calculate reward with improved shaping and anti-loop mechanism."""
        if done:
            return self.success_reward
        
        # Distance normalization and penalties
        current_distance = np.linalg.norm(np.array(env.maze.agent_pos) - np.array(env.maze.goal_pos))
        normalized_distance = current_distance / self.map_diagonal
        reward = -normalized_distance * self.distance_penalty_coef - self.step_penalty
        
        # Simplified anti-loop penalty
        if len(position_history) >= self.min_history_for_loop:
            dists = np.linalg.norm(np.array(position_history) - env.maze.agent_pos, axis=1)
            if np.any(dists <= self.loop_threshold):
                reward -= self.loop_penalty
        
        return reward


class TunedRewardFunction(BaseRewardFunction):    
    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.map_diagonal = self.config.get('map_diagonal', 46)  # Default for efficiency
        self.success_reward = self.config.get('success_reward', 100.0)
        self.distance_penalty_coef = self.config.get('distance_penalty_coef', 0.1)
        self.step_penalty = self.config.get('step_penalty', 0.1)
        self.loop_penalty = self.config.get('loop_penalty', 0.5)
        self.loop_threshold = self.config.get('loop_threshold', 0.2)
        self.min_history_for_loop = self.config.get('min_history_for_loop', 3)
    
    def __call__(self, env: Environment, done: bool, position_history: deque, **kwargs) -> float:
        """Calculate reward with improved shaping and anti-loop mechanism."""
        if done:
            return self.success_reward * (1 / np.log(env.t + 1))
        
        # Distance normalization and penalties
        current_distance = np.linalg.norm(np.array(env.maze.agent_pos) - np.array(env.maze.goal_pos))
        normalized_distance = current_distance / self.map_diagonal
        reward = -normalized_distance * self.distance_penalty_coef - self.step_penalty
        
        # Simplified anti-loop penalty
        if len(position_history) >= self.min_history_for_loop:
            dists = np.linalg.norm(np.array(position_history) - env.maze.agent_pos, axis=1)
            if np.any(dists <= self.loop_threshold):
                reward = -self.loop_penalty
        
        return reward
