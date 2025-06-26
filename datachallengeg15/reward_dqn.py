import numpy as np
from abc import ABC, abstractmethod
from collections import deque
from env_dqn import Environment

class BaseRewardFunction(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def __call__(self, env: Environment, done: bool, position_history: deque, **kwargs) -> float:
        pass

class RewardFunction(BaseRewardFunction):
    """Reward with shaping and anti-loop penalty, as in PPO."""
    def __init__(self, config: dict = None):
        super().__init__(config or {})
        self.map_diagonal = self.config.get('map_diagonal', 46)
        self.success_reward = self.config.get('success_reward', 100.0)
        self.distance_penalty_coef = self.config.get('distance_penalty_coef', 0.1)
        self.step_penalty = self.config.get('step_penalty', 0.1)
        self.loop_penalty = self.config.get('loop_penalty', 0.5)
        self.loop_threshold = self.config.get('loop_threshold', 0.2)
        self.min_history_for_loop = self.config.get('min_history_for_loop', 3)

    def __call__(self, env: Environment, done: bool, position_history: deque, **kwargs) -> float:
        if done:
            return self.success_reward
        current_distance = np.linalg.norm(np.array(env.maze.agent_pos) - np.array(env.maze.goal_pos))
        normalized_distance = current_distance / self.map_diagonal
        reward = -normalized_distance * self.distance_penalty_coef - self.step_penalty
        if len(position_history) >= self.min_history_for_loop:
            dists = np.linalg.norm(np.array(position_history) - env.maze.agent_pos, axis=1)
            if np.any(dists <= self.loop_threshold):
                reward -= self.loop_penalty
        return reward