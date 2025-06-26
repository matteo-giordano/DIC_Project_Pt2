from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque

class Maze:
    def __init__(self, array: np.ndarray, step_size=0.4):
        if not self._validate_array(array):
            raise ValueError("Invalid array")
        self.array = array.copy()
        self.map_height, self.map_width = self.array.shape

        self.action_map = {
            0: np.array([1., 0.]),    # North
            1: np.array([1., 1.]),    # North-East
            2: np.array([0., 1.]),    # East
            3: np.array([-1., 1.]),   # South-East
            4: np.array([-1., 0.]),   # South
            5: np.array([-1., -1.]),  # South-West
            6: np.array([0., -1.]),   # West
            7: np.array([1., -1.]),   # North-West
        }

        self.step_size = step_size
        self.goal_radius = 1.
        self.agent_radius = 0.15

        # Agent & Goal positions stored as numpy arrays
        self.agent_pos = np.array([1.5, 22.5])
        self.goal_pos = np.array(self.array.shape) - np.array([2.5, 2.5])

        # Rendering attributes
        self.fig = None
        self.ax = None
        self.agent_circle = None
        self.goal_circle = None

        # Pre-compute action vectors for efficiency
        self.action_vectors = np.array([self.action_map[i] for i in range(8)]) * self.step_size

    def step(self, action: int):
        action_vec = self.action_vectors[action]
        new_pos = self.agent_pos + action_vec
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            self.agent_pos = self._slide_along_wall(action_vec)

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.imshow(
                np.flipud(self.array), cmap='gray_r', origin='lower',
                extent=[0, self.map_width, 0, self.map_height]
            )
            self.agent_circle = patches.Circle(
                (self.agent_pos[1], self.agent_pos[0]), self.agent_radius,
                color='blue', alpha=0.8
            )
            self.ax.add_patch(self.agent_circle)
            self.goal_circle = patches.Circle(
                (self.goal_pos[1], self.goal_pos[0]), self.goal_radius,
                color='red', alpha=0.6
            )
            self.ax.add_patch(self.goal_circle)
            self.ax.set_xlim(0, self.map_width)
            self.ax.set_ylim(0, self.map_height)
            self.ax.set_aspect('equal')
            self.ax.set_title('Maze Environment')
            self.ax.grid(True, alpha=0.3)
            plt.tight_layout()
        else:
            self.agent_circle.center = (self.agent_pos[1], self.agent_pos[0])
            self.goal_circle.center = (self.goal_pos[1], self.goal_pos[0])
        plt.draw()
        plt.pause(0.01)

    def _validate_array(self, array: np.ndarray):
        assert array.ndim == 2
        assert array.dtype == np.int8
        assert array.min() >= 0 and array.max() <= 3
        assert np.all(array[0, :] == 1) and np.all(array[-1, :] == 1)
        assert np.all(array[:, 0] == 1) and np.all(array[:, -1] == 1)
        return True

    def _is_valid_position(self, position: np.ndarray) -> bool:
        y, x = position
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        sample_y = y + self.agent_radius * np.sin(angles)
        sample_x = x + self.agent_radius * np.cos(angles)
        grid_y = -np.floor(sample_y).astype(int) - 1
        grid_x = np.floor(sample_x).astype(int)
        return not np.any(self.array[grid_y, grid_x] == 1)

    def _are_valid_positions(self, positions: np.ndarray) -> np.ndarray:
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        y_coords = positions[:, 0][:, None]
        x_coords = positions[:, 1][:, None]
        sample_y = y_coords + self.agent_radius * np.sin(angles)
        sample_x = x_coords + self.agent_radius * np.cos(angles)
        grid_y = (-np.floor(sample_y) - 1).astype(int)
        grid_x = np.floor(sample_x).astype(int)
        collisions = self.array[grid_y, grid_x] == 1
        return ~np.any(collisions, axis=1)

    def _slide_along_wall(self, action_vec: np.ndarray) -> np.ndarray:
        current_pos = self.agent_pos
        if np.linalg.norm(action_vec) == 0:
            return current_pos
        direction = action_vec / np.linalg.norm(action_vec)
        min_dist, max_dist = 0.0, np.linalg.norm(action_vec)
        best_pos = current_pos
        while max_dist - min_dist > 0.01:
            mid = (min_dist + max_dist) / 2
            test_pos = current_pos + direction * mid
            if self._is_valid_position(test_pos):
                min_dist = mid
                best_pos = test_pos
            else:
                max_dist = mid
        return best_pos
    
    def is_done(self) -> bool:
        """Returns True if the agent is within goal_radius of the goal."""
        return np.linalg.norm(self.agent_pos - self.goal_pos) <= self.goal_radius



class Environment:
    """Wrapper that loads maze, injects goal and reward function."""
    def __init__(
        self,
        map_path: str,
        goal_pos: list = None,
        reward_fn = None,
        step_size: float = 0.4,
        max_steps: int = 200
    ):
        array = np.load(map_path).astype(np.int8)
        self.maze = Maze(array, step_size)
        h, w = self.maze.map_height, self.maze.map_width
        self.map_diagonal_norm = np.linalg.norm([h, w])
        if goal_pos is not None:
            self.maze.goal_pos = np.array(goal_pos)
        self.reward_fn = reward_fn
        self.position_history = deque(maxlen=50)
        self.t = 0
        self.max_steps = max_steps

    def _get_observation(self):
        # — copy/paste from datachallengeg15/env.py —
        agent_pos = self.maze.agent_pos.copy()
        goal_pos  = self.maze.goal_pos.copy()

        # Normalize positions by map diagonal
        agent_norm = agent_pos / self.map_diagonal_norm
        goal_norm  = goal_pos / self.map_diagonal_norm

        # Build your feature vector however you like,
        # e.g. concat([agent_norm, goal_norm, distance, etc.])
        observation = np.concatenate([
            agent_norm,
            goal_norm,
            [np.linalg.norm(goal_pos - agent_pos) / self.map_diagonal_norm]
        ])
        return observation

    def reset(self):
        self.position_history.clear()
        self.t = 0
        return self._get_observation()

    def step(self, action: int):
        """Step and compute reward via injected RewardFunction."""
        self.t += 1
        obs_before = self.maze.agent_pos.copy()
        self.maze.step(action)
        self.position_history.append(tuple(obs_before))

        goal_hit = self.is_done()
        timeout = (self.t >= self.max_steps)
        done = goal_hit or timeout
        
        if self.reward_fn is None:
            raise RuntimeError("Reward function missing")
        reward = self.reward_fn(self, done, self.position_history)
        obs = self._get_observation()

        info = {
            'goal_reached': bool(goal_hit),
            'timeout': bool(timeout)
        }
        return obs, reward, done, info

    def render(self):
        self.maze.render()

    def is_done(self) -> bool:
        return self.maze.is_done()


class MultiTargetEnvironment(Environment):
    def __init__(self, map_path: str, step_size=0.4, reward_fn=None):
        super().__init__(map_path, reward_fn=reward_fn, step_size=step_size)
        self.goals = [
            np.array([16, 37]), np.array([17, 18]),
            np.array([5, 6]), np.array([2, 35])
        ]
        self.maze.goal_pos = self.goals[np.random.randint(len(self.goals))]

    def reset(self):
        old = self.maze.goal_pos.copy()
        while True:
            new = self.goals[np.random.randint(len(self.goals))]
            if not np.array_equal(new, old):
                self.maze.goal_pos = new
                break
        return super()._get_observation()


if __name__ == "__main__":
    from time import sleep, time
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Environment(warehouse_map)
    t = time()
    for _ in range(10000):
        action = np.random.randint(0, 8)
        env.step(action)
    e = time() - t
    print(f"Execution time: {e:.4f} seconds")
