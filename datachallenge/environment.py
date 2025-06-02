from shapely.geometry import Point
import random
from grid import ContinuousWorld 


class Environment:
    def __init__(self, world: ContinuousWorld, reward_fn: callable, sigma: float = 0.0):
        self.world = world
        self.info = self._reset_info()
        self.reward_fn = reward_fn
        self.sigma = sigma

    def step(self, action: tuple[float, float]):
        """
        Move the agent to a new (x, y) location or a nearby one if noise is applied.
        """
        if self.sigma > 0 and random.random() < self.sigma:
            # Apply Gaussian noise to the action
            noisy_action = (
                action[0] + random.gauss(0, self.sigma),
                action[1] + random.gauss(0, self.sigma),
            )
            action = noisy_action

        try:
            self.world.move_agent(action)
        except ValueError:
            # Invalid move (into obstacle or out of bounds), stay in place
            pass

        reward = self.reward_fn(self.world, self.world.agent)
        done = self.world.is_done()

        self.info["cumulative_reward"] += reward
        self.info["total_steps"] += 1

        return (self.world.agent.x, self.world.agent.y), reward, done, self.info

    def reset(self):
        self.world.reset()
        self.info = self._reset_info()
        return (self.world.agent.x, self.world.agent.y)
    
    def render(self):
        pass  # Could be implemented using matplotlib visualization of ContinuousWorld

    @staticmethod
    def _reset_info() -> dict:
        return {
            "cumulative_reward": 0,
            "total_steps": 0,
        }