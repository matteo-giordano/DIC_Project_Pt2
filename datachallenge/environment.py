from shapely.geometry import Point
import random
from grid import ContinuousWorld 


class Environment:
    def __init__(self, world: ContinuousWorld, reward_fn: callable, sigma: float = 0.0):
        self.world = world
        self.info = self._reset_info()
        self.reward_fn = reward_fn
        self.sigma = sigma

    def step(self, action_idx: int):
        """Agent takes an action in one of the discrete directions."""
        current_pos = (self.world.agent.x, self.world.agent.y)
        target_pos = self.world.direction_to_point(current_pos, action_idx)

        if self.sigma > 0 and random.random() < self.sigma:
            target_pos = (
                target_pos[0] + random.gauss(0, self.sigma),
                target_pos[1] + random.gauss(0, self.sigma)
            )

        try:
            self.world.move_agent(target_pos)
        except ValueError:
            # Invalid move, no-op
            pass

        reward = self.reward_fn(self.world, self.world.agent)
        done = self.world.is_done()

        # LOG direction + distance for training use
        moved_distance = Point(current_pos).distance(self.world.agent)

        self.info["cumulative_reward"] += reward
        self.info["total_steps"] += 1
        self.info["direction_taken"] = action_idx
        self.info["distance_moved"] = moved_distance

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