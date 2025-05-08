from grid import Grid


class Environment:
    def __init__(self, grid: Grid, reward_fn: callable):
        self.grid = grid
        self.info = self._reset_info()
        self.reward_fn = reward_fn

    def step(self, action: int):
        self.grid.move_agent(action)

        reward = self.reward_fn(self.grid, self.grid.agent_cell)
        done = self.grid.is_done()

        self.info["cumulative_reward"] += reward
        self.info["total_steps"] += 1

        return self.grid.agent_cell, reward, done, self.info

    def reset(self):
        self.grid.reset()
        self.info = self._reset_info()
        return self.grid.agent_cell
    
    def render(self):
        pass

    @staticmethod
    def _reset_info() -> dict:
        return {
            "cumulative_reward": 0,
            "total_steps": 0,
        }
    


