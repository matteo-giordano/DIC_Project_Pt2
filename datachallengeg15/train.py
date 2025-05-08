from dataset import Dataset
from environment import Environment
from agent import BaseAgent
from grid import Grid
from tqdm import tqdm


class Trainer:
    def __init__(self, agent_cls, dataset: Dataset, reward_fn: callable):
        self.agent_cls = agent_cls
        self.dataset = dataset
        self.agent = None
        self.reward_fn = reward_fn

    def train_on_dataset(self, episodes: int):
        for i in tqdm(range(len(self.dataset.maps))):
            raise NotImplementedError("Not implemented")
            grid = self.dataset.maps[i]
            self.train_on_map(grid, episodes)


    def train_on_map(self, grid: Grid, episodes: int):
        self.agent = self.agent_cls(grid.graph)
        env = Environment(grid, self.reward_fn)
        for _ in tqdm(range(episodes)):
            self._run_episode(env, self.agent)

    def _run_episode(self, env: Environment, agent: BaseAgent, max_steps=2_000):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            steps += 1
            agent.update(state, action, reward, next_state)
            state = next_state
            if done:
                break
        return steps    


    