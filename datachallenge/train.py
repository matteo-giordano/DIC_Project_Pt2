from dataset import Dataset
from environment import Environment
from agent import BaseAgent
from grid import ContinuousWorld as Grid
from tqdm import tqdm


class Trainer:
    def __init__(self, agent_cls, reward_fn: callable, agent_kwargs: dict = {}, early_stopping_threshold: int = None):
        # For ValueIterationAgent:
        # - Must pass reward_fn as first argument when calling self.agent_cls(...)
        self.agent_cls = agent_cls
        self.agent = None
        self.reward_fn = reward_fn
        self.agent_kwargs = agent_kwargs
        self.early_stopping_threshold = early_stopping_threshold

    def train_on_dataset(self, dataset: Dataset, episodes: int):
        for i in tqdm(range(len(dataset.maps))):
            raise NotImplementedError("Not implemented")

    def train_on_map(self, grid: Grid, episodes: int, max_steps: int = 2_000, sigma: float = 0.0):
        # VI train
        if self.agent_cls.__name__ == "ValueIterationAgent":
            self.agent = self.agent_cls(self.reward_fn, grid.graph, **self.agent_kwargs)
            self.agent.solve(grid, sigma=sigma, max_iterations=episodes) # Episodes and max_iterations are the same thing in VI
            # Logging / optional inspection
            #path = self.agent.extract_policy_path(grid.start_cell, grid.target_cell)
            #valid = path[-1] == grid.target_cell if path else False
            #print(f"[VI] Finished after {self.agent.iterations} iterations.")
            #print(f"[VI] Final path length: {len(path)}, Valid: {valid}")
            return self.agent.iterations, []  # VI training does not create reward log
        
        # Q-learning and Monte Carlo train
        self.agent = self.agent_cls(grid.graph, **self.agent_kwargs)
        env = Environment(grid, self.reward_fn, sigma)
        cumulative_rewards = []

        # Early stopping variables
        unchanged_episodes = 0
        prev_path = None

        for episode in range(episodes):
        # for episode in tqdm(range(episodes)):
            _, info = self._run_episode(env, self.agent, max_steps)
            cumulative_rewards.append(info["cumulative_reward"])
            
            # Early stopping check
            if self.early_stopping_threshold is not None:
                # Get optimal path from start to end using agent's policy
                current_path = self.agent.extract_policy_path(grid.start_cell, grid.target_cell)
                
                if prev_path is not None and current_path == prev_path:
                    unchanged_episodes += 1
                    if unchanged_episodes >= self.early_stopping_threshold:
                        # print(f"Early stopping at episode {episode+1}: optimal path unchanged for {self.early_stopping_threshold} episodes")
                        episode += 1 # The counter "episode" won't automatically add 1 if we break it
                        break
                else:
                    unchanged_episodes = 0
                
                prev_path = current_path
        return episode, cumulative_rewards
    
    def evaluate_on_map(self, grid: Grid, episodes: int, max_steps: int = 2_000, sigma: float = 0.0):
        assert self.agent is not None, "Agent must be trained before evaluation"
        env = Environment(grid, self.reward_fn, sigma=sigma)
        self.agent.epsilon = 0.0
        cumulative_rewards = []
        for _ in range(episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = self.agent.take_action(state)
                next_state, _, done, info = env.step(action)
                steps += 1
                state = next_state
                if done:
                    cumulative_rewards.append(info["cumulative_reward"])    
                    break
            if not done:
                cumulative_rewards.append([None])
        return cumulative_rewards


    def _run_episode(self, env: Environment, agent: BaseAgent, max_steps: int):
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
        
        # Call end_episode if the agent has this method (for Monte Carlo agent)
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
            
        return steps, env.info

    