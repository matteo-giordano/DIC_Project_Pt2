from dataset import Dataset
from environment import Environment
from agent import BaseAgent
from grid import Grid
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

    def train_on_map(self, grid: Grid, episodes: int, max_steps: int = 2_000):
        self.agent = self.agent_cls(grid.graph, **self.agent_kwargs)
        env = Environment(grid, self.reward_fn)
        
        # Early stopping variables
        unchanged_episodes = 0
        prev_path = None
        
        for episode in range(episodes):
        # for episode in tqdm(range(episodes)):
            self._run_episode(env, self.agent, max_steps)
            
            # Early stopping check
            if self.early_stopping_threshold is not None:
                # Get optimal path from start to end using agent's policy
                current_path = self.agent.extract_policy_path(grid.start_cell, grid.target_cell)
                
                if prev_path is not None and current_path == prev_path:
                    unchanged_episodes += 1
                    if unchanged_episodes >= self.early_stopping_threshold:
                        # print(f"Early stopping at episode {episode+1}: optimal path unchanged for {self.early_stopping_threshold} episodes")
                        break
                else:
                    unchanged_episodes = 0
                
                prev_path = current_path
        return episode
    
    def plan_on_map(self, grid, sigma=0.0, max_iterations = 5000):
        """
        Runs Value Iteration on the given map using the ValueIterationAgent.

        Args:
            grid (Grid): the map environment (must contain .graph and .target_cell)
            sigma (float): deviation probability from intended action
            max_iterations (int): maximum number of iterations allowed.
        Returns:
            dict: {
                "path": List[Tuple[int, int]],
                "path_length": int,
                "valid_path": bool,
                "iters": int
            }
        """
        # Initialize the agent with parameters like gamma, theta
        self.agent = self.agent_cls(
            self.reward_fn,
            grid.graph,
            **self.agent_kwargs)

        # Run value iteration (with sigma)
        self.agent.solve(grid, sigma=sigma, max_iterations = max_iterations)

        # Extract policy path
        path = self.agent.extract_policy_path(grid.start_cell, grid.target_cell)

        # Return result summary
        return {
            "path": path,
            "path_length": len(path),
            "valid_path": path[-1] == grid.target_cell if path else False,
            "iters": getattr(self.agent, "iterations", -1)  # optional count from solve()
        }


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
            
        return steps    

    