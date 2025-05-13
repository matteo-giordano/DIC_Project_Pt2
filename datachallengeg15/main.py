import numpy as np
from agent import TabularQLearningAgent, MonteCarloAgent, ValueIterationAgent
from train import Trainer
from grid import Grid
from env_viz import visualize_q_values
from reward import reward_fn, reward_dont_revisit
import os

def main():
    import random
    random.seed(12369)
    np.random.seed(12369)

    # Choose agent
    agent = MonteCarloAgent
    agent_kwargs = {"epsilon": 0.9, "gamma": 0.999, "epsilon_decay": 0.999, "epsilon_min": 0.1}
    
    # agent = TabularQLearningAgent
    # agent_kwargs = {"epsilon": 0.4, "gamma": 0.999, "alpha": 0.1}

    # Load the grid
    base_dir = os.path.dirname(__file__) # Absolute path of main.py
    grid_path = os.path.join(base_dir, "grid_configs", "A1_grid.npy") # Absolute path of A1_grid.npy
    grid = Grid(array=np.load(grid_path), start_cell=(11, 3)) # Load the A1 grid
    graph = grid.graph

    # Value Iteration test
    trainer_VI = Trainer(
        agent_cls=ValueIterationAgent,
        reward_fn=reward_fn,
        agent_kwargs={"gamma": 0.99, "theta": 1e-4}) 
    result = trainer.plan_on_map(grid, stochasticity=0.1)
    # Print result summary
    print("Path:", result["path"])
    print("Reached Goal:", result["valid_path"])
    print("Path Length:", result["path_length"])
    print("Iterations:", result["iters"])
    visualize_q_values(trainer_VI.agent, grid, grid.start_cell, grid.target_cell)

    # A1 test grid
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer.train_on_map(grid, 10_000, 10_000)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    print(trainer.agent.extract_policy_path(grid.start_cell, grid.target_cell))

    # Complicated maze
    arr = np.load("/home/barry/Desktop/uni/dataChallenge/dataChallengeG15/datachallengeg15/datasets/chico_20250509_005154/maps/map_00000.npy")
    arr[49, 1] = 3  
    start_cell = (49, 1)
    grid = Grid(array=arr, start_cell=start_cell)
    
    trainer = Trainer(agent, reward_dont_revisit, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer.train_on_map(grid, 10_000, 1_000_000)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    print(trainer.agent.extract_policy_path(grid.start_cell, grid.target_cell))

if __name__ == "__main__":
    main()






