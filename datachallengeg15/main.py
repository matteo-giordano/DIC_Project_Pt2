import numpy as np
from agent import TabularQLearningAgent, MonteCarloAgent
from train import Trainer
from grid import Grid
from env_viz import visualize_q_values
from reward import reward_fn, reward_dont_revisit


def main():
    import random
    random.seed(12369)
    np.random.seed(12369)

    # Choose agent
    # agent = MonteCarloAgent
    # agent_kwargs = {"epsilon": 0.9, "gamma": 0.999, "epsilon_decay": 0.995, "epsilon_min": 0.1}
    # Optimized kwargs for Monte-Carlo
    # agent_kwargs = {
    #   "epsilon": 0.0546,
    #   "gamma": 0.8140,
    #   "epsilon_decay": 0.9599,
    #   "epsilon_min": 0.0501
    # } 
    agent = TabularQLearningAgent
    agent_kwargs = {"epsilon": 0.4, "gamma": 0.999, "alpha": 0.1}

    # A1 test grid
    grid = Grid(array=np.load("../datachallengeg15/grid_configs/A1_grid.npy"), start_cell=(11, 3))
    
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer.train_on_map(grid, 10_000, 10_000)
    cum_rewards = trainer.evaluate_on_map(grid, 100, sigma=0.0)
    print(cum_rewards)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    print(trainer.agent.extract_policy_path(grid.start_cell, grid.target_cell))

    # Complicated maze
    arr = np.load("../datachallengeg15/datasets/chico_20250509_005154/maps/map_00000.npy")
    arr[49, 1] = 3  
    start_cell = (49, 1)
    grid = Grid(array=arr, start_cell=start_cell)
    
    trainer = Trainer(agent, reward_dont_revisit, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer.train_on_map(grid, 100_000, 1_000_000)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    print(trainer.agent.extract_policy_path(grid.start_cell, grid.target_cell))

if __name__ == "__main__":
    main()






