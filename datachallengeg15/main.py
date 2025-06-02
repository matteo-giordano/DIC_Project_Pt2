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

    # Choose agent, uncommit the corresponding agent name and parameters
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
    # agent = TabularQLearningAgent
    # agent_kwargs = {"epsilon": 0.4, "gamma": 0.999, "alpha": 0.1}
    agent = ValueIterationAgent
    agent_kwargs={"gamma": 0.98, "theta": 1e-5}

    # Load the grid
    grid_name = "A1_grid.npy" # Name of the grid. Change different grid files by changing this if they are in the same folder
    base_dir = os.path.dirname(__file__) # Absolute path of main.py
    # Usage: If your grid file is in ../a/b/c/$grid_name$, change the line below
    # grid_path = os.path.join(base_dir, "a", "b", "c", grid_name)
    grid_path = os.path.join(base_dir, "grid_configs", grid_name) # Absolute path of A1_grid.npy
    arr = array=np.load(grid_path)
    arr[3, 11] = 3
    #grid = Grid(array=np.load(grid_path), start_cell=(3, 11)) # Load the A1 grid
    grid = Grid(arr, start_cell=(3, 11)) # Load the A1 grid
    graph = grid.graph

    # A1 test grid
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer.train_on_map(grid, 10_000, 10_000)
    cum_rewards = trainer.evaluate_on_map(grid, 100, sigma=0.0)
    print(cum_rewards)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    print(trainer.agent.extract_policy_path(grid.start_cell, grid.target_cell))
    
    # Complicated maze
    maze_name = "continuous_maze_20250602_101644.json" # Name of the tough grid. Change different grid files by changing this if they are in the same folder
    base_dir = os.path.dirname(__file__)
    maze_path = os.path.join(base_dir, "grid_configs", maze_name) # Absolute path of A1_grid_TOUGH.npy
    arr_maze = np.load(maze_path)
    arr_maze[49, 1] = 3  
    start_cell_maze = (49, 1)
    grid_maze = Grid(array=arr_maze, start_cell=start_cell_maze)
    
    #trainer = Trainer(agent, reward_dont_revisit, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=250)
    trainer.train_on_map(grid_maze, 100_000, 1_000_000, sigma=0.0)
    visualize_q_values(trainer.agent, grid_maze, grid_maze.start_cell, grid_maze.target_cell)
    print(trainer.agent.extract_policy_path(grid_maze.start_cell, grid_maze.target_cell))
    
if __name__ == "__main__":
    main()






