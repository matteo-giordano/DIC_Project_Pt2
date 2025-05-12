import numpy as np
from agent import TabularQLearningAgent, BaseAgent, MonteCarloAgent
from environment import Environment
from dataset import Dataset
from train import Trainer
from grid import Grid
from env_viz import visualize_q_values
from reward import reward_fn


def main():
    import random
    import numpy as np
    random.seed(69)
    np.random.seed(69)

    # Choose agent
    # agent = MonteCarloAgent
    # agent_kwargs = {"epsilon": 0.8, "gamma": 0.999}
    
    agent = TabularQLearningAgent
    agent_kwargs = {"epsilon": 0.4, "gamma": 0.999, "alpha": 0.1}

    # A1 test grid
    grid = Grid(array=np.load("../datachallengeg15/grid_configs/A1_grid.npy"), start_cell=(11, 3))
    graph = grid.graph
    
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=500)
    trainer.train_on_map(grid, 10_000, 10_000)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    
    # Complicated maze
    arr = np.load("/home/barry/Desktop/uni/dataChallenge/dataChallengeG15/datachallengeg15/datasets/chico_20250509_005154/maps/map_00000.npy")
    arr[49, 1] = 3  
    start_cell = (49, 1)
    grid = Grid(array=arr, start_cell=start_cell)
    
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=1500)
    trainer.train_on_map(grid, 10_000, 10_000)
    visualize_q_values(trainer.agent, grid, start_cell, grid.target_cell)


if __name__ == "__main__":
    main()






