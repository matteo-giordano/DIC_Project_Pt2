import numpy as np
from agent import TabularQLearningAgent, BaseAgent
from environment import Environment
from dataset import Dataset
from train import Trainer
from grid import Grid
from env_viz import visualize_q_values
from reward import reward_fn


def main():
    dataset = None

    # A1 test grid
    grid = Grid(array=np.load("../datachallengeg15/grid_configs/A1_grid.npy"), start_cell=(11, 3))
    trainer = Trainer(TabularQLearningAgent, dataset, reward_fn, agent_kwargs={"alpha": 0.1, "epsilon": 0.2, "gamma": 0.9})
    trainer.train_on_map(grid, 10_000, 10_000)
    visualize_q_values(trainer.agent, grid, grid.start_cell, grid.target_cell)
    
    # Complicated maze
    arr = np.load("/home/barry/Desktop/uni/dataChallenge/dataChallengeG15/datachallengeg15/datasets/chico_20250509_005154/maps/map_00000.npy")
    arr[49, 1] = 3  # (49, 1) means that the target cell is at (1, 49)
    start_cell = (49, 1)
    grid = Grid(array=arr, start_cell=start_cell)
    
    trainer = Trainer(TabularQLearningAgent, dataset, reward_fn, agent_kwargs={"alpha": 0.1, "epsilon": 0.4, "gamma": 0.9999})
    trainer.train_on_map(grid, 5_000, 5_000)
    visualize_q_values(trainer.agent, grid, start_cell, grid.target_cell)


if __name__ == "__main__":
    main()






