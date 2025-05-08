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
    grid = Grid(array=np.load("../datachallengeg15/grid_configs/A1_grid.npy"), start_cell=(11, 3))
    trainer = Trainer(TabularQLearningAgent, dataset, reward_fn)
    trainer.train_on_map(grid, 2_000)
    visualize_q_values(trainer.agent, grid, (11, 3), grid.target_cell)


if __name__ == "__main__":
    main()






