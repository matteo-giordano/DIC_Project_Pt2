import numpy as np
from agent import TabularQLearningAgent, MonteCarloAgent, ValueIterationAgent, RandomAgent
from train import Trainer
from grid import ContinuousWorld as Grid
from env_viz import visualize_q_values
from reward import reward_fn, reward_dont_revisit
from shapely.geometry import shape, Point
import json
import os
import matplotlib.pyplot as plt


def plot_continuous_result(grid, path, title="Policy Path on Continuous Maze", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot obstacles
    for poly in grid.obstacles:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='black')

    # Plot path
    if path and len(path) > 1:
        px, py = zip(*path)
        ax.plot(px, py, color='red', linewidth=2, marker='o', label='Policy Path')

    # Plot start and goal
    sx, sy = grid.start_cell
    gx, gy = grid.target_cell
    ax.plot(sx, sy, 'go', markersize=10, label='Start')
    ax.plot(gx, gy, 'ro', markersize=10, label='Goal')

    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def main():
    import random
    random.seed(12369)
    np.random.seed(12369)

    # Choose agent
    agent = RandomAgent

    # Load continuous maze JSON
    grid_name = "continuous_maze_20250602_113010.json"
    base_dir = os.path.dirname(__file__)
    grid_path = os.path.join(base_dir, "grid_configs", grid_name)

    with open(grid_path, 'r') as f:
        data = json.load(f)

    width = data["width"]
    height = data["height"]
    obstacles = [shape(geom) for geom in data["obstacles"]]

    # Choose start and goal points
    start_cell = (5.0, 5.0)
    target_cell = (95.0, 95.0)

    agent_kwargs = {"goal": target_cell}

    # Initialize continuous space world
    grid = Grid(width=width, height=height, obstacles=obstacles, start=start_cell, goal=target_cell)
    graph = grid.graph

    # Trainer setup
    trainer = Trainer(agent, reward_fn, agent_kwargs=agent_kwargs, early_stopping_threshold=250)

    # Train and evaluate
    trainer.train_on_map(grid, episodes=1000, max_steps=2000)
    cum_rewards = trainer.evaluate_on_map(grid, episodes=100, sigma=0.00)
    print("Evaluation cumulative rewards:", cum_rewards)

    # Extract and visualize policy path
    path = trainer.agent.extract_policy_path(grid.start_cell, grid.target_cell)
    print("Extracted policy path:")
    print(path)

    # Plot final figure
    plot_continuous_result(grid, path, title="RandomAgent Policy Path in Continuous Maze")


if __name__ == "__main__":
    main()