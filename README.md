# DataChallengeG15 - Maze Navigation Framework

A framework for training agents to navigate through maze environments using reinforcement learning.

## Overview

This project provides a flexible framework for training and evaluating reinforcement learning agents in maze navigation tasks. The codebase supports:

- Loading and manipulating maze datasets
- Building environment representations
- Training agents with customizable reward functions
- Visualizing agent performance and learned policies

## Code Structure

The codebase is organized into the following main modules:

- `dataset.py`: Handles loading, filtering, and manipulating maze datasets
- `grid.py`: Represents the maze grid and provides navigation functionality
- `environment.py`: Defines the reinforcement learning environment  
- `agent.py`: Contains agent implementations, including a TabularQLearningAgent
- `train.py`: Provides training utilities for agents
- `reward.py`: Defines reward functions for the environment
- `env_viz.py`: Visualization tools for the environment and agent behavior
- `main.py`: Entry point for running experiments

## Data Formats

### Grid Format

Grids are represented as 2D NumPy arrays of type `np.int8` with the following cell values:
- `0`: Empty space (navigable)
- `1`: Wall (obstacle)
- `3`: Target cell (goal)

Grid requirements:
- 2D array with odd dimensions
- Surrounded by walls on all edges
- Contains at least one target cell (value 3)

Example of creating a Grid:

```python
import numpy as np
from grid import Grid

# Create a grid from a numpy array
array = np.ones((21, 21), dtype=np.int8)  # Start with all walls
array[1:-1:2, 1:-1:2] = 0  # Create some paths
array[10, 10] = 3  # Set target cell
grid = Grid(array=array, start_cell=(1, 1))
```

### Action Format

Actions in this framework are represented as:

1. In agent implementations: 
   - Actions are represented by the coordinates of the **target cell** as a tuple `(y, x)` 
   - These are neighbor cells of the current state in the graph representation

2. When calling `Environment.step()`:
   - The action is a coordinate tuple `(y, x)` representing the cell to move to
   - This must be a valid neighbor of the current state

Example of valid actions from a state:
```python
state = (5, 5)  # Current position (y, x)
valid_actions = list(grid.graph.neighbors(state))  # e.g., [(4, 5), (6, 5), (5, 4), (5, 6)]
```

### Dataset Format

The dataset handles collections of maze maps and their metadata. Each dataset contains:

1. Maps: 2D NumPy arrays (as described in Grid Format)
2. Metadata: Dictionary containing information about each map:
   - `generator_type`: Algorithm used to generate the maze (e.g., "prim", "recursive")
   - `width`, `height`: Dimensions of the maze
   - Additional properties specific to the generation method

Datasets can be loaded from disk or created programmatically:

```python
from dataset import Dataset

# Load an existing dataset
dataset = Dataset().load("/path/to/dataset")

# Dataset folder structure:
# /path/to/dataset/
#   metadata.json          # Contains dataset info and map metadata
#   maps/
#     map_0000.npy         # Individual map files (numpy arrays)
#     map_0001.npy
#     ...
```

## Usage

### Basic Usage

```python
from agent import TabularQLearningAgent
from environment import Environment
from grid import Grid
from train import Trainer
from reward import reward_fn
from env_viz import visualize_q_values
import numpy as np

# Load a maze
grid = Grid(array=np.load("path/to/maze.npy"), start_cell=(11, 3))

# Create a trainer with the agent type and reward function
trainer = Trainer(TabularQLearningAgent, None, reward_fn)

# Train the agent on the maze
trainer.train_on_map(grid, episodes=2000)

# Visualize the learned policy
visualize_q_values(trainer.agent, grid, (11, 3), grid.target_cell)
```

### Working with Datasets

```python
from dataset import Dataset

# Load an existing dataset
dataset = Dataset().load("path/to/dataset")

# Filter dataset by criteria
filtered_dataset = dataset.filter({
    "generator_type": "prim",
    "width": 21
})

# Visualize sample maps
dataset.visualize(max_maps=4)

# Sample a random map with specific properties
map_data, metadata = dataset.sample(
    generator_type="prim",
    width=21,
    height=21,
    random_transform=True,
    add_endpoint=True
)
```

## Implementing New Agents

To implement a new agent, extend the `BaseAgent` abstract class in `agent.py`:

```python
from agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, graph, **kwargs):
        super().__init__()
        self.graph = graph
        # Initialize your agent-specific parameters
        
    def take_action(self, state):
        # Implement your action selection logic
        # Return the selected action
        pass
        
    def update(self, state, action, reward, next_state):
        # Implement your learning/update logic
        pass
        
    # Add any additional methods needed for your agent
```

Then, use your new agent with the existing framework:

```python
trainer = Trainer(MyCustomAgent, dataset, reward_fn)
trainer.train_on_map(grid, episodes=1000)
```

## API Between Modules

### Environment-Agent Interface

- `Environment.step(action)`: Takes an action and returns `(next_state, reward, done, info)`
- `Environment.reset()`: Resets the environment and returns the initial state
- `Agent.take_action(state)`: Takes the current state and returns an action
- `Agent.update(state, action, reward, next_state)`: Updates the agent based on experience

### Grid-Environment Interface

- `Grid.move_agent(action)`: Updates the agent's position based on action
- `Grid.is_done()`: Checks if the target has been reached
- `Grid.reset()`: Resets the agent to the starting position
- `Grid.build_graph()`: Constructs a graph representation of the maze

### Trainer Interface

- `Trainer.train_on_map(grid, episodes)`: Trains an agent on a specific grid
- `Trainer.train_on_dataset(episodes)`: Trains an agent on multiple maps from a dataset
- `Trainer._run_episode(env, agent, max_steps)`: Runs a single training episode

## Reward Functions

Custom reward functions can be defined in `reward.py`. They should take the grid and agent's current cell as inputs and return a floating-point reward value:

```python
def custom_reward_fn(grid, agent_cell):
    # Calculate custom reward based on grid state and agent position
    # Return a float value
    pass
```

## Visualization

The `env_viz.py` module provides visualization tools for the agent's learned policy:

```python
visualize_q_values(agent, grid, start_cell, goal_cell)
```

This shows a heatmap of Q-values and the optimal path extracted from the agent's policy.
