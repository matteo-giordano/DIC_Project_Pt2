# DataChallenge - Maze Navigation Framework

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

## Dataset Builder Web Application

The project includes a web-based tool for creating and managing maze datasets with a user-friendly interface.

### Features

- Generate mazes using multiple algorithms:
  - **Prim Algorithm**: Creates perfect mazes with exactly one path between any two points
  - **Recursive Division**: Generates mazes by recursively dividing the space
  - **Wilson's Algorithm**: Creates unbiased samples of perfect mazes
  - **Terrain Generator**: Creates more organic, terrain-like structures
  - **Gaussian Noise**: Generates random mazes with controllable smoothness
  - **Manual Drawing**: Draw custom mazes by hand

- Configure generation parameters:
  - Maze dimensions (11×11 to 51×51)
  - Algorithm-specific parameters (e.g., density, thresholds)
  - Batch generation settings

- Dataset management:
  - Create datasets with mixed maze types
  - Preview generated mazes before saving
  - Ensure connectivity and uniqueness across mazes
  - Save datasets to the `./datasets` folder
  - Load and explore existing datasets

- Exploration tools:
  - Filter mazes by type and size
  - View metadata for each maze
  - Navigate through large collections

### Running the Dataset Builder

To launch the web application:

```bash
# Navigate to the project directory
cd datachallengeg15

# Run the web application
python dataset_builder_web.py
```

The application will be available at http://localhost:5000 in your web browser.

### Workflow

1. **Create New Dataset**: Select maze generator types and configure parameters
2. **Generate Sample Mazes**: Preview how mazes will look with current settings
3. **Configure Dataset**: Set batch generation parameters for the full dataset
4. **Generate Dataset**: Create the complete dataset with all configured maze types
5. **Explore Results**: Browse, filter, and inspect the generated dataset

Generated datasets are automatically saved to the `./datasets` directory and can be loaded directly into the framework using the Dataset class.

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

## Hyperparameter Optimization (HPO)

The framework includes a hyperparameter optimization module (`hpo.py`) that allows for efficient tuning of agent parameters using random search with parallel execution.

### Features

- **Random Search**: Efficiently explores the parameter space by randomly sampling parameter combinations
- **Parallel Execution**: Runs multiple trials concurrently to speed up the optimization process
- **Multiple Seeds**: Tests each parameter combination with multiple random seeds for robust evaluation
- **Configurable via YAML**: All settings controlled through a simple YAML configuration file

### Usage

```python
from hpo import HPO

# Create and run an HPO experiment using a config file
hpo = HPO("ql-hpo.yaml")
results_df = hpo.run_experiment()
```

### YAML Configuration Format

The HPO module uses YAML configuration files to define the experiment parameters. Here's an example:

```yaml
# Agent configuration
algorithm: TabularQLearningAgent  # Agent class name
reward_fn: reward_fn              # Reward function to use

# Environment configuration
map: path/to/grid_map.npy         # Path to the maze map file
start_cell: [11, 3]               # Starting position [y, x]

# Training parameters
max_episodes: 10_000              # Maximum episodes per trial
max_steps: 10_000                 # Maximum steps per episode
early_stopping_threshold: 500     # Stop if target reached consistently

# HPO parameters
max_workers: 10                   # Number of parallel processes
n_trials: 500                     # Number of random parameter combinations to try
n_seeds: 10                       # Number of random seeds per parameter combination

# Parameter ranges to search (min and max values)
algorithm_params:
    "epsilon": [0.01, 0.99]       # Exploration rate range
    "gamma": [0.01, 0.99]         # Discount factor range
    # Add other parameters as needed
```

### Results

The HPO process saves results to a CSV file containing:
- All parameter values for each trial
- Random seed used
- Number of training iterations required
- Optimal path length found
- Whether a valid path was discovered

Results are saved with a timestamp in the filename for easy tracking:
```
TabularQLearningAgent_A1_grid_reward_fn_23_12-45.csv
```

### Example: Running Hyperparameter Optimization

```python
# Run HPO experiment
hpo = HPO("ql-hpo.yaml")
results = hpo.run_experiment()

# Analyze results
best_params = results.loc[results["iters"].idxmin()]
print(f"Best parameters: {best_params}")
```

#### Reproduction of Code 

1. Path Heatmap Visualization

To visualize the path found by the agent in either `A1_grid` or `A1_grid_TOUGH` from Assignment 1, open `main.py` and uncomment the lines that define the agent you wish to reproduce. For example, to visualize the path found by Value Iteration, uncomment the following:

```python
# agent = ValueIterationAgent
# agent_kwargs = {"gamma": 0.98, "theta": 1e-5}

Then run main.py. The first figure that appears will show the path and heatmap for A1_grid. After closing that figure, a second heatmap will appear showing the path found in A1_grid_TOUGH.


2. Reproducing Experiment Figures and Additional Outputs

This section explains how to regenerate all the agents' visualizations presented in the report, as well as additional experiment figures and path visualizations not included in the report.

---

### 1. Reproduce Experiment Results (A1_grid and A1_grid_TOUGH)

#### Step 1: Configure Map and Target Cell

Open `hpo.py` and locate the `load_map(self)` function. To ensure the correct target is used (due to internal transpose operations), set:

* For **A1_grid_TOUGH**:

  ```
  arr[49, 1] = 3  # Target cell → corresponds to [1, 49] in original
  ```
* For **A1_grid**:

  ```
  arr[3, 11] = 3  # Target cell → corresponds to [11, 3] in original
  ```

#### Step 2: Configure YAML File

Open `ql-hpo.yaml` and choose the agent you wish to evaluate by setting `algorithm`. Uncomment the corresponding block under `algorithm_params` and edit hyperparameter ranges as needed.

To test both grid types:

* Set `map: A1_grid_TOUGH.npy` and `start_cell: [49, 1]` for the tough grid.
* Set `map: A1_grid.npy` and `start_cell: [3, 11]` for the normal grid.

#### Step 3: Run HPO Script

Run the experiment by executing:

```bash
python hpo.py --yaml_path ql-hpo.yaml
```

Progress bars will indicate training status. After completion, a new `.csv` file will appear in the `results/` directory.

---

### 2. Generate Experiment Figures from Results

Once the `.csv` file is created in `results/`, copy its filename and proceed as follows:

#### (1) For Q-Learning:

* Manually create an output folder:

  ```bash
  mkdir aimgs
  ```
* Open `plots3.py`, go to line 13 and replace the filename:

  ```python
  df = pd.read_csv('../results/Your_QLearning_Result.csv')
  ```
* Then run:

  ```bash
  python plots3.py
  ```
* All figures will be saved in the `aimgs/` directory.

#### (2) For Value Iteration:

* Open `plot_VI.py` and replace the file path at the top:

  ```python
  CSV_PATH = '../results/Your_VI_Result.csv'
  ```
* Run the script:

  ```bash
  python plot_VI.py
  ```
* Figures will be generated in the `aimgs_vi/` folder (created automatically).


#### (3) For Monte Carlo:

* Open `MC_plot.py` and replace the file path at the top:

  ```python
  CSV_PATH = ['../results/MC_A1_2.csv', '../results/MC_Tough_2.csv']
  ```
* Run the script:

  ```bash
  python MC_plot.py
  ```
* Figures will be generated in the ``../results/imgs/`` folder (created automatically).

