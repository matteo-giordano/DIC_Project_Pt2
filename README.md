# DataChallengeG15 - PPO Maze Navigation

A reinforcement learning framework for training agents to navigate through maze environments using Proximal Policy Optimization (PPO).

## Installation

1. Create a virtual environment:
```bash
python -m venv A2DIC
```

2. Activate the virtual environment:
```bash
# On Linux/Mac
source A2DIC/bin/activate

# On Windows
A2DIC\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Overview

This project implements a complete PPO (Proximal Policy Optimization) agent for maze navigation tasks. The framework includes:

- PPO agent with actor-critic networks
- Continuous maze environment with collision detection
- Multi-target environment support
- Training and testing utilities
- Real-time visualization

## Code Structure

The codebase consists of three main components:

- `ppo.py`: Complete PPO implementation with actor-critic networks, experience replay, and training utilities
- `env.py`: Maze environment classes with continuous movement and collision detection
- `warehouse.npy`: Pre-built warehouse maze map for training and testing

## Environment

### Maze Class

The `Maze` class provides a continuous 2D navigation environment with:

- **8-directional movement**: North, Northeast, East, Southeast, South, Southwest, West, Northwest
- **Collision detection**: Agent cannot move through walls, with sliding mechanics along obstacles
- **Continuous positioning**: Agent position stored as floating-point coordinates
- **Real-time rendering**: Matplotlib-based visualization with agent and goal tracking

### Environment Class

The base `Environment` class wraps the maze and provides:

- **State observation**: 11-dimensional state vector including:
  - Normalized agent position (2 values)
  - Normalized distance to goal (1 value) 
  - Obstacle detection in 8 directions (8 values)
- **Reward calculation**: Distance-based rewards with anti-loop penalties
- **Episode management**: Reset functionality with position randomization

### MultiTargetEnvironment Class

Extends the base environment with multiple goal positions for varied training:

- **Multiple goals**: 4 predefined target locations in the warehouse
- **Random goal selection**: Different goal chosen each episode
- **Enhanced state space**: 13-dimensional observations including goal position

## PPO Implementation

### Agent Architecture

The PPO agent consists of:

- **Actor Network**: Policy network with softmax output for action probabilities
- **Critic Network**: Value function network for state value estimation
- **Optimized Experience Buffer**: Pre-allocated NumPy arrays for fast storage and retrieval
- **Dual optimizers**: Separate Adam optimizers with weight decay regularization

### Key Features

- **Clipped surrogate objective**: Prevents large policy updates
- **Entropy regularization**: Encourages exploration
- **Gradient clipping**: Prevents exploding gradients
- **GAE (Generalized Advantage Estimation)**: Improved advantage calculation with configurable lambda
- **Mini-batch processing**: Efficient batch training with configurable batch sizes
- **Early stopping**: Automatic termination when performance targets are met
- **Configurable hyperparameters**: Easy tuning via PPOConfig class

### Training Process

The training loop includes:

1. **Experience collection**: Agent interacts with environment using optimized memory storage
2. **Mini-batch updates**: Multiple epochs of policy optimization with batch processing
3. **Progress tracking**: Episode rewards, success rates, losses, and early stopping monitoring
4. **Automatic termination**: Training stops when success rate targets are consistently achieved
5. **Model persistence**: Save/load trained models with full state preservation

## Usage

### Basic Training

```python
from datachallengeg15.ppo import train_ppo_on_maze, PPOConfig
import numpy as np

# Configure hyperparameters
config = PPOConfig(
    hidden_size=128,
    lr_actor=3e-4,           # Learning rate
    lr_critic=1e-3,          # Learning rate
    gamma=0.99,
    clip_epsilon=0.2,
    k_epochs=4,
    entropy_coef=0.01,
    memory_size=2048,        # Reduced for faster updates
    batch_size=64,           # Mini-batch size
    gae_lambda=0.95          # GAE parameter
)

# Train the agent
agent, rewards, lengths = train_ppo_on_maze(
    episodes=500,                    # Maximum episodes
    max_steps_per_episode=500,       
    update_frequency=5,              
    config=config,
    early_stop_success_rate=100.0,   
    early_stop_patience=2            
)
```

### Testing Trained Agent

```python
from datachallengeg15.ppo import test_trained_agent

# Test the trained model
test_trained_agent(
    model_path="ppo_maze_model.pth",
    episodes=2,
    max_steps=250
)
```

### Custom Environment Usage

```python
from datachallengeg15.env import Environment, MultiTargetEnvironment
import numpy as np

# Load warehouse map
warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)

# Create single-target environment
env = Environment(warehouse_map)

# Or create multi-target environment
multi_env = MultiTargetEnvironment(warehouse_map)

# Basic interaction loop
state = env.reset()
for step in range(100):
    action = np.random.randint(0, 8)  # Random action
    next_state, done = env.step(action)
    env.render()  # Visualize
    if done:
        break
    state = next_state
```

## Data Format

### Map Format

Maps are 2D NumPy arrays of type `np.int8` with:
- `0`: Empty space (navigable)
- `1`: Wall (obstacle)
- Odd dimensions with wall boundaries

### Action Format

Actions are integers 0-7 representing 8 directions:
- `0`: North, `1`: Northeast, `2`: East, `3`: Southeast
- `4`: South, `5`: Southwest, `6`: West, `7`: Northwest

### State Format

State observations are 11-dimensional vectors (13 for MultiTargetEnvironment):
- Agent position (normalized): 2 values
- Goal position (normalized, MultiTarget only): 2 values
- Distance to goal (normalized): 1 value
- Obstacle detection (8 directions): 8 values

## Configuration

### PPOConfig Parameters

```python
@dataclass
class PPOConfig:
    state_dim: int = 11          # State space dimension
    action_dim: int = 8          # Action space dimension
    hidden_size: int = 128       # Neural network hidden layer size 
    lr_actor: float = 3e-4       # Actor learning rate 
    lr_critic: float = 1e-3      # Critic learning rate 
    gamma: float = 0.99          # Discount factor
    clip_epsilon: float = 0.2    # PPO clipping parameter
    k_epochs: int = 4            # Update epochs per batch
    entropy_coef: float = 0.01   # Entropy regularization coefficient
    max_grad_norm: float = 0.5   # Gradient clipping threshold
    memory_size: int = 2048      # Experience buffer size 
    batch_size: int = 64         # Mini-batch size for training
    gae_lambda: float = 0.95     # GAE lambda parameter
```

### Early Stopping Parameters

```python
# Early stopping configuration
early_stop_success_rate: float = 100.0  # Target success rate (%)
early_stop_patience: int = 2             # Consecutive perfect windows required
```

## Reward Function

The reward system includes:

- **Goal reward**: +120 for reaching the target
- **Distance penalty**: Proportional to distance from goal (optimized calculation)
- **Step penalty**: Small negative reward per step (-0.01, reduced)
- **Loop penalty**: Additional penalty for revisiting recent positions (-0.5)

## Model Persistence

Trained models are saved as PyTorch state dictionaries containing:
- Actor and critic network weights
- Optimizer states with weight decay
- Configuration parameters
- Training metadata

```python
# Save model
agent.save("my_model.pth")

# Load model
agent.load("my_model.pth")
```
