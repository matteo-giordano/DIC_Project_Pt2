# DataChallengeG15 - PPO Maze Navigation

A reinforcement learning framework for training agents to navigate through maze environments using Proximal Policy Optimization (PPO) with a modern configuration-based training system.

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

This project implements a complete PPO (Proximal Policy Optimization) agent for maze navigation tasks with a modern, configuration-driven architecture. The framework includes:

- **Configuration-based training**: YAML config files for easy hyperparameter management
- **Modular architecture**: Separate classes for environments, rewards, and training
- **PPO agent**: Actor-critic networks with GAE and experience replay
- **Multiple environments**: Single and multi-target maze navigation
- **Live tracking**: Real-time visualization of training metrics
- **Extensible design**: Base classes for custom environments and reward functions

## Code Structure

```
datachallengeg15/
├── main.py              # Entry point with argument parsing
├── trainer.py           # PPOTrainer class for training/testing
├── ppo.py              # PPO agent implementation
├── env.py              # Environment classes (Maze, Environment, MultiTargetEnvironment)
├── reward.py           # Reward function classes
├── viz.py              # Live tracking visualization
├── config.yaml         # Configuration file
└── warehouse.npy       # Pre-built warehouse maze map
```

## Usage

### Training

Train an agent using the configuration file:

```bash
# Train with default config
python datachallengeg15/main.py

# Train with custom config
python datachallengeg15/main.py -c path/to/config.yaml

# Test a trained model
python datachallengeg15/main.py -t
```

### Configuration File Structure

The training is controlled by a YAML configuration file (`config.yaml`):

```yaml
# Global settings
seed: 69                           # Random seed for reproducibility
model_path: "ppo_maze_model.pth"   # Path to save/load model

# PPO Agent Configuration
PPO:
  state_dim: 5                     # State space dimension (auto-set by environment)
  action_dim: 8                    # Action space dimension (8 directions)
  hidden_size: 128                 # Neural network hidden layer size
  lr_actor: 0.0005                 # Actor learning rate
  lr_critic: 0.0005                # Critic learning rate
  gamma: 0.99                      # Discount factor
  clip_epsilon: 0.2                # PPO clipping parameter
  k_epochs: 4                      # Update epochs per batch
  entropy_coef: 0.01               # Entropy regularization coefficient
  max_grad_norm: 0.5               # Gradient clipping threshold
  memory_size: 4096                # Experience buffer size
  batch_size: 64                   # Mini-batch size for training
  gae_lambda: 0.95                 # GAE lambda parameter
  recent_history_length: 16        # Position history for loop detection

# Reward Function Configuration
reward:
  name: RewardFunction             # Reward function class name
  args:
    success_reward: 120.0          # Reward for reaching goal
    distance_penalty_coef: 0.1     # Distance-based penalty coefficient
    step_penalty: 0.01             # Per-step penalty
    loop_penalty: 0.5              # Penalty for revisiting positions

# Training Configuration
trainer:
  episodes: 1000                   # Maximum training episodes
  max_steps: 250                   # Maximum steps per episode
  update_frequency: 5              # Update agent every N episodes
  early_stop_success_rate: 100.0   # Success rate threshold for early stopping
  early_stop_patience: 10          # Consecutive windows needed for early stop
  enable_live_tracking: true       # Enable real-time visualization
  test_episodes: 2                 # Number of episodes for testing

# Environment Configuration
env:
  map_path: "datachallengeg15/warehouse.npy"  # Path to maze map
  name: MultiTargetEnvironment     # Environment class name
  step_size: 0.4                   # Agent movement step size
  goals: [[15.8, 37], [17, 18], [5, 6], [2, 35]]  # Goals (MultiTargetEnvironment only)
  start_pos: [15.8, 37]           # Start position (Environment only)
```

### Configuration Parameters

#### PPO Parameters
- **state_dim/action_dim**: Automatically set by environment, but can be overridden
- **hidden_size**: Size of neural network hidden layers (64-256 typical)
- **lr_actor/lr_critic**: Learning rates (1e-5 to 1e-3 typical)
- **gamma**: Discount factor for future rewards (0.9-0.999)
- **clip_epsilon**: PPO clipping parameter (0.1-0.3)
- **k_epochs**: Number of optimization epochs per update (3-10)
- **entropy_coef**: Exploration encouragement (0.001-0.1)
- **memory_size**: Experience buffer size (1024-8192)
- **batch_size**: Mini-batch size for training (32-128)
- **gae_lambda**: GAE parameter for advantage estimation (0.9-0.99)

#### Training Parameters
- **episodes**: Maximum training episodes
- **max_steps**: Maximum steps per episode
- **update_frequency**: How often to update the agent (episodes)
- **early_stop_success_rate**: Success rate threshold for early stopping (%)
- **early_stop_patience**: Consecutive evaluation windows needed
- **enable_live_tracking**: Real-time training visualization

## Environments

### Available Environments

1. **Environment**: Single-target maze navigation
   - Fixed start and goal positions
   - 5-dimensional state space
   - Basic distance-based rewards

2. **MultiTargetEnvironment**: Multi-target maze navigation
   - Multiple goal positions, randomly selected each episode
   - 5-dimensional state space with goal information
   - Enhanced training variety

### Environment Features

- **Continuous movement**: Floating-point agent positions
- **8-directional actions**: North, Northeast, East, Southeast, South, Southwest, West, Northwest
- **Collision detection**: Wall sliding mechanics
- **Real-time rendering**: Matplotlib-based visualization

## Implementing Custom Environments

To create a custom environment, extend the base `Environment` class:

```python
from datachallengeg15.env import Environment
import numpy as np

class CustomEnvironment(Environment):
    def __init__(self, array: np.ndarray, step_size=0.4, **kwargs):
        super().__init__(array, step_size)
        # Add custom initialization
        self.custom_param = kwargs.get('custom_param', 1.0)
    
    def _get_observation(self):
        """Override to provide custom state representation."""
        # Get base observation
        base_obs = super()._get_observation()
        
        # Add custom features
        custom_features = np.array([self.custom_param])
        
        return np.concatenate([base_obs, custom_features])
    
    def reset(self):
        """Override to customize reset behavior."""
        # Custom reset logic
        self.maze.agent_pos = self._get_random_start_position()
        self.maze.goal_pos = self._get_random_goal_position()
        return self._get_observation()
    
    def is_done(self):
        """Override to customize termination conditions."""
        # Custom termination logic
        distance = np.linalg.norm(self.maze.agent_pos - self.maze.goal_pos)
        return distance <= self.maze.goal_radius
```

### Using Custom Environments

1. **Add to config file**:
```yaml
env:
  name: CustomEnvironment
  map_path: "path/to/map.npy"
  step_size: 0.4
  custom_param: 2.0  # Custom parameters
```

2. **Update state dimension** in PPO config:
```yaml
PPO:
  state_dim: 6  # Adjust based on your observation space
```

## Implementing Custom Reward Functions

To create a custom reward function, extend the `BaseRewardFunction` class:

```python
from datachallengeg15.reward import BaseRewardFunction
from datachallengeg15.env import Environment
from collections import deque
import numpy as np

class CustomRewardFunction(BaseRewardFunction):
    def __init__(self, config: dict = None):
        super().__init__(config or {})
        # Initialize custom parameters from config
        self.success_reward = self.config.get('success_reward', 100.0)
        self.custom_penalty = self.config.get('custom_penalty', 1.0)
    
    def __call__(self, env: Environment, done: bool, position_history: deque, **kwargs) -> float:
        """Calculate reward based on current state."""
        if done:
            return self.success_reward
        
        # Custom reward logic
        distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
        
        # Example: Reward based on progress toward goal
        reward = -distance * 0.1
        
        # Example: Custom penalty for specific conditions
        if self._check_custom_condition(env):
            reward -= self.custom_penalty
        
        return reward
    
    def _check_custom_condition(self, env: Environment) -> bool:
        """Custom condition checking."""
        # Implement your custom logic
        return False
```

### Using Custom Reward Functions

Add to config file:
```yaml
reward:
  name: CustomRewardFunction
  args:
    success_reward: 150.0
    custom_penalty: 2.0
```

## Base Classes Reference

### Environment Base Class

Key methods to override:
- `_get_observation()`: Return state representation
- `reset()`: Reset environment to initial state
- `is_done()`: Check if episode is complete
- `step(action)`: Execute action and return (next_state, done)

### Reward Function Base Class

Key methods to implement:
- `__call__(env, done, position_history, **kwargs)`: Calculate reward
- `__init__(config)`: Initialize from configuration

## Live Tracking

Enable real-time training visualization:

```yaml
trainer:
  enable_live_tracking: true
```

Features:
- Real-time plots of episode rewards, lengths, and losses
- Automatic plot saving with timestamps
- Configurable update intervals and window sizes

## Model Persistence

Models are automatically saved as PyTorch state dictionaries containing:
- Actor and critic network weights
- Optimizer states
- Configuration parameters
- Training metadata

```python
# Models are saved automatically during training
# Load for testing:
python datachallengeg15/main.py -t
```

## Examples

### Basic Training
```bash
# Train with default settings
python datachallengeg15/main.py

# Train with custom config
python datachallengeg15/main.py -c my_config.yaml
```

### Testing
```bash
# Test trained model
python datachallengeg15/main.py -t

# Test with custom config
python datachallengeg15/main.py -c my_config.yaml -t
```

### Custom Configuration
Create a custom config file with your preferred settings:

```yaml
seed: 42
model_path: "my_model.pth"

PPO:
  hidden_size: 256
  lr_actor: 0.0003
  lr_critic: 0.001
  memory_size: 8192

trainer:
  episodes: 2000
  max_steps: 500
  enable_live_tracking: false

env:
  name: Environment
  step_size: 0.3
  start_pos: [10, 10]
```

## Troubleshooting

### Common Issues

1. **State dimension mismatch**: Ensure `PPO.state_dim` matches your environment's observation space
2. **Memory issues**: Reduce `memory_size` or `batch_size` for limited RAM
3. **Slow training**: Disable live tracking or reduce update frequency
4. **Poor performance**: Adjust learning rates, increase hidden size, or modify reward function

### Performance Tips

- Use GPU acceleration when available (automatic detection)
- Adjust `update_frequency` based on environment complexity
- Tune `early_stop_success_rate` and `early_stop_patience` for your task
- Experiment with different reward function parameters
