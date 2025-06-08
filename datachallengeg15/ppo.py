"""
PPO (Proximal Policy Optimization) implementation for maze navigation.

This module provides a complete PPO implementation with actor-critic networks,
experience replay, and training and evaluation utilities for maze environments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, List, Optional
from dataclasses import dataclass
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
from datetime import datetime


@dataclass
class PPOConfig:
    """Configuration for PPO hyperparameters."""
    state_dim: int = 13
    action_dim: int = 8
    hidden_size: int = 128  
    lr_actor: float = 3e-4  
    lr_critic: float = 1e-3
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    k_epochs: int = 4
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    memory_size: int = 2048  
    batch_size: int = 64
    gae_lambda: float = 0.95 
    recent_history_length: int = 16


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ActorNetwork(nn.Module):
    """Actor network for policy approximation."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(x), dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for value function approximation."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOMemory:
    """Experience replay buffer for PPO with pre-allocated arrays for speed."""
    def __init__(self, max_size: int = 2048, obs_dim: int = 13):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate arrays for better performance
        self.states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.action_logprobs = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.states_next = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
    
    def store(self, state: np.ndarray, action: int, action_logprob: float, 
              reward: float, state_next: np.ndarray, done: bool) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.action_logprobs[self.ptr] = action_logprob
        self.rewards[self.ptr] = reward
        self.states_next[self.ptr] = state_next
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get_all(self) -> Tuple[np.ndarray, ...]:
        """Return all stored transitions as numpy arrays."""
        idx = slice(0, self.size)
        return (self.states[idx], self.actions[idx], self.action_logprobs[idx],
                self.rewards[idx], self.states_next[idx], self.dones[idx])
    
    def clear(self) -> None:
        """Clear all stored transitions."""
        self.ptr = 0
        self.size = 0
    
    def __len__(self) -> int:
        return self.size


class PPO:
    """Proximal Policy Optimization agent"""
    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = ActorNetwork(self.config.state_dim, self.config.hidden_size, self.config.action_dim).to(self.device)
        self.critic = CriticNetwork(self.config.state_dim, self.config.hidden_size).to(self.device)
        self.actor_old = ActorNetwork(self.config.state_dim, self.config.hidden_size, self.config.action_dim).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Initialize optimizers with weight decay for regularization
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor, weight_decay=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic, weight_decay=1e-5)
        
        self.memory = PPOMemory(self.config.memory_size, self.config.state_dim)
        print(f"PPO Agent initialized on device: {self.device}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor_old(state_tensor)
            
        if training:
            # Sample action from probability distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action).item()
        else:
            return torch.argmax(action_probs).item(), 0.0
    
    def store_transition(self, state: np.ndarray, action: int, action_logprob: float, 
                        reward: float, state_next: np.ndarray, done: bool) -> None:
        """Store transition in experience buffer."""
        self.memory.store(state, action, action_logprob, reward, state_next, done)
    
    def _compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                               values_next: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = values_next[t] * (~dones[t])
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (~dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns
    
    def update(self) -> Tuple[float, float]:
        states, actions, action_logprobs, rewards, states_next, dones = self.memory.get_all()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_action_logprobs = torch.FloatTensor(action_logprobs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        states_next = torch.FloatTensor(np.array(states_next)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Calculate advantages using GAE
        with torch.no_grad():
            values = self.critic(states).squeeze()
            values_next = self.critic(states_next).squeeze()
            advantages, returns = self._compute_gae_advantages(rewards, values, values_next, dones)
        
        total_actor_loss = total_critic_loss = 0.0
        
        # Mini-batch training for better efficiency
        batch_size = min(self.config.batch_size, len(states))
        indices = torch.randperm(len(states))
        
        for _ in range(self.config.k_epochs):
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_action_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy predictions
                action_probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_action_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # Calculate ratio (importance sampling)
                ratio = torch.exp(new_action_logprobs - batch_old_logprobs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                
                # Actor loss with entropy bonus
                actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy.mean()
                critic_loss = F.mse_loss(self.critic(batch_states).squeeze(), batch_returns)
                
                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
        
        # Copy new weights to old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.memory.clear()
        
        num_updates = self.config.k_epochs * ((len(states) + batch_size - 1) // batch_size)
        return total_actor_loss / num_updates, total_critic_loss / num_updates
    
    def save(self, filepath: str) -> None:
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config.__dict__
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


class LiveTracker:
    """Live tracking and plotting of training metrics."""
    
    def __init__(self, update_interval: int = 50, window_size: int = 100):
        self.update_interval = update_interval
        self.window_size = window_size
        
        # Data storage
        self.episodes = []
        self.rewards = []
        self.lengths = []
        self.actor_losses = []
        self.critic_losses = []
        
        # Moving averages for smoothing
        self.reward_ma = []
        self.length_ma = []
        self.actor_loss_ma = []
        self.critic_loss_ma = []
        
        # Plot setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('PPO Training Metrics - Live Tracking', fontsize=14, fontweight='bold')
        
        # Configure subplots
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Steps')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('Actor Loss')
        self.axes[1, 0].set_xlabel('Update')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Critic Loss')
        self.axes[1, 1].set_xlabel('Update')
        self.axes[1, 1].set_ylabel('Loss')
        self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
        
        self.update_count = 0
    
    def add_episode_data(self, episode: int, reward: float, length: int):
        """Add episode data for tracking."""
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.lengths.append(length)
        
        # Calculate moving averages
        if len(self.rewards) >= self.window_size:
            self.reward_ma.append(np.mean(self.rewards[-self.window_size:]))
            self.length_ma.append(np.mean(self.lengths[-self.window_size:]))
        else:
            self.reward_ma.append(np.mean(self.rewards))
            self.length_ma.append(np.mean(self.lengths))
    
    def add_loss_data(self, actor_loss: float, critic_loss: float):
        """Add loss data for tracking."""
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
        # Calculate moving averages for losses
        if len(self.actor_losses) >= self.window_size:
            self.actor_loss_ma.append(np.mean(self.actor_losses[-self.window_size:]))
            self.critic_loss_ma.append(np.mean(self.critic_losses[-self.window_size:]))
        else:
            self.actor_loss_ma.append(np.mean(self.actor_losses))
            self.critic_loss_ma.append(np.mean(self.critic_losses))
        
        self.update_count += 1
    
    def update_plots(self):
        """Update all plots with current data."""
        if len(self.episodes) == 0:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Rewards plot
        self.axes[0, 0].plot(self.episodes, self.rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        if len(self.reward_ma) > 0:
            self.axes[0, 0].plot(self.episodes, self.reward_ma, 'b-', linewidth=2, label=f'MA({self.window_size})')
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].legend()
        
        # Lengths plot
        self.axes[0, 1].plot(self.episodes, self.lengths, 'g-', alpha=0.3, linewidth=0.5, label='Raw')
        if len(self.length_ma) > 0:
            self.axes[0, 1].plot(self.episodes, self.length_ma, 'g-', linewidth=2, label=f'MA({self.window_size})')
        self.axes[0, 1].set_title('Episode Lengths')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Steps')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].legend()
        
        # Actor loss plot
        if len(self.actor_losses) > 0:
            updates = list(range(1, len(self.actor_losses) + 1))
            self.axes[1, 0].plot(updates, self.actor_losses, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
            if len(self.actor_loss_ma) > 0:
                self.axes[1, 0].plot(updates, self.actor_loss_ma, 'r-', linewidth=2, label=f'MA({self.window_size})')
            self.axes[1, 0].set_title('Actor Loss')
            self.axes[1, 0].set_xlabel('Update')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].grid(True, alpha=0.3)
            self.axes[1, 0].legend()
        
        # Critic loss plot
        if len(self.critic_losses) > 0:
            updates = list(range(1, len(self.critic_losses) + 1))
            self.axes[1, 1].plot(updates, self.critic_losses, 'm-', alpha=0.3, linewidth=0.5, label='Raw')
            if len(self.critic_loss_ma) > 0:
                self.axes[1, 1].plot(updates, self.critic_loss_ma, 'm-', linewidth=2, label=f'MA({self.window_size})')
            self.axes[1, 1].set_title('Critic Loss')
            self.axes[1, 1].set_xlabel('Update')
            self.axes[1, 1].set_ylabel('Loss')
            self.axes[1, 1].grid(True, alpha=0.3)
            self.axes[1, 1].legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to allow plot update
    
    def should_update(self, episode: int) -> bool:
        """Check if plots should be updated based on interval."""
        return episode % self.update_interval == 0
    
    def close(self):
        """Close the plotting window."""
        plt.ioff()
        plt.close(self.fig)
    
    def save_plot(self, filepath: Optional[str] = None) -> str:
        """Save the current plot to file with timestamp."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"ppo_training_metrics_{timestamp}.png"
        
        # Ensure we have the latest plot
        self.update_plots()
        
        # Save with high quality
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Training metrics plot saved to: {filepath}")
        return filepath
    
    def close_and_save(self, filepath: Optional[str] = None) -> str:
        """Save the plot and then close the window."""
        saved_path = self.save_plot(filepath)
        self.close()
        return saved_path


def train_ppo(episodes: int = 2000, max_steps_per_episode: int = 1000, 
                     update_frequency: int = 10, config: Optional[PPOConfig] = None,
                     early_stop_success_rate: float = 100.0, early_stop_patience: int = 3,
                     enable_live_tracking: bool = True, reward_config: dict = None) -> Tuple[PPO, List[float], List[int]]:
    """Train PPO agent on maze environment with early stopping and optional live tracking."""
    from env import Environment, MultiTargetEnvironment
    
    # Load the warehouse map
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = MultiTargetEnvironment(warehouse_map)
    
    # Initialize PPO agent and tracking variables
    agent = PPO(config)
    episode_rewards, episode_lengths = [], []
    success_count = 0
    consecutive_perfect_windows = 0
    
    # Initialize reward function
    reward_fn = MazeRewardFunction(reward_config)
    
    # Initialize live tracker if enabled
    tracker = None
    if enable_live_tracking:
        tracker = LiveTracker(update_interval=update_frequency, window_size=50)
        
    print("Starting PPO training on maze environment...")
    print(f"Episodes: {episodes}, Max steps: {max_steps_per_episode}")
    print(f"Update frequency: {update_frequency}")
    print(f"Live tracking: {'Enabled' if enable_live_tracking else 'Disabled'}")
    print(f"Early stopping: {early_stop_success_rate}% success rate for {early_stop_patience} consecutive windows")
    print("-" * 60)
    
    try:
        for episode in tqdm(range(episodes), desc="Training"):
            state = env.reset()
            episode_reward = 0.0
            episode_length = 0
            position_history = deque(maxlen=config.recent_history_length)
            
            for step in range(max_steps_per_episode):
                # Select action
                action, action_logprob = agent.select_action(state, training=True)
                
                # Take step in environment
                next_state, done = env.step(action)
                
                # Calculate reward
                current_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
                reward = reward_fn(env, done, position_history, current_distance=current_distance)
                position_history.append(env.maze.agent_pos)

                # Store transition
                agent.store_transition(state, action, action_logprob, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Add episode data to tracker
            if tracker:
                tracker.add_episode_data(episode + 1, episode_reward, episode_length)
            
            # Update policy
            if (episode + 1) % update_frequency == 0 and len(agent.memory) > 0:
                actor_loss, critic_loss = agent.update()
                
                # Add loss data to tracker
                if tracker:
                    tracker.add_loss_data(actor_loss, critic_loss)
                
                # Print progress less frequently for speed
                if (episode + 1) % (update_frequency * 4) == 0:
                    recent_rewards = episode_rewards[-update_frequency*4:]
                    recent_lengths = episode_lengths[-update_frequency*4:]
                    avg_reward = np.mean(recent_rewards)
                    avg_length = np.mean(recent_lengths)
                    success_rate = success_count / (episode + 1) * 100
                    recent_success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards) * 100
                    
                    if not tracker:
                        print(f"Episode {episode + 1}/{episodes}")
                        print(f"  Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
                        print(f"  Success Rate: {success_rate:.1f}% (Recent: {recent_success_rate:.1f}%)")
                        print(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
                    
                    if recent_success_rate >= early_stop_success_rate:
                        consecutive_perfect_windows += 1
                        print(f"  Perfect success rate achieved! ({consecutive_perfect_windows}/{early_stop_patience})")
                        
                        if consecutive_perfect_windows >= early_stop_patience:
                            print(f"\n🎉 EARLY STOPPING: Achieved {early_stop_success_rate}% success rate for {early_stop_patience} consecutive evaluation windows!")
                            print(f"Training completed at episode {episode + 1}/{episodes}")
                            break
                    else:
                        consecutive_perfect_windows = 0
                    
                    if not tracker:
                        print("-" * 50)
            
            # Update plots periodically
            if tracker and tracker.should_update(episode + 1):
                tracker.update_plots()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        # Clean up tracker
        if tracker:
            tracker.update_plots()  # Final update
            saved_path = tracker.save_plot()  # Auto-save with timestamp
            print("Live tracking plots updated. Close the plot window when done viewing.")
            print(f"Training metrics automatically saved to: {saved_path}")
    
    agent.save("ppo_maze_model.pth")
    print("Training completed! Model saved as 'ppo_maze_model.pth'")
    
    return agent, episode_rewards, episode_lengths


def test_trained_agent(config: PPOConfig, model_path: str = "ppo_maze_model.pth", episodes: int = 2, max_steps: int = 250, reward_config: dict = None) -> None:
    """Test the trained PPO agent."""
    from env import Environment, MultiTargetEnvironment
    
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = MultiTargetEnvironment(warehouse_map)
    
    # Load trained agent
    agent = PPO(config)
    agent.load(model_path)
    
    # Initialize reward function
    reward_fn = MazeRewardFunction(reward_config)
    
    print(f"Testing PPO agent: Start {env.maze.agent_pos}, Goal {env.maze.goal_pos}")
    print("-" * 50)
    
    success_count = total_steps = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        position_history = deque(maxlen=config.recent_history_length)
        
        env.render()  
        
        done = False
        while steps < max_steps and not done:
            action, _ = agent.select_action(state, training=False)
            next_state, done = env.step(action)
            
            current_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
            reward = reward_fn(env, done, position_history, current_distance=current_distance)
            position_history.append(env.maze.agent_pos)

            episode_reward += reward
            steps += 1
            
            env.render()
            state = next_state
        
        total_steps += steps
        
        if done:
            success_count += 1
            print(f"Episode {episode + 1}: SUCCESS in {steps} steps! Reward: {episode_reward:.2f}")
        else:
            print(f"Episode {episode + 1}: FAILED after {steps} steps. Reward: {episode_reward:.2f}")
    
    print("-" * 50)
    print(f"Results: {success_count}/{episodes} successful ({success_count/episodes*100:.1f}%)")
    print(f"Average Steps: {total_steps/episodes:.1f}")


if __name__ == "__main__":
    set_random_seeds(69)
    config = PPOConfig(
        hidden_size=128,
        lr_actor=1e-4, 
        lr_critic=1e-4, 
        gamma=0.99, 
        clip_epsilon=0.2, 
        k_epochs=4, 
        entropy_coef=0.01,
        memory_size=2048,
        batch_size=64,
        gae_lambda=0.95,
        state_dim=5,
    )
    agent, rewards, lengths = train_ppo(
        episodes=10_000, 
        max_steps_per_episode=500, 
        update_frequency=5, 
        config=config,
        early_stop_success_rate=100.0, 
        early_stop_patience=10,
        enable_live_tracking=True
    )
    test_trained_agent(config=config, episodes=12)  