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


@dataclass
class PPOConfig:
    """Configuration for PPO hyperparameters."""
    state_dim: int = 11
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


def calculate_reward(env, done: bool, position_history: deque, current_distance: float) -> float:
    """Calculate reward with improved shaping and anti-loop mechanism."""
    map_diagonal = 46 # TODO: hardcoded for efficiency
    current_pos = tuple(env.maze.agent_pos)
    position_history.append(current_pos)
    
    if done:
        return 120.0
    # Distance normalization and penalties
    normalized_distance = current_distance / map_diagonal
    reward = -normalized_distance * 0.1 - 0.01  
    
    # Simplified anti-loop penalty
    if len(position_history) >= 3 and current_pos in list(position_history)[-3:-1]:
        reward -= 0.5
    return reward


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
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate arrays for better performance
        self.states = np.zeros((max_size, 11), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.action_logprobs = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.states_next = np.zeros((max_size, 11), dtype=np.float32)
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
        
        self.memory = PPOMemory(self.config.memory_size)
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


def train_ppo_on_maze(episodes: int = 2000, max_steps_per_episode: int = 1000, 
                     update_frequency: int = 10, config: Optional[PPOConfig] = None,
                     early_stop_success_rate: float = 100.0, early_stop_patience: int = 3) -> Tuple[PPO, List[float], List[int]]:
    """Train PPO agent on maze environment with early stopping."""
    from env import Environment
    
    # Load the warehouse map
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Environment(warehouse_map)
    
    # Initialize PPO agent and tracking variables
    agent = PPO(config)
    episode_rewards, episode_lengths = [], []
    success_count = 0
    consecutive_perfect_windows = 0
        
    print("Starting PPO training on maze environment...")
    print(f"Episodes: {episodes}, Max steps: {max_steps_per_episode}")
    print(f"Update frequency: {update_frequency}")
    print(f"Early stopping: {early_stop_success_rate}% success rate for {early_stop_patience} consecutive windows")
    print("-" * 60)
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        position_history = deque(maxlen=4)
        
        for step in range(max_steps_per_episode):
            # Select action
            action, action_logprob = agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, done = env.step(action)
            
            # Calculate reward
            current_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
            reward = calculate_reward(env, done, position_history, current_distance)
            
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
        
        # Update policy
        if (episode + 1) % update_frequency == 0 and len(agent.memory) > 0:
            actor_loss, critic_loss = agent.update()
            
            # Print progress less frequently for speed
            if (episode + 1) % (update_frequency * 4) == 0:
                recent_rewards = episode_rewards[-update_frequency*4:]
                recent_lengths = episode_lengths[-update_frequency*4:]
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                success_rate = success_count / (episode + 1) * 100
                recent_success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards) * 100
                
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
                
                print("-" * 50)
    
    agent.save("ppo_maze_model.pth")
    print("Training completed! Model saved as 'ppo_maze_model.pth'")
    
    return agent, episode_rewards, episode_lengths


def test_trained_agent(model_path: str = "ppo_maze_model.pth", episodes: int = 2, max_steps: int = 250) -> None:
    """Test the trained PPO agent."""
    from env import Environment
    
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Environment(warehouse_map)
    
    # Load trained agent
    agent = PPO()
    agent.load(model_path)
    
    print(f"Testing PPO agent: Start {env.maze.agent_pos}, Goal {env.maze.goal_pos}")
    print("-" * 50)
    
    success_count = total_steps = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        steps = 0
        position_history = deque(maxlen=4)
        
        env.render()  
        
        done = False
        while steps < max_steps and not done:
            action, _ = agent.select_action(state, training=False)
            next_state, done = env.step(action)
            
            current_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
            reward = calculate_reward(env, done, position_history, current_distance)
            
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
        lr_actor=3e-4, 
        lr_critic=1e-3, 
        gamma=0.99, 
        clip_epsilon=0.2, 
        k_epochs=4, 
        entropy_coef=0.01,
        memory_size=2048,
        batch_size=64,
        gae_lambda=0.95
    )
    agent, rewards, lengths = train_ppo_on_maze(
        episodes=500, 
        max_steps_per_episode=500, 
        update_frequency=5, 
        config=config,
        early_stop_success_rate=100.0, 
        early_stop_patience=2
    )
    test_trained_agent(episodes=6)  