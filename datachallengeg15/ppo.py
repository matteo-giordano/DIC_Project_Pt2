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
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """Configuration for PPO hyperparameters."""
    state_dim: int = 13                 # Dimension of input state (e.g., observation vector)
    action_dim: int = 8                 # Number of discrete actions
    hidden_size: int = 128              # Hidden layer size for both networks
    lr_actor: float = 3e-4              # Learning rate for actor
    lr_critic: float = 1e-3             # Learning rate for critic
    gamma: float = 0.99                 # Discount factor
    clip_epsilon: float = 0.2           # PPO clipping range
    k_epochs: int = 4                   # Number of epochs per update
    entropy_coef: float = 0.01          # Entropy bonus coefficient
    max_grad_norm: float = 0.5          # Gradient clipping threshold
    memory_size: int = 2048             # Experience buffer capacity
    batch_size: int = 64                # Mini-batch size
    gae_lambda: float = 0.95            # Lambda for GAE (bias-variance trade-off)
    recent_history_length: int = 16     # Position history length (for anti-loop reward)


class ActorNetwork(nn.Module):
    """Actor network for policy approximation."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(                             # A feedforward network with 2 hidden layers
            nn.Linear(input_size, hidden_size), nn.ReLU(),        # Layer 1: state → hidden_size (e.g., 13 → 128) + ReLU Non-linear activation
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),  # Layer 2: hidden_size → hidden_size/2 (e.g., 128 → 64) + ReLU activation
            nn.Linear(hidden_size // 2, output_size)              # Output: hidden_size/2 → output_size (e.g., 64 → 8) (unnormalized logits for actions)
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # Xavier for stable gradients
                nn.init.constant_(module.bias, 0.0)    # Zero bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action probabilities π(a|s) from input state.
        Args:
            x (Tensor): input state tensor of shape (batch_size, input_size)
        Returns:
            Tensor: action probabilities of shape (batch_size, output_size)
        """
        return F.softmax(self.network(x), dim=-1) # Normalize output into probability distribution


class CriticNetwork(nn.Module):
    """Critic network for value function approximation."""
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),       # Layer 1: state → hidden_size (e.g., 13 → 128) + ReLU activation
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), # Layer 2: hidden_size → hidden_size/2 (e.g., 128 → 64) + ReLU activation
            nn.Linear(hidden_size // 2, 1)                       # Output: scalar value V(s). hidden_size/2 → 1 (e.g., 64 → 1)   
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight) # Xavier for stable gradients
                nn.init.constant_(module.bias, 0.0)    # Zero bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute estimated value V(s) for each input state.
        Args:
            x (Tensor): input state of shape (batch_size, input_size)
        Returns:
            Tensor: scalar state values of shape (batch_size, 1)
        """
        return self.network(x)


class PPOMemory:
    """Experience replay buffer for PPO with pre-allocated arrays for speed."""
    def __init__(self, max_size: int = 2048, obs_dim: int = 13):
        self.max_size = max_size # Maximum number of transitions to store
        self.ptr = 0             # Write pointer/index
        self.size = 0            # Current number of stored transitions
    
        # Pre-allocate arrays for better performance
        self.states = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.action_logprobs = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.states_next = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)
    
    def store(self, state: np.ndarray, action: int, action_logprob: float, 
              reward: float, state_next: np.ndarray, done: bool) -> None:
        """
        Store a new transition into the buffer.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            action_logprob (float): Log-probability of the action.
            reward (float): Reward received.
            state_next (np.ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        # Insert transition at current pointer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.action_logprobs[self.ptr] = action_logprob
        self.rewards[self.ptr] = reward
        self.states_next[self.ptr] = state_next
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size     # Advance pointer with wraparound
        self.size = min(self.size + 1, self.max_size) # Track actual number of items stored (capped at max_size)
    
    def get_all(self) -> Tuple[np.ndarray, ...]:
        """Return all stored transitions as numpy arrays in the buffer.
        Returns:
            Tuple of (states, actions, logprobs, rewards, next_states, dones).
        """
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
        """
        Select action using current policy (actor network).
        Args:
            state (np.ndarray): Current state vector (1D).
            training (bool): Whether to sample (True) or take argmax (False).
        Returns:
            Tuple[int, float]: Chosen action and its log probability.
        """
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
        """
        Store transition in experience buffer.
        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            action_logprob (float): Log probability of the action.
            reward (float): Received reward.
            state_next (np.ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        self.memory.store(state, action, action_logprob, reward, state_next, done)
    
    def _compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                               values_next: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        Args:
            rewards (Tensor): Rewards from environment.
            values (Tensor): Critic estimates for current states.
            values_next (Tensor): Critic estimates for next states.
            dones (Tensor): Terminal flags.
        Returns:
            Tuple[Tensor, Tensor]: Normalized advantages and returns.
        """
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
