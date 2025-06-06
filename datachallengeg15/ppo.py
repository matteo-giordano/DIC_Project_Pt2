import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from tqdm import tqdm


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:
    def __init__(self, state_dim=11, action_dim=8, lr_actor=1e-3, lr_critic=3e-3, 
                 gamma=0.99, clip_epsilon=0.2, k_epochs=4, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, 128, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, 128).to(self.device)
        self.actor_old = ActorNetwork(state_dim, 128, action_dim).to(self.device)
        
        # Copy parameters to old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Experience buffer
        self.memory = PPOMemory()
        
    def select_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor_old(state)
            
        if training:
            # Sample action from probability distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            
            return action.item(), action_logprob.item()
        else:
            # Take greedy action for evaluation
            return torch.argmax(action_probs).item(), 0.0
    
    def store_transition(self, state, action, action_logprob, reward, state_next, done):
        self.memory.store(state, action, action_logprob, reward, state_next, done)
    
    def update(self):
        # Get data from memory
        states, actions, action_logprobs, rewards, states_next, dones = self.memory.get_all()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_action_logprobs = torch.FloatTensor(action_logprobs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        states_next = torch.FloatTensor(states_next).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Calculate advantages and returns
        with torch.no_grad():
            values = self.critic(states).squeeze()
            values_next = self.critic(states_next).squeeze()
            
            # Calculate TD targets
            td_targets = rewards + self.gamma * values_next * (~dones)
            advantages = td_targets - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for k epochs
        for _ in range(self.k_epochs):
            # Get current policy predictions
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_action_logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Calculate ratio
            ratio = torch.exp(new_action_logprobs - old_action_logprobs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            # Actor loss
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            
            # Critic loss
            values_pred = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values_pred, td_targets)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
        
        # Copy new weights to old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Clear memory
        self.memory.clear()
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_logprobs = []
        self.rewards = []
        self.states_next = []
        self.dones = []
    
    def store(self, state, action, action_logprob, reward, state_next, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_logprobs.append(action_logprob)
        self.rewards.append(reward)
        self.states_next.append(state_next)
        self.dones.append(done)
    
    def get_all(self):
        return (self.states, self.actions, self.action_logprobs, 
                self.rewards, self.states_next, self.dones)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.action_logprobs.clear()
        self.rewards.clear()
        self.states_next.clear()
        self.dones.clear()


def train_ppo_on_maze(episodes=2000, max_steps_per_episode=1000, update_frequency=10, seed=42):
    """
    Train PPO agent on the maze environment
    
    Args:
        episodes (int): Number of training episodes
        max_steps_per_episode (int): Maximum steps per episode
        update_frequency (int): How often to update the policy
        seed (int): Random seed for reproducible results
    """
    # Set random seeds for reproducible results
    set_random_seeds(seed)
    
    from env import Environment
    
    # Load the warehouse map
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Environment(warehouse_map)
    
    # Initialize PPO agent
    agent = PPO()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print("Starting PPO training on maze environment...")
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        initial_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
        best_distance = initial_distance
        position_history = deque(maxlen=10)  # Track recent positions to detect loops
        episode_success = False
        
        for step in range(max_steps_per_episode):
            # Select action
            action, action_logprob = agent.select_action(state, training=True)
            
            # Take step in environment
            next_state, done = env.step(action)
            
            # Calculate reward with anti-loop mechanism
            current_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
            reward = calculate_reward(env, done, position_history, current_distance)
            
            # Update best distance achieved
            if current_distance < best_distance:
                best_distance = current_distance
            
            # Store transition
            agent.store_transition(state, action, action_logprob, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                episode_success = True
                success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Update policy
        if (episode + 1) % update_frequency == 0:
            actor_loss, critic_loss = agent.update()
            
            # Print progress
            avg_reward = np.mean(episode_rewards[-update_frequency:])
            avg_length = np.mean(episode_lengths[-update_frequency:])
            success_rate = success_count / (episode + 1) * 100
            recent_success_rate = sum(1 for i in range(max(0, episode + 1 - update_frequency), episode + 1) 
                                    if i < len(episode_rewards) and episode_rewards[i] > 50) / update_frequency * 100
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Success Rate: {success_rate:.1f}% (Recent: {recent_success_rate:.1f}%)")
            print(f"  Actor Loss: {actor_loss:.4f}")
            print(f"  Critic Loss: {critic_loss:.4f}")
            print("-" * 50)
    
    # Save trained model
    agent.save("ppo_maze_model.pth")
    print("Training completed! Model saved as 'ppo_maze_model.pth'")
    
    return agent, episode_rewards, episode_lengths


def calculate_reward(env, done, position_history, current_distance):
    """
    Calculate reward based on environment state with better reward shaping
    """
    reward = 0.0
    current_pos = tuple(env.maze.agent_pos)
    
    # Add current position to history
    position_history.append(current_pos)
    
    if done:
        # Large positive reward for reaching the goal
        reward += 100.0
    else:
        # Distance-based reward shaping
        # Normalize distance by map diagonal for consistent scaling
        map_diagonal = np.linalg.norm(np.array(env.maze.array.shape))
        normalized_distance = current_distance / map_diagonal
        
        # Penalty for revisiting recent positions (anti-loop mechanism)
        if len(position_history) >= 3:
            recent_positions = list(position_history)[-3:]  # Last 3 positions
            if current_pos in recent_positions[:-1]:  # If current position was visited in last 2 steps
                reward -= 0.5  # Loop penalty
        
        # Small distance-based penalty (encourages staying close to goal)
        map_diagonal = np.linalg.norm(np.array(env.maze.array.shape))
        normalized_distance = current_distance / map_diagonal
        reward -= normalized_distance * 0.05
        
        # Small time penalty to encourage efficiency
        reward -= 0.02
    
    return reward


def test_trained_agent(model_path="ppo_maze_model.pth", episodes=10, seed=42):
    """
    Test the trained PPO agent
    
    Args:
        model_path (str): Path to the trained model
        episodes (int): Number of test episodes
        seed (int): Random seed for reproducible results
    """
    # Set random seeds for reproducible results
    set_random_seeds(seed)
    
    from env import Environment
    
    # Load environment
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Environment(warehouse_map)
    
    # Load trained agent
    agent = PPO()
    agent.load(model_path)
    
    print("Testing trained PPO agent...")
    print(f"start: {env.maze.agent_pos}, end: {env.maze.goal_pos}")
    for episode in range(episodes):
        state = env.reset()

        episode_reward = 0
        steps = 0
        initial_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
        best_distance = initial_distance
        position_history = deque(maxlen=10)
        
        env.render()  # Show initial state
        
        done = False
        while steps < 250 or not done:
            action, _ = agent.select_action(state, training=False)
            next_state, done = env.step(action)
            
            current_distance = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
            reward = calculate_reward(env, done, position_history, current_distance)
            
            # Update best distance achieved
            if current_distance < best_distance:
                best_distance = current_distance
                
            episode_reward += reward
            steps += 1
            
            env.render()  # Show current state
            
            if done:
                print(f"Episode {episode + 1}: Reached goal in {steps} steps! Reward: {episode_reward:.2f}")
                break
            
            state = next_state
        
        if not done:
            print(f"Episode {episode + 1}: Did not reach goal. Steps: {steps}, Reward: {episode_reward:.2f}")


if __name__ == "__main__":
    # Set a consistent seed for reproducible results
    SEED = 69
    
    # Train the agent
    # agent, rewards, lengths = train_ppo_on_maze(episodes=1000, max_steps_per_episode=2000, update_frequency=10, seed=SEED)
    
    # Test the trained agent
    test_trained_agent(seed=SEED)