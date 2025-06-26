import torch.nn as nn
from collections import deque
import random
import numpy as np
import torch
import torch.optim as optim
from env_dqn import Environment

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)
    

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def train(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.q_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min) 
    
    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())

# def train_dqn(episodes=500, max_steps=250, update_freq=10):
#     maze = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
#     env = Environment(maze)
#     observation_dim = len(env.reset()) # infer from first observation
#     action_dim = 8 # it's 8 directions
#     agent = DQNAgent(observation_dim, action_dim)

#     for ep in range(episodes):
#         state = env.reset()
#         total_reward = 0
#         for step in range(max_steps):
#             action = agent.select_action(state)
#             next_state, reward, done, _ = env.step(action)
#             agent.replay.store((state, action, reward, next_state, float(done)))
#             agent.train()
#             state = next_state
#             total_reward += reward
#             if done:
#                 break

#         if ep % update_freq == 0:
#             agent.update_target()
#         agent.decay_epsilon()
#         print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.2f}")   
#     return agent

def train_dqn(config: dict):
    # Load environment
    maze = np.load(config['env']['map_path']).astype(np.int8)
    env = Environment(maze, step_size=config['env'].get('step_size', 0.4), reward_config=config['reward'])
    
    # Get observation/action dimensions from environment
    obs_dim = len(env.reset())
    action_dim = config['DQN'].get('action_dim', 8)

    # Initialize agent with config values
    agent = DQNAgent(
        state_dim=obs_dim,
        action_dim=action_dim,
        lr=config['DQN']['learning_rate'],
        gamma=config['DQN']['gamma'],
        epsilon=config['DQN']['epsilon_start'],
        epsilon_decay=np.exp(np.log(config['DQN']['epsilon_end'] / config['DQN']['epsilon_start']) / config['DQN']['epsilon_decay']),
        epsilon_min=config['DQN']['epsilon_end']
    )

    # Training loop
    episodes = config['trainer']['episodes']
    max_steps = config['trainer']['max_steps']
    update_freq = config['DQN']['target_update_frequency']

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.replay.store((state, action, reward, next_state, float(done)))
            agent.train(batch_size=config['DQN']['batch_size'])

            state = next_state
            total_reward += reward

            if done:
                break

        if ep % update_freq == 0:
            agent.update_target()
        agent.decay_epsilon()

        print(f"Episode {ep}, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.2f}")

    return agent, env

def test_dqn_agent(agent, env, episodes=3, max_steps=250, return_metrics=False):
    success_count = total_steps = 0

    print(f"Testing DQN agent: Start {env.maze.agent_pos}, Goal {env.maze.goal_pos}")
    print("-" * 50)

    for episode in range(episodes):
        state = env.reset()
        steps = 0
        episode_reward = 0.0
        position_history = deque(maxlen=4)
        
        env.render()

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            steps += 1
            state = next_state

            env.render()

            if done:
                success_count += 1
                print(f"Episode {episode + 1}: SUCCESS in {steps} steps! Reward: {episode_reward:.2f}")
                break
        else:
            print(f"Episode {episode + 1}: FAILED after {steps} steps. Reward: {episode_reward:.2f}")

        total_steps += steps
    
    avg_steps = total_steps / episodes
    success_rate = success_count / episodes

    print("-" * 50)
    print(f"Results: {success_count}/{episodes} successful ({success_count / episodes * 100:.1f}%)")
    print(f"Average Steps: {total_steps / episodes:.1f}")

    if return_metrics:
        return success_rate, avg_steps