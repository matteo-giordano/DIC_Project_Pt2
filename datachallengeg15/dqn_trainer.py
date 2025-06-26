import copy
import numpy as np
import random
import torch
from tqdm import tqdm, trange
from env_dqn import Environment, MultiTargetEnvironment
from reward_dqn import RewardFunction
from dqn import DQNAgent
from utils import set_random_seeds

class DQNTrainer:
    def __init__(self, config: dict, reward_fn=None):
        # reproducibility
        set_random_seeds(config.get('seed'))
        self.config = copy.deepcopy(config)
        self.reward_fn = reward_fn or RewardFunction(self.config['reward']['args'])

        # hyperparams
        tc = self.config['trainer']
        self.episodes = tc['episodes']
        self.max_steps = tc['max_steps']
        self.enable_live = tc.get('live_tracking', False)

        # environment
        env_cfg = self.config['env']
        if env_cfg.get('name') == 'MultiTargetEnvironment':
            self.env = MultiTargetEnvironment(
                map_path=env_cfg['map_path'],
                step_size=env_cfg.get('step_size', 0.4),
                reward_fn=self.reward_fn,
                max_steps=self.max_steps
            )
        else:
            self.env = Environment(
                map_path=env_cfg['map_path'],
                goal_pos=env_cfg.get('goal_pos'),
                reward_fn=self.reward_fn,
                step_size=env_cfg.get('step_size', 0.4),
                max_steps=self.max_steps
            )

        # agent
        obs_dim = len(self.env.reset())
        action_dim = self.config['agent'].get('action_dim', len(self.env.maze.action_vectors))
        ag = self.config['agent']
        self.agent = DQNAgent(
            state_dim=obs_dim, action_dim=action_dim,
            lr=ag['learning_rate'], gamma=ag['gamma'],
            epsilon=ag['epsilon_start'], epsilon_decay=ag['epsilon_decay_steps'],
            epsilon_min=ag['epsilon_end']
        )
        self.target_update_freq = ag['target_update_freq']
        self.batch_size = ag['batch_size']

    def train_with_history(self):
        """Train and collect per‐episode metrics for HPO."""
        episode_data = []
        for ep in trange(1, self.episodes+1, desc="Train Ep", leave=False):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            success = False

            # track Q‐value estimates
            q_values = []

            for _ in range(self.max_steps):
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                with torch.no_grad():
                    q_pred = self.agent.q_net(state_tensor).cpu().numpy()[0]
                q_values.append(float(np.max(q_pred)))

                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.agent.replay.store((state, action, reward, next_state, float(done)))
                self.agent.train(batch_size=self.batch_size)

                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    success = info.get('goal_reached', False)
                    break

            # soft target update & epsilon decay
            if ep % self.target_update_freq == 0:
                self.agent.update_target()
            self.agent.decay_epsilon()

            episode_data.append({
                'total_reward': total_reward,
                'steps': steps,
                'success': success,
                'avg_value': float(np.mean(q_values)) if q_values else 0.0,
                'value_estimates': q_values.copy()
            })

        # Expose for HPO runner
        self.episode_data = episode_data
        return episode_data

    def test(self, episodes: int = 3, max_steps: int = None):
        """Evaluate the trained agent (greedy policy)."""
        max_s = max_steps or self.max_steps
        success_count = 0
        total_steps = 0
        for ep in range(1, episodes+1):
            state = self.env.reset()
            steps = 0
            for _ in range(max_s):
                action = self.agent.select_action(state, test_mode=True)
                state, _, done, info = self.env.step(action)
                steps += 1
                if done:
                    success_count += int(info.get('goal_reached', False))
                    break
            total_steps += steps
        return success_count/episodes, total_steps/episodes
