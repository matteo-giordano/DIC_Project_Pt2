import numpy as np
from tqdm import tqdm
from viz import LiveTracker
from dqn import DQNAgent

class DQNTrainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.episodes = cfg['trainer']['episodes']
        self.max_steps = cfg['trainer']['max_steps']
        self.enable_live_tracking = cfg['trainer']['enable_live_tracking']
        self.model_path = cfg['model_path']

        # Environment
        self.env = self._init_env(cfg['env'])

        # Agent
        dqn_cfg = cfg['DQN']
        self.agent = DQNAgent(
            state_dim=dqn_cfg['state_dim'],
            action_dim=dqn_cfg['action_dim'],
            lr=dqn_cfg['learning_rate'],
            gamma=dqn_cfg['gamma'],
            epsilon=dqn_cfg['epsilon_start'],
            epsilon_decay=np.exp(np.log(dqn_cfg['epsilon_end'] / dqn_cfg['epsilon_start']) / dqn_cfg['epsilon_decay']),
            epsilon_min=dqn_cfg['epsilon_end']
        )

        self.target_update_frequency = dqn_cfg['target_update_frequency']
        self.batch_size = dqn_cfg['batch_size']

    def _init_env(self, env_config):
        env_class = eval(env_config['name'])
        maze = np.load(env_config['map_path']).astype(np.int8)
        env = env_class(maze, step_size=env_config['step_size'])
        if env_config['name'] == 'MultiTargetEnvironment' and env_config.get('goals'):
            env.goals = np.array(env_config['goals'])
        if env_config['name'] == 'Environment' and env_config.get('start_pos'):
            env.maze.agent_pos = np.array(env_config['start_pos'])
        return env

    def train(self):
        if self.enable_live_tracking:
            self.tracker = LiveTracker(update_interval=10, window_size=50)

        for ep in tqdm(range(self.episodes), desc="DQN Training"):
            state = self.env.reset()
            episode_reward = 0.0

            for step in range(self.max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)  # reward comes from env
                self.agent.replay.store((state, action, reward, next_state, float(done)))
                self.agent.train(batch_size=self.batch_size)

                state = next_state
                episode_reward += reward

                if done:
                    break

            if ep % self.target_update_frequency == 0:
                self.agent.update_target()
            self.agent.decay_epsilon()

            if self.enable_live_tracking:
                self.tracker.add_episode_data(ep + 1, episode_reward, step + 1)
                if self.tracker.should_update(ep + 1):
                    self.tracker.update_plots()

        self.agent.q_net.cpu()
        self.agent.save(self.model_path)
        print(f"\nDQN model saved to: {self.model_path}")

    def test(self, episodes=3, max_steps=250):
        self.agent.load(self.model_path)
        print(f"\nTesting DQN Agent")

        success_count, total_steps = 0, 0

        for ep in range(episodes):
            state = self.env.reset()
            steps = 0
            episode_reward = 0.0

            self.env.render()

            for _ in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1
                self.env.render()

                if done:
                    success_count += 1
                    print(f"Episode {ep + 1}: SUCCESS in {steps} steps! Reward: {episode_reward:.2f}")
                    break
            else:
                print(f"Episode {ep + 1}: FAILED after {steps} steps. Reward: {episode_reward:.2f}")

            total_steps += steps

        print("-" * 50)
        print(f"Results: {success_count}/{episodes} successful ({100 * success_count / episodes:.1f}%)")
        print(f"Average Steps: {total_steps / episodes:.1f}")