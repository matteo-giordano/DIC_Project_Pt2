import numpy as np
from ppo import PPOConfig, PPO
from viz import LiveTracker
from env import Environment, MultiTargetEnvironment
from collections import deque
from tqdm import tqdm
from reward import *


class PPOTrainer:
    def __init__(self, cfg: dict):
        self.episodes = cfg['trainer']['episodes'] # Total number of training episodes
        self.max_steps = cfg['trainer']['max_steps'] # Maximum steps per episode
        self.update_frequency = cfg['trainer']['update_frequency'] # How often to update the agent (in episodes)
        self.early_stop_success_rate = cfg['trainer']['early_stop_success_rate'] # Success rate threshold to trigger early stopping
        self.early_stop_patience = cfg['trainer']['early_stop_patience'] # Number of consecutive windows to meet early stop criterion
        self.enable_live_tracking = cfg['trainer']['enable_live_tracking'] # Enable real-time plot tracking

        self.ppo_config = PPOConfig(**cfg['PPO']) # Initialize PPO hyperparameters
        self.env = self._init_env(cfg['env']) # Load and configure environment
        self.agent = PPO(self.ppo_config) 
        self.reward = eval(cfg['reward']['name'])(cfg['reward']) # Initialize reward function
        self.model_path = cfg['model_path'] # Path to save the trained PPO model

    def _init_env(self, env_config: dict) -> Environment:
        env = eval(env_config['name'])
        array = np.load(env_config['map_path']).astype(np.int8)
        env = env(array, step_size=env_config['step_size'])
        if env_config['name'] == 'MultiTargetEnvironment': # Optional: override goal positions if provided
            if env_config.get('goals') is not None:
                env.goals = np.array(env_config['goals'])
        
        if env_config['name'] == 'Environment': # Optional: set custom starting position
            if env_config.get('start_pos') is not None:
                env.maze.agent_pos = np.array(env_config['start_pos'])
            if env_config.get('goal_pos') is not None:
                env.maze.goal_pos = np.array(env_config['goal_pos'])
        return env

    def train(self):
        """
        Run the PPO training loop.
        Handles environment interaction, reward collection, PPO updates, early stopping,
        and optional live metric tracking.
        Returns:
            Tuple: Trained agent, list of episode rewards, and list of episode lengths.
        """
        if self.enable_live_tracking: # Set up live metric visualization
            print("Live tracking enabled")
            self.tracker = LiveTracker(update_interval=self.update_frequency, window_size=50)

        episode_rewards, episode_lengths = [], []
        success_count = 0
        consecutive_perfect_windows = 0

        print(f"Starting PPO training on {self.env.__class__.__name__} environment...")
        print(f"Episodes: {self.episodes}, Max steps: {self.max_steps}")
        print(f"Update frequency: {self.update_frequency}")
        print(f"Early stopping: {self.early_stop_success_rate}% success rate for {self.early_stop_patience} consecutive windows")
        print("-" * 60)

        try:
            for episode in tqdm(range(self.episodes), desc="Training"):
                state = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                position_history = deque(maxlen=self.ppo_config.recent_history_length)  # Track recent agent positions for loop detection

                for step in range(self.max_steps):
                    action, action_logprob = self.agent.select_action(state, training=True) # Select action using current policy
                    next_state, done = self.env.step(action)
                    reward = self.reward(self.env, done, position_history) # Compute shaped reward
                    position_history.append(self.env.maze.agent_pos)

                    self.agent.store_transition(state, action, action_logprob, reward, next_state, done) # Save experience to buffer

                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                    if done:
                        success_count += 1
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                if self.enable_live_tracking:
                    self.tracker.add_episode_data(episode + 1, episode_reward, episode_length)

                if (episode + 1) % self.update_frequency == 0 and len(self.agent.memory) > 0: # Perform PPO update after specified number of episodes
                    actor_loss, critic_loss = self.agent.update()
                    if self.enable_live_tracking:
                        self.tracker.add_loss_data(actor_loss, critic_loss)

                    if (episode + 1) % (self.update_frequency * 4) == 0: # Periodic evaluation block
                        recent_rewards = episode_rewards[-self.update_frequency*4:]
                        recent_lengths = episode_lengths[-self.update_frequency*4:]
                        avg_reward = np.mean(recent_rewards)
                        avg_length = np.mean(recent_lengths)
                        success_rate = success_count / (episode + 1) * 100
                        recent_success_rate = sum(1 for r in recent_rewards if r > 50) / len(recent_rewards) * 100

                        if not self.enable_live_tracking:
                            print(f"Episode {episode + 1}/{self.episodes}")
                            print(f"  Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
                            print(f"  Success Rate: {success_rate:.1f}% (Recent: {recent_success_rate:.1f}%)")
                            print(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

                        if recent_success_rate >= self.early_stop_success_rate: # Check for early stopping condition
                            consecutive_perfect_windows += 1
                            print(f"  Perfect success rate achieved! ({consecutive_perfect_windows}/{self.early_stop_patience})")

                            if consecutive_perfect_windows >= self.early_stop_patience:
                                print(f"\n🎉 EARLY STOPPING: Achieved {self.early_stop_success_rate}% success rate for {self.early_stop_patience} consecutive evaluation windows!")
                                print(f"Training completed at episode {episode + 1}/{self.episodes}")
                                break
                        else:
                            consecutive_perfect_windows = 0

                        if not self.enable_live_tracking:
                            print("-" * 50)
                        
                if self.enable_live_tracking and self.tracker.should_update(episode + 1):
                    self.tracker.update_plots()

        except KeyboardInterrupt:
            print("\nTraining stopped. Saving model...")
            self.agent.save(self.model_path)

        finally:
            if self.enable_live_tracking:
                self.tracker.update_plots()
                saved_path = self.tracker.save_plot()
                print("Live tracking plots updated. Close the plot window when done viewing.")
                print(f"Training metrics automatically saved to: {saved_path}")

        self.agent.save(self.model_path)
        print(f"Training completed! Model saved as {self.model_path}")

        return self.agent, episode_rewards, episode_lengths

    def test(self, episodes: int = 2, max_steps: int = 250):
        """
        Evaluate the trained PPO agent over several episodes.
        Args:
            episodes (int): Number of test episodes to run.
            max_steps (int): Maximum steps allowed per episode.
        Returns:
            float: Success rate (successful episodes / total episodes).
        """
        self.agent.load(self.model_path) # Load saved model checkpoint

        print(f"Testing trained PPO agent: Start {self.env.maze.agent_pos}, Goal {self.env.maze.goal_pos}")
        print("-" * 50)

        success_count = total_steps = 0

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0.0
            steps = 0
            position_history = deque(maxlen=self.ppo_config.recent_history_length)

            self.env.render() # Visualize agent and goal in environment
            
            done = False
            while steps < max_steps and not done: # Run agent until success or max steps
                action, _ = self.agent.select_action(state, training=False)
                next_state, done = self.env.step(action)

                reward = self.reward(self.env, done, position_history)
                position_history.append(self.env.maze.agent_pos)

                episode_reward += reward
                steps += 1

                self.env.render()
                state = next_state
                
            total_steps += steps
            
            if done:
                success_count += 1
                print(f"Episode {episode + 1}: SUCCESS in {steps} steps! Reward: {episode_reward:.2f}") # Log success and reward per episode
            else:
                print(f"Episode {episode + 1}: FAILED after {steps} steps. Reward: {episode_reward:.2f}") # Log failure and reward per episode

        print("-" * 50)
        print(f"Results: {success_count}/{episodes} successful ({success_count/episodes*100:.1f}%)")
        print(f"Average Steps: {total_steps/episodes:.1f}")

        return success_count / episodes
    