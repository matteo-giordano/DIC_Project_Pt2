import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Optional

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