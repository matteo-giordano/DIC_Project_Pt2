import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_mean_reward_vs_reward_config(csv_path: str, save_path: str = "images/PPO_mean_reward_vs_reward_config.png"):
    """
    Plot the mean reward for different reward function configurations for PPO and DQN agents.
    This figure helps visualize the impact of reward design (an ablation study) on algorithm performance.
    
    Args:
        csv_path (str): Path to the CSV file containing experimental results.
                        Must include columns: 'reward_config', 'mean_reward', 'agent'.
        save_path (str): Path to save the generated figure. Default is 'images/PPO_mean_reward_vs_reward_config.png'.
    
    Behavior:
        - Creates 'images/' directory if it does not exist.
        - Saves a seaborn boxplot comparing PPO vs DQN under different reward configurations.
    """
    # Create the directory to store figures if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load the data
    df = pd.read_csv(csv_path)

    # Check required columns
    required_columns = {'reward_config', 'mean_reward', 'agent'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='reward_config', y='mean_reward', hue='agent')
    plt.title('Mean Reward under Different Reward Configurations (PPO vs DQN)', fontsize=14)
    plt.xlabel('Reward Function Configuration', fontsize=12)
    plt.ylabel('Mean Total Reward', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Agent')
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

def plot_mean_reward_vs_target(csv_path: str, save_path: str = "images/PPO_mean_reward_vs_target.png"):
    """
    Plot the mean reward achieved by PPO and DQN agents for different target positions.
    This figure visualizes the performance sensitivity to the target location and helps assess generalization.
    
    Args:
        csv_path (str): Path to the CSV file containing experimental results.
                        Must include columns: 'target_x', 'target_y', 'mean_reward', 'agent'.
        save_path (str): Path to save the generated figure. Default is 'images/PPO_mean_reward_vs_target.png'.
    
    Behavior:
        - Groups results by unique target location.
        - Plots bar plot of mean reward per target for both agents.
        - Saves the plot to the specified path.
    """
    # Create the directory to store figures if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = {'target_x', 'target_y', 'mean_reward', 'agent'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain the following columns: {required_cols}")

    # Create a target identifier column for plotting
    df['target_id'] = df.apply(lambda row: f"({int(row['target_x'])},{int(row['target_y'])})", axis=1)

    # Aggregate mean reward per target and agent
    grouped = df.groupby(['target_id', 'agent'])['mean_reward'].mean().reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=grouped, x='target_id', y='mean_reward', hue='agent')
    plt.title('Mean Reward per Target Location (PPO vs DQN)', fontsize=14)
    plt.xlabel('Target Position (x, y)', fontsize=12)
    plt.ylabel('Mean Total Reward', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Agent')
    plt.tight_layout()

    # Save
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
