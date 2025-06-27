import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def simple_grouped_mean(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if 'status' in df.columns:
        if 'success' in df['status'].unique():
            df = df[df['status'] == 'success']
        elif 'ok' in df['status'].unique():
            df = df[df['status'] == 'ok']
    cols = ['distance_penalty_coef', 'loop_penalty', 'final_reward']
    df = df.dropna(subset=cols)
    grouped = df.groupby(['distance_penalty_coef', 'loop_penalty'])['final_reward'].mean().reset_index()
    grouped.rename(columns={'final_reward': 'final_reward_mean'}, inplace=True)
    grouped.to_csv(output_csv, index=False)
    print(f"Saved grouped means to {output_csv}")

def plot_3d_scatter(datafile, title, save_path):
    df = pd.read_csv(datafile)

    x = df['distance_penalty_coef'].values
    y = df['loop_penalty'].values
    z = df['final_reward_mean'].values

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=50)
    ax.set_xlabel('Distance Penalty Coefficient')
    ax.set_ylabel('Loop Penalty')
    ax.set_zlabel('Mean Final Reward')
    ax.set_title(title)

    fig.colorbar(sc, shrink=0.5, aspect=5, label='Mean Final Reward')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")
def main():
    simple_grouped_mean('hpo_ppo_reward.csv', 'ppo_grouped_means.csv')
    simple_grouped_mean('hpo_dqn_reward.csv', 'dqn_grouped_means.csv')

    plot_3d_scatter('ppo_grouped_means.csv', 'PPO Final Reward Scatter', 'results/ppo_final_reward_scatter.png')
    plot_3d_scatter('dqn_grouped_means.csv', 'DQN Final Reward Scatter', 'results/dqn_final_reward_scatter.png')


if __name__ == '__main__':
    main()
