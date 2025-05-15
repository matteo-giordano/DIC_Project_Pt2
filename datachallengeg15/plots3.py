import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import plasma
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load new dataset
df = pd.read_csv('../results/TabularQLearningAgent_A1_grid_TOUGH_reward_fn__15_16-14.csv')
df['rewards'] = df['rewards'].apply(eval)
df = df[df['rewards'].apply(lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) for i in x))]
df['mean_reward'] = df['rewards'].apply(np.mean)
df['reward_variance'] = df['rewards'].apply(np.var)

# -----------------------------
# QL-distribution-hyperparams-failed-successfull-runs.png
# -----------------------------
op_len = df["optimal_path_length"].mode()[0]
invalid_runs = df[(df['valid_path'] == False) | (df['optimal_path_length'] != op_len)]
valid_runs = df[(df['valid_path'] == True) & (df['optimal_path_length'] == op_len)]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
hyperparams = ['alpha', 'gamma', 'epsilon', 'sigma']
for i, param in enumerate(hyperparams):
    sns.kdeplot(df[param], label='All Runs', ax=axs[i//2, i%2])
    sns.kdeplot(invalid_runs[param], label='Invalid/Non-optimal', ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(f'Distribution of {param}')
    axs[i//2, i%2].legend()
plt.tight_layout()
plt.savefig('aimgs/QL-distribution-hyperparams-failed-successfull-runs.png')

# -----------------------------
# QL-reward_mean_fixed_sigma_hyperparams.png
# -----------------------------
fixed_sigma_df = df[np.isclose(df['sigma'], 0.267, atol=1e-3)]
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=fixed_sigma_df, x='alpha', y='mean_reward', ax=axs[0])
sns.scatterplot(data=fixed_sigma_df, x='gamma', y='mean_reward', ax=axs[1])
sns.scatterplot(data=fixed_sigma_df, x='epsilon', y='mean_reward', ax=axs[2])
for i, param in enumerate(['alpha', 'gamma', 'epsilon']):
    axs[i].set_title(f'Mean Reward vs {param} (Sigma \u2248 0.267)')
plt.tight_layout()
plt.savefig('aimgs/QL-reward_mean_fixed_sigma_hyperparams.png')

# -----------------------------
# QL-reward_surfaces_2x2_colored_by_iters.png
# -----------------------------
selected_sigmas = sorted(df['sigma'].round(3).unique())[:4]
fig = plt.figure(figsize=(16, 12))
canvas = FigureCanvas(fig)
for i, sigma_val in enumerate(selected_sigmas):
    sub_df = df[np.isclose(df['sigma'], sigma_val, atol=1e-3)]
    alpha_vals = np.linspace(sub_df['alpha'].min(), sub_df['alpha'].max(), 50)
    epsilon_vals = np.linspace(sub_df['epsilon'].min(), sub_df['epsilon'].max(), 50)
    alpha_grid, epsilon_grid = np.meshgrid(alpha_vals, epsilon_vals)
    reward_grid = griddata((sub_df['alpha'], sub_df['epsilon']), sub_df['mean_reward'], (alpha_grid, epsilon_grid), method='cubic')
    iters_grid = griddata((sub_df['alpha'], sub_df['epsilon']), sub_df['iters'], (alpha_grid, epsilon_grid), method='cubic')
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.plot_surface(alpha_grid, epsilon_grid, reward_grid, facecolors=plasma(iters_grid / np.nanmax(iters_grid)), rstride=1, cstride=1, linewidth=0)
    ax.set_title(f'Mean Reward (Sigma \u2248 {sigma_val})\nColored by Iterations')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Mean Reward')
norm = Normalize(vmin=0, vmax=df['iters'].max())
sm = ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sm, cax=cbar_ax, label='Iterations to Converge')
canvas.print_figure('aimgs/QL-reward_surfaces_2x2_colored_by_iters.png')

# -----------------------------
# QL-reward_variance_fixed_sigma_hyperparams.png
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=fixed_sigma_df, x='alpha', y='reward_variance', ax=axs[0])
sns.scatterplot(data=fixed_sigma_df, x='gamma', y='reward_variance', ax=axs[1])
sns.scatterplot(data=fixed_sigma_df, x='epsilon', y='reward_variance', ax=axs[2])
for i, param in enumerate(['alpha', 'gamma', 'epsilon']):
    axs[i].set_title(f'Reward Variance vs {param} (Sigma \u2248 0.267)')
plt.tight_layout()
plt.savefig('aimgs/QL-reward_variance_fixed_sigma_hyperparams.png')

# -----------------------------
# QL-sigma-reward.png
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sigma', y='mean_reward', hue='epsilon')
plt.title('Mean Reward vs Sigma (colored by Epsilon)')
plt.xlabel('Sigma')
plt.ylabel('Mean Reward')
plt.tight_layout()
plt.savefig('aimgs/QL-sigma-reward.png')

# -----------------------------
# QL-stable-unstable-params.png
# -----------------------------
valid = df[(df['valid_path'] == True) & (df['optimal_path_length'] == 23)]
grouped = valid.groupby(['alpha', 'gamma', 'epsilon', 'sigma'])
summary = grouped['mean_reward'].agg(['mean', 'std', 'count']).reset_index()
summary.rename(columns={'mean': 'reward_mean', 'std': 'reward_std', 'count': 'num_seeds'}, inplace=True)
most_variable = summary.sort_values('reward_std', ascending=False).head(10).copy()
least_variable = summary.sort_values('reward_std').head(10).copy()
most_variable['stability'] = 'Most Variable'
least_variable['stability'] = 'Most Stable'
top10_configs = pd.concat([most_variable, least_variable], ignore_index=True)

# Ensure that stability is treated as a category for consistent boxplot rendering
top10_configs['stability'] = pd.Categorical(top10_configs['stability'], categories=['Most Stable', 'Most Variable'])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for ax, param in zip(axs.flatten(), ['alpha', 'gamma', 'epsilon', 'sigma']):
    sns.boxplot(data=top10_configs, x='stability', y=param, ax=ax)
    ax.set_title(f'{param.capitalize()} Distribution in Top 10 Stable vs Unstable Configs')
plt.tight_layout()
plt.savefig('aimgs/QL-stable-unstable-params.png')

# -----------------------------
# reward_surfaces_by_sigma.png
# -----------------------------
fig = plt.figure(figsize=(16, 4 * len(selected_sigmas)))
canvas = FigureCanvas(fig)
for i, sigma_val in enumerate(selected_sigmas):
    sub_df = df[np.isclose(df['sigma'], sigma_val, atol=1e-3)]
    alpha_vals = np.linspace(sub_df['alpha'].min(), sub_df['alpha'].max(), 50)
    epsilon_vals = np.linspace(sub_df['epsilon'].min(), sub_df['epsilon'].max(), 50)
    alpha_grid, epsilon_grid = np.meshgrid(alpha_vals, epsilon_vals)
    reward_grid = griddata((sub_df['alpha'], sub_df['epsilon']), sub_df['mean_reward'], (alpha_grid, epsilon_grid), method='cubic')
    ax = fig.add_subplot(len(selected_sigmas), 1, i + 1, projection='3d')
    ax.plot_surface(alpha_grid, epsilon_grid, reward_grid, cmap='viridis', edgecolor='none')
    ax.set_title(f'Mean Reward Surface (Sigma \u2248 {sigma_val})')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Mean Reward')
canvas.print_figure('aimgs/reward_surfaces_by_sigma.png')
