import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D projection
from matplotlib.cm import plasma, viridis
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------------------------------------------------------------------------
# Value‑Iteration visualisations
# ---------------------------------------------------------------------------
# This script is intentionally kept as close as possible to plots3.py (the
# Q‑learning visualiser).  Only the *minimal* changes required to substitute
# VI‑specific hyper‑parameters (gamma, theta, sigma) have been applied.
# Added inline English comments to highlight every divergence.
# ---------------------------------------------------------------------------

# -----------------------------
# 0. Load dataset
# -----------------------------
# Replace with your results file if different
CSV_PATH = '../results/ValueIterationAgent_results.csv'
df = pd.read_csv(CSV_PATH)

# Convert stringified list -> list  – identical to plots3.py
# ---------------------------------------------------------
df['rewards'] = df['rewards'].apply(eval)
df = df[df['rewards'].apply(lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) for i in x))]
df['mean_reward'] = df['rewards'].apply(np.mean)
df['reward_variance'] = df['rewards'].apply(np.var)

# -----------------------------
# 1. Distribution of hyper‑params (γ, θ, σ)
# -----------------------------
# Swapped α/ε -> θ; otherwise logic is identical.
hyperparams = ['gamma', 'theta', 'sigma']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
# Successful vs unsuccessful split (same heuristic as plots3)
if {'valid_path', 'optimal_path_length'}.issubset(df.columns):
    op_len = df['optimal_path_length'].mode().iloc[0]
    invalid_runs = df[(df['valid_path'] == False) | (df['optimal_path_length'] != op_len)]
else:
    invalid_runs = pd.DataFrame(columns=df.columns)

for i, param in enumerate(hyperparams):
    sns.kdeplot(df[param], label='All Runs', ax=axs[i//2, i%2])
    if not invalid_runs.empty:
        sns.kdeplot(invalid_runs[param], label='Invalid/Non‑optimal', ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(f'Distribution of {param}')
    axs[i//2, i%2].legend()

# Hide the unused 4th subplot to preserve layout parity
axs[1, 1].axis('off')
plt.tight_layout()
plt.savefig('aimgs_vi/VI-distribution-hyperparams.png')

# -----------------------------
# 2. Mean reward vs hyper‑params at fixed σ
# -----------------------------
fixed_sigma_df = df[np.isclose(df['sigma'], 0.267, atol=1e-3)]
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
# Keep three axes to match plots3; the third is intentionally blank.
sns.scatterplot(data=fixed_sigma_df, x='gamma', y='mean_reward', ax=axs[0])
sns.scatterplot(data=fixed_sigma_df, x='theta', y='mean_reward', ax=axs[1])
axs[2].axis('off')
for i, param in enumerate(['gamma', 'theta']):
    axs[i].set_title(f'Mean Reward vs {param} (Sigma ≈ 0.267)')
plt.tight_layout()
plt.savefig('aimgs_vi/VI-reward_mean_fixed_sigma_hyperparams.png')

# -----------------------------
# 3. Reward variance vs hyper‑params at fixed σ
# -----------------------------
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=fixed_sigma_df, x='gamma', y='reward_variance', ax=axs[0])
sns.scatterplot(data=fixed_sigma_df, x='theta', y='reward_variance', ax=axs[1])
axs[2].axis('off')
for i, param in enumerate(['gamma', 'theta']):
    axs[i].set_title(f'Reward Variance vs {param} (Sigma ≈ 0.267)')
plt.tight_layout()
plt.savefig('aimgs_vi/VI-reward_variance_fixed_sigma_hyperparams.png')

# -----------------------------
# 4. Sigma vs mean reward (colour = θ)
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sigma', y='mean_reward', hue='theta', palette='viridis')
plt.title('Mean Reward vs Sigma (colored by Theta)')
plt.xlabel('Sigma')
plt.ylabel('Mean Reward')
plt.tight_layout()
plt.savefig('aimgs_vi/VI-sigma-reward.png')

# -----------------------------
# 5. Reward surfaces – 2×2 grid; optional colour‑bar for iters
# -----------------------------
selected_sigmas = sorted(df['sigma'].round(3).unique())[:4]
fig = plt.figure(figsize=(16, 12))
canvas = FigureCanvas(fig)
for i, sigma_val in enumerate(selected_sigmas):
    sub_df = df[np.isclose(df['sigma'], sigma_val, atol=1e-3)]
    gamma_vals = np.linspace(sub_df['gamma'].min(), sub_df['gamma'].max(), 50)
    theta_vals = np.linspace(sub_df['theta'].min(), sub_df['theta'].max(), 50)
    gamma_grid, theta_grid = np.meshgrid(gamma_vals, theta_vals)
    reward_grid = griddata((sub_df['gamma'], sub_df['theta']), sub_df['mean_reward'],
                           (gamma_grid, theta_grid), method='cubic')
    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    surf = ax.plot_surface(gamma_grid, theta_grid, reward_grid, cmap='viridis', edgecolor='none')
    ax.set_title(f'Mean Reward Surface (Sigma ≈ {sigma_val})')
    ax.set_xlabel('Gamma'); ax.set_ylabel('Theta'); ax.set_zlabel('Mean Reward')

# Colour‑bar for iteration count (only if column exists, mirroring plots3)
if 'iters' in df.columns:
    norm = Normalize(vmin=0, vmax=df['iters'].max())
    sm = ScalarMappable(cmap='plasma', norm=norm); sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Iterations to Converge')
canvas.print_figure('aimgs_vi/VI-reward_surfaces_2x2_colored_by_iters.png')

# -----------------------------
# 6. Reward surfaces stacked by σ (identical logic)
# -----------------------------
fig = plt.figure(figsize=(16, 4 * len(selected_sigmas)))
canvas = FigureCanvas(fig)
for i, sigma_val in enumerate(selected_sigmas):
    sub_df = df[np.isclose(df['sigma'], sigma_val, atol=1e-3)]
    gamma_vals = np.linspace(sub_df['gamma'].min(), sub_df['gamma'].max(), 50)
    theta_vals = np.linspace(sub_df['theta'].min(), sub_df['theta'].max(), 50)
    gamma_grid, theta_grid = np.meshgrid(gamma_vals, theta_vals)
    reward_grid = griddata((sub_df['gamma'], sub_df['theta']), sub_df['mean_reward'],
                           (gamma_grid, theta_grid), method='cubic')
    ax = fig.add_subplot(len(selected_sigmas), 1, i + 1, projection='3d')
    ax.plot_surface(gamma_grid, theta_grid, reward_grid, cmap='viridis', edgecolor='none')
    ax.set_title(f'Mean Reward Surface (Sigma ≈ {sigma_val})')
    ax.set_xlabel('Gamma'); ax.set_ylabel('Theta'); ax.set_zlabel('Mean Reward')
canvas.print_figure('aimgs_vi/VI-reward_surfaces_by_sigma.png')

# -----------------------------
# 7. Stable vs Unstable parameter distributions
# -----------------------------
# Same technique as plots3 but reduced to γ, θ, σ.
most_variable = df.sort_values('reward_variance', ascending=False).head(10).copy()
least_variable = df.sort_values('reward_variance', ascending=True).head(10).copy()
most_variable['stability'] = 'Most Variable'
least_variable['stability'] = 'Most Stable'
box_df = pd.concat([most_variable, least_variable], ignore_index=True)
box_df['stability'] = pd.Categorical(box_df['stability'], categories=['Most Stable', 'Most Variable'])

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for ax, param in zip(axs.flatten(), ['gamma', 'theta', 'sigma']):
    sns.boxplot(data=box_df, x='stability', y=param, ax=ax)
    ax.set_title(f'{param.capitalize()} Distribution in Top 10 Stable vs Unstable Configs')
# Hide unused subplot to keep grid symmetrical
axs[1, 1].axis('off')
plt.tight_layout()
plt.savefig('aimgs_vi/VI-stable-unstable-params.png')

print(f"\n[OK] VI visualisations saved to aimgs_vi/ using input file: {CSV_PATH}\n")
