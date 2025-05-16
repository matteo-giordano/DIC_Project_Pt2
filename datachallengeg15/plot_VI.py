import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.interpolate import griddata
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

"""
Value‑Iteration visualiser – robust colour handling (no SettingWithCopy warnings)
-------------------------------------------------------------------------------
* Keeps original plots3.py layout for easy Q‑learning comparison
* Uses `smart_scatter` that operates on a **local copy** of the DataFrame slice,
  eliminating pandas SettingWithCopyWarning.
* Guards against empty slices / small‑sample cubic‑interp crashes
* Closes every figure to prevent memory leakage
"""

# -----------------------------
# Utility – smart scatter
# -----------------------------

def smart_scatter(ax, data, x, y, hue, *, max_cat=6, cmap='viridis'):
    """Plot scatter with automatic discrete/continuous hue handling.

    A local copy of `data` is created so we never modify the caller's
    DataFrame slice, avoiding SettingWithCopyWarning.
    """
    uniques = np.unique(data[hue])

    if len(uniques) <= max_cat:
        data_plot = data.copy()
        data_plot['hue_cat'] = data_plot[hue].astype(str)
        base_pal = sns.color_palette('tab10')  # vivid & well‑spaced
        if len(uniques) <= len(base_pal):
            palette = base_pal[:len(uniques)]
        else:
            # fallback to husl which maximises distance in HSV space
            palette = sns.color_palette('husl', n_colors=len(uniques))
        sns.scatterplot(data=data_plot, x=x, y=y, hue='hue_cat', palette=palette,
                        ax=ax, legend='brief')
    else:
        hmin, hmax = data[hue].min(), data[hue].max()
        norm = LogNorm(vmin=hmin, vmax=hmax) if hmin > 0 and np.log10(hmax/hmin) > 2 else None
        sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=cmap,
                        ax=ax, legend='brief', hue_norm=norm)

# -----------------------------
# 0. Load dataset
# -----------------------------
CSV_PATH = '../results/ValueIterationAgent_A1_grid_TOUGH_reward_fn__16_06-23.csv'
OUTPUT_DIR = 'aimgs_vi'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

df['rewards'] = df['rewards'].apply(eval)
mask = df['rewards'].apply(lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) for i in x))
df = df[mask].copy()

df['mean_reward'] = df['rewards'].apply(np.mean)
df['reward_variance'] = df['rewards'].apply(np.var)

# -----------------------------
# 1. Hyper‑param distributions
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i, p in enumerate(['gamma', 'theta', 'sigma']):
    sns.kdeplot(df[p], ax=axs[i//2, i%2])
    axs[i//2, i%2].set_title(f'Distribution of {p}')
axs[1, 1].axis('off')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/VI-distribution-hyperparams.png'); plt.close()

# -----------------------------
# 2 & 3. Fixed‑σ scatter (auto σ)
# -----------------------------
σ_vals = sorted(df['sigma'].unique())
fix_σ = σ_vals[len(σ_vals)//2]
slice_df = df[np.isclose(df['sigma'], fix_σ, atol=1e-6)]
if not slice_df.empty:
    # mean reward
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    smart_scatter(axs[0], slice_df, 'gamma', 'mean_reward', 'theta')
    smart_scatter(axs[1], slice_df, 'theta', 'mean_reward', 'theta')
    axs[2].axis('off')
    plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/VI-reward_mean_fixed_sigma.png'); plt.close()
    # variance
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    smart_scatter(axs[0], slice_df, 'gamma', 'reward_variance', 'theta')
    smart_scatter(axs[1], slice_df, 'theta', 'reward_variance', 'theta')
    axs[2].axis('off')
    plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/VI-reward_var_fixed_sigma.png'); plt.close()
else:
    print('[warn] chosen σ slice empty; skip fixed‑σ scatter')

# -----------------------------
# 4. Sigma vs mean reward
# -----------------------------
plt.figure(figsize=(8, 6))
ax = plt.gca()
smart_scatter(ax, df, 'sigma', 'mean_reward', 'theta')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/VI-sigma-reward.png'); plt.close()

# -----------------------------
# 5. 2×2 reward surfaces
# -----------------------------
fig = plt.figure(figsize=(16, 12)); canvas = FigureCanvas(fig)
subplot_idx = 0
for σ in σ_vals[:4]:
    sub = df[np.isclose(df['sigma'], σ, atol=1e-6)]
    if sub.shape[0] < 3:
        continue
    subplot_idx += 1
    g, t = np.meshgrid(np.linspace(sub['gamma'].min(), sub['gamma'].max(), 40),
                       np.linspace(sub['theta'].min(), sub['theta'].max(), 40))
    z = griddata((sub['gamma'], sub['theta']), sub['mean_reward'], (g, t), method='cubic')
    ax = fig.add_subplot(2, 2, subplot_idx, projection='3d')
    ax.plot_surface(g, t, z, cmap='viridis', edgecolor='none')
    ax.set_title(f'σ≈{σ}')
if subplot_idx:
    if 'iters' in df.columns:
        norm = Normalize(vmin=0, vmax=df['iters'].max()); sm = ScalarMappable(cmap='plasma', norm=norm); sm.set_array([])
        fig.colorbar(sm, ax=fig.axes, shrink=0.5, label='Iterations')
    canvas.print_figure(f'{OUTPUT_DIR}/VI-reward_surfaces_2x2.png')
plt.close()

# -----------------------------
# 6. Stability boxplots
# -----------------------------
most = df.nlargest(10, 'reward_variance').assign(stability='Most Variable')
least = df.nsmallest(10, 'reward_variance').assign(stability='Most Stable')
box = pd.concat([least, most])
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for ax, p in zip(axs.flatten(), ['gamma', 'theta', 'sigma']):
    sns.boxplot(data=box, x='stability', y=p, ax=ax)
axs[1, 1].axis('off'); plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/VI-stability.png'); plt.close()

