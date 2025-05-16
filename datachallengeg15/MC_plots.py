import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import seaborn as sns
import ast
import os
from matplotlib.cm import plasma
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

CSV_PATH = ['../results/MonteCarloAgent_A1_grid_TOUGH_reward_fn__15_23-22.csv', '../results/MonteCarloAgent_A1_grid_reward_fn__16_01-40-all-parameters-3.csv']
IMG_DIR = '../imgs/'

def load_data(path):
    df = pd.read_csv(path)

    # Compute mean reward only for rows where rewards are valid
    def safe_mean_reward(r):
        try:
            rewards = ast.literal_eval(r)
            if isinstance(rewards, list) and rewards:
                return np.mean(rewards)
        except:
            pass
        return np.nan

    df['mean_reward'] = df['rewards'].apply(safe_mean_reward)
    return df 

def plot_3d_surface(df):
    df = df[df['valid_path'] == True]
    if df.empty:
        print("No valid paths found in the DataFrame.")
        return
    # Create grid
    grid_x, grid_y = np.mgrid[df['gamma'].min():df['gamma'].max():100j,
                              df['epsilon'].min():df['epsilon'].max():100j]
    points = df[['gamma', 'epsilon']].values
    values = df['mean_reward'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Mean Reward')
    ax.set_title('3D Surface: Gamma vs Epsilon vs Mean Reward')
    fig.colorbar(surf, ax=ax, label='Mean Reward')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '3d_surface_reward.png'))
    plt.close()
    

# Plot 3D surface where color is iterations (learning speed)
def plot_3d_surface_colored_by_iters(df):
    df = df[df['valid_path'] == True]
    if df.empty:
        print("No valid paths found in the DataFrame.")
        return
    # Interpolation grid
    grid_x, grid_y = np.mgrid[
        df['gamma'].min():df['gamma'].max():100j,
        df['epsilon'].min():df['epsilon'].max():100j
    ]
    points = df[['gamma', 'epsilon']].values
    values = df['mean_reward'].values
    iters = df['iters'].values

    # Interpolate both Z (reward) and color (iters)
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    grid_c = griddata(points, iters, (grid_x, grid_y), method='cubic')

    # Normalize colors for colormap
    norm = plt.Normalize(np.nanmin(grid_c), np.nanmax(grid_c))
    colors = plt.cm.plasma(norm(grid_c))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, facecolors=colors, edgecolor='none')
    ax.set_xlabel('Gamma')
    ax.set_ylabel('Epsilon')
    ax.set_zlabel('Mean Reward')
    ax.set_title('3D Surface: Colored by Iterations to Converge')

    m = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, label='Iterations to Converge')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '3d_surface_iter.png'))
    plt.close()


def plot_3d_surface_decay(df):
    df = df[df['valid_path'] == True]
    if df.empty:
        print("No valid paths found in the DataFrame.")
        return
    # Interpolation grid
    grid_x, grid_y = np.mgrid[
        df['epsilon_decay'].min():df['epsilon_decay'].max():100j,
        df['epsilon_min'].min():df['epsilon_min'].max():100j
    ]
    points = df[['epsilon_decay', 'epsilon_min']].values
    values = df['mean_reward'].values

    # Interpolate both Z (reward) and color (iters)
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    grid_c = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Normalize colors for colormap
    norm = plt.Normalize(np.nanmin(grid_c), np.nanmax(grid_c))
    colors = plt.cm.plasma(norm(grid_c))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_x, grid_y, grid_z, facecolors=colors, edgecolor='none')
    ax.set_xlabel('Epsilon Decay')
    ax.set_ylabel('Epsilon Min')
    ax.set_zlabel('Mean Reward')
    ax.set_title('Performance by Epsilon Decay and Epsilon Min')

    m = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    m.set_array([])
    fig.colorbar(m, ax=ax, label='Mean Reward')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '3d_surface_decay.png'))
    plt.close()


def plot_reward_by_sigma(df):
    df = df[df['valid_path'] == True]
    if df.empty:    
        print("No valid paths found in the DataFrame.")
        return

    # Print boxplot statistics
    grouped = df.groupby('sigma')['mean_reward']
    print("Boxplot statistics by sigma:")
    for sigma_val, group in grouped:
        desc = group.describe()
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_whisker = group[group >= q1 - 1.5 * iqr].min()
        upper_whisker = group[group <= q3 + 1.5 * iqr].max()
        outliers = group[(group < q1 - 1.5 * iqr) | (group > q3 + 1.5 * iqr)]

        print(f"\nSigma = {sigma_val}")
        print(desc)
        print(f"Q1 (25th percentile): {q1:.3f}")
        print(f"Q3 (75th percentile): {q3:.3f}")
        print(f"IQR: {iqr:.3f}")
        print(f"Lower whisker: {lower_whisker:.3f}")
        print(f"Upper whisker: {upper_whisker:.3f}")
        print(f"Number of outliers: {len(outliers)}")
        if not outliers.empty:
            print("Outliers:", outliers.values)

    # Plot boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sigma', y='mean_reward', data=df)
    plt.title('Mean Reward by Sigma')
    plt.xlabel('Sigma (Environment Stochasticity)')
    plt.ylabel('Mean Reward')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'plot_reward_by_sigma.png'))
    plt.close()


def plot_valid_path_by_sigma(df):
    grouped = df.groupby('sigma')['valid_path'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='sigma', y='valid_path', data=grouped, marker='o')
    plt.title('Valid Path Rate by Sigma')
    plt.xlabel('Sigma')
    plt.ylabel('Valid Path Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'plot_valid_path_rate_by_sigma.png'))
    plt.close()


def plot_valid_path_rate_by_hyperparams(df):
    # Round values to clean the axis labels
    df['gamma_rounded'] = df['gamma'].round(2)
    df['epsilon_rounded'] = df['epsilon'].round(2)

    # Group and compute valid path rate
    grouped = df.groupby(['gamma_rounded', 'epsilon_rounded'])['valid_path'].mean().reset_index()

    # Pivot into grid format
    pivot = grouped.pivot(index='gamma_rounded', columns='epsilon_rounded', values='valid_path')

    # Sort for consistent axis ordering
    pivot = pivot.sort_index().sort_index(axis=1)

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot,
        cmap='YlGnBu',
        annot=False,  # no value labels
        cbar_kws={'label': 'Valid Path Rate'},
        linewidths=0.5,
        linecolor='lightgray'
    )
    plt.title('Valid Path Rate by Gamma and Epsilon')
    plt.xlabel('Epsilon (rounded)')
    plt.ylabel('Gamma (rounded)')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'plot_valid_path_rate_by_hyperpars.png'))
    plt.close()


def plot_hyperparam_distributions(df):
    # Preprocess rewards
    df['rewards'] = df['rewards'].apply(ast.literal_eval)
    df = df[df['rewards'].apply(lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) for i in x))]
    df['mean_reward'] = df['rewards'].apply(np.mean)
    df['reward_variance'] = df['rewards'].apply(np.var)

    # Identify optimal path length
    if not df['optimal_path_length'].empty:
        op_len = df['optimal_path_length'].mode()[0]
    else:
        op_len = None

    # Split data
    invalid_runs = df[(df['valid_path'] == False) | (df['optimal_path_length'] != op_len)]
    valid_runs = df[(df['valid_path'] == True) & (df['optimal_path_length'] == op_len)]

    # Parameters to compare
    hyperparams = ['gamma', 'epsilon', 'epsilon_decay', 'epsilon_min', 'sigma']

    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    axs = axs.flatten()

    for i, param in enumerate(hyperparams):
        if param in df.columns:
            sns.kdeplot(df[param], label='All Runs', ax=axs[i])
            sns.kdeplot(invalid_runs[param], label='Invalid/Non-optimal', ax=axs[i])
            axs[i].set_title(f'Distribution of {param}')
            axs[i].legend()
        else:
            axs[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'plot_valid_path_rate_by_hp.png'))
    plt.close()


def plot_mc_reward_surfaces(df):

    # Safe parsing of rewards
    def safe_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return np.nan
        elif isinstance(val, list):
            return val
        else:
            return np.nan

    df['rewards'] = df['rewards'].apply(safe_eval)
    df = df[df['rewards'].apply(lambda x: isinstance(x, list) and all(isinstance(i, (int, float)) for i in x))]
    df['mean_reward'] = df['rewards'].apply(np.mean)
    df = df[df['valid_path'] == True]

    # Select first 4 sigma values
    selected_sigmas = sorted(df['sigma'].round(3).unique())[:4]

    fig = plt.figure(figsize=(16, 12))
    canvas = FigureCanvas(fig)

    for i, sigma_val in enumerate(selected_sigmas):
        sub_df = df[np.isclose(df['sigma'], sigma_val, atol=1e-3)]

        if sub_df.empty:
            continue

        gamma_vals = np.linspace(sub_df['gamma'].min(), sub_df['gamma'].max(), 50)
        epsilon_vals = np.linspace(sub_df['epsilon'].min(), sub_df['epsilon'].max(), 50)
        gamma_grid, epsilon_grid = np.meshgrid(gamma_vals, epsilon_vals)

        reward_grid = griddata((sub_df['gamma'], sub_df['epsilon']), sub_df['mean_reward'], (gamma_grid, epsilon_grid), method='cubic')
        iters_grid = griddata((sub_df['gamma'], sub_df['epsilon']), sub_df['iters'], (gamma_grid, epsilon_grid), method='cubic')

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Normalize colors safely
        valid_iters = iters_grid[np.isfinite(iters_grid)]
        norm = Normalize(vmin=np.min(valid_iters), vmax=np.max(valid_iters))
        facecolors = plasma(norm(iters_grid))

        ax.plot_surface(
            gamma_grid, epsilon_grid, reward_grid,
            facecolors=facecolors,
            rstride=1, cstride=1, linewidth=0, antialiased=True
        )
        ax.set_title(f'Mean Reward (Sigma ≈ {sigma_val})\nColored by Iterations')
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Epsilon')
        ax.set_zlabel('Mean Reward')

    # Shared colorbar
    norm = Normalize(vmin=df['iters'].min(), vmax=df['iters'].max())
    sm = ScalarMappable(cmap='plasma', norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label='Iterations to Converge')

    os.makedirs(IMG_DIR, exist_ok=True)
    canvas.print_figure(os.path.join(IMG_DIR, 'MC-reward_surfaces_2x2_colored_by_iters.png'))
    plt.close()


def run_analysis(path):
    df = load_data(path)
    if not df[df['valid_path'] == True].empty:
        plot_3d_surface(df)
        plot_3d_surface_colored_by_iters(df) 
        plot_3d_surface_decay(df)
        plot_reward_by_sigma(df)
        plot_valid_path_by_sigma(df)
        plot_valid_path_rate_by_hyperparams(df)  
        plot_hyperparam_distributions(df)
        plot_mc_reward_surfaces(df)
    else:
        print("No valid paths found in the DataFrame.")

if __name__ == '__main__':
    os.makedirs(IMG_DIR, exist_ok=True)  # Ensure the folder exists
    for path in CSV_PATH:
        run_analysis(path)


