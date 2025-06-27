import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Experiment 1: Reward-Function Sweep
df_dqn = pd.read_csv("hpo_dqn_reward.csv")
df_ppo = pd.read_csv("hpo_ppo_reward.csv")

# Plot 1: Step Penalty vs Final Reward (colored by Distance Penalty)
# DQN
norm_dqn = plt.Normalize(df_dqn["distance_penalty_coef"].min(), df_dqn["distance_penalty_coef"].max())
colors_dqn = cm.viridis(norm_dqn(df_dqn["distance_penalty_coef"]))
plt.scatter(df_dqn["step_penalty"], df_dqn["final_reward"], c=colors_dqn, s=50, label="DQN", marker="o")

# PPO
norm_ppo = plt.Normalize(df_ppo["distance_penalty_coef"].min(), df_ppo["distance_penalty_coef"].max())
colors_ppo = cm.viridis(norm_ppo(df_ppo["distance_penalty_coef"]))
plt.scatter(df_ppo["step_penalty"], df_ppo["final_reward"], c=colors_ppo, s=50, label="PPO", marker="^")

plt.title("Step Penalty vs Final Reward\n(colored by Distance Penalty)")
plt.xlabel("Step Penalty")
plt.ylabel("Final Reward")
plt.colorbar(cm.ScalarMappable(norm=norm_dqn, cmap="viridis"), label="Distance Penalty Coef")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/compare_step_penalty_vs_final_reward.png")
plt.close()

# Plot 2: Episode Reward Distributions for Representative Step Penalties
# Select representative step penalty values: minimum, median, and maximum
unique_steps = sorted(df_dqn["step_penalty"].unique())
rep_steps = [unique_steps[0], unique_steps[len(unique_steps) // 2], unique_steps[-1]]
combined_rewards = []
xtick_labels = []

for sp in rep_steps:
    # Select the first matching configuration for that step penalty
    dqn_row = df_dqn[df_dqn["step_penalty"] == sp].iloc[0]
    ppo_row = df_ppo[df_ppo["step_penalty"] == sp].iloc[0]

    # Parse the stringified episode reward lists into actual Python lists
    dqn_rewards = ast.literal_eval(dqn_row["episode_rewards"])
    ppo_rewards = ast.literal_eval(ppo_row["episode_rewards"])
    
    # Append to the reward list
    combined_rewards.extend([dqn_rewards, ppo_rewards])
    
    # Create alternating labels: DQN and PPO for each step value
    xtick_labels.extend([f"DQN\n{sp:.3f}", f"PPO\n{sp:.3f}"])

# Plot
plt.figure(figsize=(10, 6))
plt.boxplot(combined_rewards, labels=xtick_labels)
plt.title("Episode Reward Distributions\n(Representative Step Penalties)")
plt.xlabel("Algorithm and Step Penalty")
plt.ylabel("Episode Reward")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/episode_reward_distributions_dqn_vs_ppo.png")
plt.close()

# Plot 3: Success Rate vs Step Penalty (colored by Distance Penalty), DQN vs PPO
plt.figure(figsize=(8, 6))

norm_dqn = plt.Normalize(df_dqn["distance_penalty_coef"].min(), df_dqn["distance_penalty_coef"].max())
colors_dqn = cm.plasma(norm_dqn(df_dqn["distance_penalty_coef"]))
plt.scatter(df_dqn["step_penalty"], df_dqn["success_rate"], c=colors_dqn, s=50, label="DQN", marker="o", edgecolors="black")

norm_ppo = plt.Normalize(df_ppo["distance_penalty_coef"].min(), df_ppo["distance_penalty_coef"].max())
colors_ppo = cm.plasma(norm_ppo(df_ppo["distance_penalty_coef"]))
plt.scatter(df_ppo["step_penalty"], df_ppo["success_rate"], c=colors_ppo, s=50, label="PPO", marker="^", edgecolors="black")

plt.title("Step Penalty vs Success Rate\n(colored by Distance Penalty)")
plt.xlabel("Step Penalty")
plt.ylabel("Success Rate")
plt.colorbar(cm.ScalarMappable(norm=norm_dqn, cmap="plasma"), label="Distance Penalty Coef")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/step_penalty_vs_success_rate_dqn_vs_ppo.png")
plt.close()

# Experiment 2: Target/Seed Sweep
df2_dqn = pd.read_csv("hpo_dqn_target.csv")
df2_ppo = pd.read_csv("hpo_ppo_target_seed.csv")
# Create a string label for each target coordinate
df2_dqn["target_label"] = df2_dqn.apply(lambda row: f"({row['target_x']}, {row['target_y']})", axis=1)
df2_ppo["target_label"] = df2_ppo.apply(lambda row: f"({row['target_x']}, {row['target_y']})", axis=1)

# Group by target label
group_dqn = df2_dqn.groupby("target_label")
group_ppo = df2_ppo.groupby("target_label")

# Compute mean rewards
mean_rewards_dqn = group_dqn["final_reward"].mean()
mean_rewards_ppo = group_ppo["final_reward"].mean()

# Ensure same target order
targets = sorted(set(mean_rewards_dqn.index) & set(mean_rewards_ppo.index))
x = np.arange(len(targets))  # x locations for groups
bar_width = 0.35

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, [mean_rewards_dqn[t] for t in targets], width=bar_width, label="DQN")
plt.bar(x + bar_width/2, [mean_rewards_ppo[t] for t in targets], width=bar_width, label="PPO")
plt.title("Mean Final Reward by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Final Reward")
plt.xticks(x, targets, rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/target_vs_mean_final_reward_dqn_vs_ppo.png")
plt.close()

# Plot 2: Mean Success Rate by Target Location
# Compute mean success rates by target for both algorithms
mean_success_dqn = df2_dqn.groupby("target_label")["success_rate"].mean()
mean_success_ppo = df2_ppo.groupby("target_label")["success_rate"].mean()

# Ensure target labels are aligned and sorted
common_targets = sorted(set(mean_success_dqn.index) & set(mean_success_ppo.index))
x = np.arange(len(common_targets))  # x-axis positions
bar_width = 0.35

# Create comparison bar chart
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width / 2,
        [mean_success_dqn[t] for t in common_targets],
        width=bar_width, label="DQN", color="cornflowerblue")

plt.bar(x + bar_width / 2,
        [mean_success_ppo[t] for t in common_targets],
        width=bar_width, label="PPO", color="lightcoral")

# Annotate and format the plot
plt.title("Mean Success Rate by Target Location (DQN vs PPO)")
plt.xlabel("Target Location")
plt.ylabel("Mean Success Rate")
plt.xticks(ticks=x, labels=common_targets, rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/target_vs_mean_success_rate_dqn_vs_ppo.png")
plt.close()

# Plot 3: Mean Steps to Goal by Target Location
# Compute mean steps by target for both algorithms
mean_steps_dqn = df2_dqn.groupby("target_label")["avg_steps"].mean()
mean_steps_ppo = df2_ppo.groupby("target_label")["avg_steps"].mean()

# Find common target labels and sort them
common_targets = sorted(set(mean_steps_dqn.index) & set(mean_steps_ppo.index))
x = np.arange(len(common_targets))
bar_width = 0.35

# Create side-by-side bar chart for comparison
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width / 2,
        [mean_steps_dqn[t] for t in common_targets],
        width=bar_width, label="DQN", color="cornflowerblue")

plt.bar(x + bar_width / 2,
        [mean_steps_ppo[t] for t in common_targets],
        width=bar_width, label="PPO", color="lightcoral")

# Plot
plt.title("Mean Steps to Goal by Target Location (DQN vs PPO)")
plt.xlabel("Target Location")
plt.ylabel("Mean Steps")
plt.xticks(ticks=x, labels=common_targets, rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/target_vs_mean_steps_dqn_vs_ppo.png")
plt.close()

# Plot 4: Final Reward Distribution by Target Location
common_targets = sorted(set(df2_dqn["target_label"]) & set(df2_ppo["target_label"]))
box_data = []
xtick_labels = []

for tgt in common_targets:
    rewards_dqn = df2_dqn[df2_dqn["target_label"] == tgt]["final_reward"].values
    rewards_ppo = df2_ppo[df2_ppo["target_label"] == tgt]["final_reward"].values
    box_data.extend([rewards_dqn, rewards_ppo])
    xtick_labels.extend([f"DQN\n{tgt}", f"PPO\n{tgt}"])

plt.figure(figsize=(max(12, len(xtick_labels) * 0.6), 6))
plt.boxplot(box_data, labels=xtick_labels)
plt.title("Final Reward Distribution by Target Location (DQN vs PPO)")
plt.xlabel("Target Location")
plt.ylabel("Final Reward")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/target_reward_variability_dqn_vs_ppo.png")
plt.close()