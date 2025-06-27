import ast
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Experiment 1: Reward-Function Sweep
df1 = pd.read_csv("hpo_dqn_reward.csv")

# Plot 1: Step Penalty vs Final Reward (colored by Distance Penalty)
plt.figure()
# Normalize the distance-penalty coefficient for the colormap
norm = plt.Normalize(df1["distance_penalty_coef"].min(), df1["distance_penalty_coef"].max())
colors = cm.viridis(norm(df1["distance_penalty_coef"]))
plt.scatter(df1["step_penalty"], df1["final_reward"], c=colors, s=50)
plt.title("Step Penalty vs Final Reward\n(colored by Distance Penalty)")
plt.xlabel("Step Penalty")
plt.ylabel("Final Reward")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), label="Distance Penalty Coef")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/step_penalty_vs_final_reward.png")
plt.close()

# Plot 2: Episode Reward Distributions for Representative Step Penalties
# Choose three representative step_penalty values: minimum, median, maximum
unique_steps = sorted(df1["step_penalty"].unique())
rep_steps = [unique_steps[0], unique_steps[len(unique_steps)//2], unique_steps[-1]]
labels = [f"Step={s:.3f}" for s in rep_steps]
rewards_data = []

for sp in rep_steps:
    # Select the first matching configuration for that step penalty
    row = df1[df1["step_penalty"] == sp].iloc[0]
    rewards = ast.literal_eval(row["episode_rewards"])
    rewards_data.append(rewards)

plt.figure()
plt.boxplot(rewards_data, labels=labels)
plt.title("Episode Reward Distributions\n(Representative Step Penalties)")
plt.xlabel("Step Penalty")
plt.ylabel("Episode Reward")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/episode_reward_distributions_by_step_penalty.png")
plt.close()

# Experiment 2: Target/Seed Sweep
df2 = pd.read_csv("hpo_dqn_target.csv")

# Create a string label for each target coordinate
df2["target_label"] = df2.apply(lambda row: f"({row['target_x']}, {row['target_y']})", axis=1)

group = df2.groupby("target_label")

# Plot 1: Mean Final Reward by Target Location
mean_rewards = group["final_reward"].mean()
plt.figure()
plt.bar(mean_rewards.index, mean_rewards.values)
plt.title("Mean Final Reward by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Final Reward")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/target_vs_mean_final_reward.png")
plt.close()

# Plot 2: Mean Success Rate by Target Location
mean_success = group["success_rate"].mean()
plt.figure()
plt.bar(mean_success.index, mean_success.values)
plt.title("Mean Success Rate by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Success Rate")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/target_vs_mean_success_rate.png")
plt.close()

# Plot 3: Mean Steps to Goal by Target Location
mean_steps = group["avg_steps"].mean()
plt.figure()
plt.bar(mean_steps.index, mean_steps.values)
plt.title("Mean Steps to Goal by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Steps")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/target_vs_mean_steps.png")
plt.close()

# Plot 4: Final Reward Distribution by Target Location
reward_lists = [group.get_group(lbl)["final_reward"].values for lbl in mean_rewards.index]
plt.figure()
plt.boxplot(reward_lists, labels=mean_rewards.index)
plt.title("Final Reward Distribution by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Final Reward")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/target_reward_variability.png")
plt.close()

# Plot 5: Success Rate vs Step Penalty (colored by Distance Penalty)
plt.figure()
# Normalize the distance-penalty coefficient for color mapping
norm = plt.Normalize(df1["distance_penalty_coef"].min(), df1["distance_penalty_coef"].max())
colors = cm.plasma(norm(df1["distance_penalty_coef"]))
plt.scatter(df1["step_penalty"], df1["success_rate"], c=colors, s=50)
plt.title("Step Penalty vs Success Rate\n(colored by Distance Penalty)")
plt.xlabel("Step Penalty")
plt.ylabel("Success Rate")
plt.colorbar(cm.ScalarMappable(norm=norm, cmap="plasma"), label="Distance Penalty Coef")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/step_penalty_vs_success_rate.png")
plt.close()