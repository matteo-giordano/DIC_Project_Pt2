import ast
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def fix_ppo_episode_rewards(df):
    corrected_rows = []
    i = 0
    while i < len(df):
        row = df.iloc[i].copy()
        episode_str = str(row['episode_rewards']).strip()
        if episode_str.startswith("[") and not episode_str.endswith("]"):
            fragments = [episode_str]
            j = i + 1
            while j < len(df):
                next_row = df.iloc[j]
                values = [
                    str(float(x)) for x in next_row.values 
                    if isinstance(x, (float, int, np.floating, np.integer))
                ]
                if len(values) == 0:
                    break
                fragments.append(",".join(values))
                j += 1
            full_str = ",".join(fragments)
            full_str = full_str if full_str.startswith("[") else "[" + full_str
            full_str = full_str if full_str.endswith("]") else full_str + "]"
            try:
                parsed = ast.literal_eval(full_str)
                if isinstance(parsed, list):
                    parsed = [float(x) for x in parsed]
                    row['episode_rewards'] = str(parsed)
                else:
                    row['episode_rewards'] = "[]"
            except:
                row['episode_rewards'] = "[]"
            corrected_rows.append(row)
            i = j
        else:
            corrected_rows.append(row)
            i += 1
    return pd.DataFrame(corrected_rows)


# Experiment 1: Reward-Function Sweep
df1_dqn = pd.read_csv("hpo_dqn_reward.csv")
df1_ppo = pd.read_csv("hpo_ppo_reward.csv")
df1_ppo = fix_ppo_episode_rewards(df1_ppo)

# Plot 1: Step Penalty vs Final Reward (colored by Distance Penalty)
fig, ax = plt.subplots(figsize=(10, 6))

combined_df = pd.concat([df1_dqn, df1_ppo])
norm = plt.Normalize(combined_df["distance_penalty_coef"].min(), combined_df["distance_penalty_coef"].max())

colors_dqn = cm.viridis(norm(df1_dqn["distance_penalty_coef"]))
colors_ppo = cm.plasma(norm(df1_ppo["distance_penalty_coef"]))

ax.scatter(df1_dqn["step_penalty"], df1_dqn["final_reward"],
           c=colors_dqn, marker='o', label='DQN', s=50, alpha=0.7, edgecolor='black')
ax.scatter(df1_ppo["step_penalty"], df1_ppo["final_reward"],
           c=colors_ppo, marker='^', label='PPO', s=50, alpha=0.7, edgecolor='black')

ax.set_title("Step Penalty vs Final Reward (DQN vs PPO)")
ax.set_xlabel("Step Penalty")
ax.set_ylabel("Final Reward")

sm = cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Distance Penalty Coefficient', rotation=270, labelpad=15)

ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("results/step_penalty_vs_final_reward_dqn_ppo.png")

# Plot 2: Episode Reward Distributions for Representative Step Penalties
# Select representative step penalties (minimum, median, maximum)
def parse_rewards(s):
    if pd.isna(s) or s == "":
        return []
    if isinstance(s, float):
        return [s]
    if s.strip().startswith("array"):
        import re
        content = re.search(r"\[(.*?)\]", s)
        if content:
            try:
                return [float(x) for x in content.group(1).split(",")]
            except Exception:
                return []
        else:
            return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, float):
            return [v]
        elif isinstance(v, list):
            return v
        else:
            return []
    except Exception:
        try:
            return [float(x) for x in s.strip().split()]
        except Exception:
            return []

combined_df = pd.concat([df1_dqn, df1_ppo])
unique_steps = sorted(combined_df["step_penalty"].unique())
rep_steps = [unique_steps[0], unique_steps[len(unique_steps)//2], unique_steps[-1]]
labels = [f"Step={s:.3f}" for s in rep_steps]

rewards_data_dqn = []
rewards_data_ppo = []

for sp in rep_steps:
    row_dqn = df1_dqn[df1_dqn["step_penalty"] == sp].iloc[0]
    rewards_dqn = parse_rewards(row_dqn["episode_rewards"])
    rewards_data_dqn.append(rewards_dqn)

    row_ppo = df1_ppo[df1_ppo["step_penalty"] == sp].iloc[0]
    rewards_ppo = parse_rewards(row_ppo["episode_rewards"])
    rewards_data_ppo.append(rewards_ppo)

positions = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

bp1 = ax.boxplot(rewards_data_dqn, positions=positions - width/2, widths=width, patch_artist=True,
                 boxprops=dict(facecolor='skyblue'), medianprops=dict(color='navy'))
bp2 = ax.boxplot(rewards_data_ppo, positions=positions + width/2, widths=width, patch_artist=True,
                 boxprops=dict(facecolor='orange'), medianprops=dict(color='darkred'))

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_title("Episode Reward Distributions by Step Penalty")
ax.set_xlabel("Step Penalty")
ax.set_ylabel("Episode Reward")
ax.grid(True, axis="y", alpha=0.3)

ax.plot([], c='skyblue', label='DQN')
ax.plot([], c='orange', label='PPO')
ax.legend()
plt.tight_layout()
plt.savefig("results/episode_reward_distributions_dqn_ppo.png")


# Plot 3: Success Rate vs Step Penalty (colored by Distance Penalty)
# Normalize color mapping for distance penalty coefficient
plt.figure(figsize=(10, 6))
colors_dqn = cm.plasma(norm(df1_dqn["distance_penalty_coef"]))
colors_ppo = cm.plasma(norm(df1_ppo["distance_penalty_coef"]))

plt.scatter(df1_dqn["step_penalty"], df1_dqn["success_rate"], c=colors_dqn, marker='o', label='DQN', s=50, edgecolor='black', alpha=0.7)
plt.scatter(df1_ppo["step_penalty"], df1_ppo["success"], c=colors_ppo, marker='^', label='PPO', s=50, edgecolor='black', alpha=0.7)

plt.title("Step Penalty vs Success Rate (DQN vs PPO)")
plt.xlabel("Step Penalty")
plt.ylabel("Success Rate")
sm = cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('Distance Penalty Coefficient', rotation=270, labelpad=15)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/step_penalty_vs_success_rate_dqn_ppo.png")


# Experiment 2: Target/Seed Sweep
df2_dqn = pd.read_csv("hpo_dqn_target.csv")
df2_ppo = pd.read_csv("hpo_ppo_target_seed.csv")

# Create a string label for each target coordinate
df2_dqn["target_label"] = df2_dqn.apply(lambda row: f"({row['target_x']}, {row['target_y']})", axis=1)
df2_ppo["target_label"] = df2_ppo.apply(lambda row: f"({row['target_x']}, {row['target_y']})", axis=1)

# Group by target_label and calculate mean final rewards
mean_rewards_dqn = df2_dqn.groupby("target_label")["final_reward"].mean()
mean_rewards_ppo = df2_ppo.groupby("target_label")["final_reward"].mean()

# Plot 1: Mean Final Reward by Target Location
# Plot bar plots side by side
plt.figure(figsize=(10, 6))
x = np.arange(len(mean_rewards_dqn.index))
width = 0.35
plt.bar(x - width/2, mean_rewards_dqn.values, width=width, label='DQN', color='skyblue')
plt.bar(x + width/2, mean_rewards_ppo.values, width=width, label='PPO', color='orange')
plt.title("Mean Final Reward by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Final Reward")
plt.xticks(x, mean_rewards_dqn.index, rotation=45)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/target_vs_mean_final_reward_dqn_ppo.png")


# Plot 2: Mean Success Rate by Target Location
# Group by target_label and calculate mean success rates (DQN has success_rate, PPO has success)
mean_success_dqn = df2_dqn.groupby("target_label")["success_rate"].mean()
mean_success_ppo = df2_ppo.groupby("target_label")["success"].mean()
x = np.arange(len(mean_success_dqn.index))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, mean_success_dqn.values, width=width, label='DQN', color='skyblue')
plt.bar(x + width/2, mean_success_ppo.values, width=width, label='PPO', color='orange')
plt.title("Mean Success Rate by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Success Rate")
plt.xticks(x, mean_success_dqn.index, rotation=45)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/target_vs_mean_success_rate_dqn_ppo.png")


# Plot 3: Mean Steps to Goal by Target Location
mean_steps_dqn = df2_dqn.groupby("target_label")["avg_steps"].mean()
mean_steps_ppo = df2_ppo.groupby("target_label")["steps"].mean()

x = np.arange(len(mean_steps_dqn.index))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, mean_steps_dqn.values, width=width, label='DQN', color='skyblue')
plt.bar(x + width/2, mean_steps_ppo.values, width=width, label='PPO', color='orange')
plt.title("Mean Steps to Goal by Target Location")
plt.xlabel("Target Location")
plt.ylabel("Mean Steps")
plt.xticks(x, mean_steps_dqn.index, rotation=45)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("results/target_vs_mean_steps_dqn_ppo.png")


# Plot 4: Final Reward Distribution by Target Location
positions = np.arange(len(mean_steps_dqn.index))
width = 0.3

reward_lists_dqn = [df2_dqn[df2_dqn["target_label"] == lbl]["final_reward"].values for lbl in mean_steps_dqn.index]
reward_lists_ppo = [df2_ppo[df2_ppo["target_label"] == lbl]["final_reward"].values for lbl in mean_steps_dqn.index]

fig, ax = plt.subplots(figsize=(10, 6))

bp1 = ax.boxplot(reward_lists_dqn, positions=positions - width/2, widths=width, patch_artist=True,
                 boxprops=dict(facecolor='skyblue'), medianprops=dict(color='navy'))
bp2 = ax.boxplot(reward_lists_ppo, positions=positions + width/2, widths=width, patch_artist=True,
                 boxprops=dict(facecolor='orange'), medianprops=dict(color='darkred'))

ax.set_xticks(positions)
ax.set_xticklabels(mean_steps_dqn.index)
ax.set_title("Final Reward Distribution by Target Location")
ax.set_xlabel("Target Location")
ax.set_ylabel("Final Reward")
ax.grid(True, axis="y", alpha=0.3)
ax.plot([], c='skyblue', label='DQN')
ax.plot([], c='orange', label='PPO')
ax.legend()
plt.tight_layout()
plt.savefig("results/target_reward_variability_dqn_ppo.png")
