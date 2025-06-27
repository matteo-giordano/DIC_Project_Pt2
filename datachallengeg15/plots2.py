# load dataset 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df_reward = pd.read_csv('hpo_ppo_reward.csv')
df_target = pd.read_csv('hpo_ppo_target_seed.csv')


# Calculate distance from (1.5, 22.5)
df_target['distance'] = np.sqrt((df_target['target_x'] - 1.5)**2 + (df_target['target_y'] - 22.5)**2)

# Get unique targets and sort by distance
unique_targets = df_target[['target_x', 'target_y', 'distance']].drop_duplicates().sort_values('distance').reset_index(drop=True)
labels = ['easy1', 'easy2', 'medium1', 'medium2', 'hard']
unique_targets['target_label'] = labels[:len(unique_targets)]

# Merge labels back to df_target
df_target = df_target.merge(unique_targets[['target_x', 'target_y', 'target_label']], on=['target_x', 'target_y'], how='left')

# Define colors for each label
target_palette = {
    'easy1': 'green',
    'easy2': 'green',
    'medium1': 'orange',
    'medium2': 'orange',
    'hard': 'red',
}

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_target, x='target_label', y='final_reward', order=labels, palette=target_palette)
sns.stripplot(data=df_target, x='target_label', y='final_reward', color='black', alpha=0.5, jitter=True, order=labels)
plt.title('Performance Variability Across Seeds for Each Target (by Difficulty)')
plt.ylabel('Final Reward')
plt.xlabel('Target (Difficulty)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_target, x='seed', y='final_reward')
sns.stripplot(data=df_target, x='seed', y='final_reward', color='black', alpha=0.5, jitter=True)
plt.title('Performance Distribution by Seed (All Targets)')
plt.xlabel('Seed')
plt.ylabel('Final Reward')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df_target, x='final_reward')
plt.title('Cumulative Distribution of Final Rewards (All Targets)')
plt.xlabel('Final Reward')
plt.ylabel('Cumulative Probability')
plt.tight_layout()
plt.show()


target_var = df_target.groupby('target_label')['final_reward'].std().reindex(labels).reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=target_var, x='target_label', y='final_reward', palette=target_palette, order=labels)
plt.title('Performance Variance (Std) by Target (by Difficulty)')
plt.xlabel('Target (Difficulty)')
plt.ylabel('Std of Final Reward')
plt.tight_layout()
plt.show()

target_stats = df_target.groupby('target_label')['final_reward'].agg(['min', 'max']).reindex(labels).reset_index()
plt.figure(figsize=(10, 5))
plt.plot(target_stats['target_label'], target_stats['min'], marker='o', label='Worst (Min)', color='gray')
plt.plot(target_stats['target_label'], target_stats['max'], marker='o', label='Best (Max)', color='black')
for i, label in enumerate(labels):
    plt.gca().axvspan(i-0.5, i+0.5, color=target_palette[label], alpha=0.15)
plt.title('Best and Worst Final Reward per Target (by Difficulty)')
plt.xlabel('Target (Difficulty)')
plt.ylabel('Final Reward')
plt.legend()
plt.tight_layout()
plt.show()

target_success = df_target.groupby('target_label')['success'].mean().reindex(labels).reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=target_success, x='target_label', y='success', palette=target_palette, order=labels)
plt.title('Mean Success Rate Across Seeds for Each Target (by Difficulty)')
plt.ylabel('Mean Success Rate')
plt.xlabel('Target (Difficulty)')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()