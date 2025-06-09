from datachallengeg15.env_dqn import Environment
from datachallengeg15.dqn import train_dqn, test_dqn_agent
import numpy as np

maze = np.load("datachallengeg15/warehouse.npy").astype(np.int8)

progress_multipliers = [1.0, 2.0, 3.0]
step_penalties = [-0.01, -0.05, -0.1]
goal_bonuses = [10.0, 20.0, 30.0]

results = []

for p in progress_multipliers:
    for s in step_penalties:
        for g in goal_bonuses:
            print(f"\nTesting: progress={p}, penalty={s}, goal_bonus={g}")
            env = Environment(maze, reward_config={"progress": p, "step": s, "goal": g})

            agent = train_dqn(episodes=10, max_steps=300, update_freq=10)

            # Evaluate agent
            success_rate, avg_steps = test_dqn_agent(agent, env, episodes=1, max_steps=100, return_metrics=True)

            results.append((p, s, g, avg_steps, success_rate))

# Sort and print best configs
results.sort(key=lambda x: x[3])  # sort by avg_steps
for r in results[:5]:
    print(f"progress={r[0]}, penalty={r[1]}, goal_bonus={r[2]} → steps={r[3]}, success={r[4]}")
