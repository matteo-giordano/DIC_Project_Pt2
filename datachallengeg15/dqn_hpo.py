import argparse
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from utils import read_config, set_random_seeds
from dqn_trainer import DQNTrainer

# --- Reward function constructor ---
def make_reward_fn(step_penalty, distance_penalty_coef, loop_penalty, goal_bonus=1.0):
    def reward_fn(env, done, history):
        r = -step_penalty
        d = np.linalg.norm(env.maze.agent_pos - env.maze.goal_pos)
        r -= distance_penalty_coef * (d / env.map_diagonal_norm)
        if tuple(env.maze.agent_pos) in history:
            r -= loop_penalty
        if done and tuple(env.maze.agent_pos) == tuple(env.maze.goal_pos):
            r += goal_bonus
        return r
    return reward_fn

# --- Utility for running a single trial ---
def run_trial(config: dict) -> dict:
    set_random_seeds(config['seed'])
    try:
        trainer = DQNTrainer(config, reward_fn=config['reward_fn'])
        data = trainer.train_with_history()

        total_rewards = [ep['total_reward'] for ep in data]
        success_flags = [ep['success'] for ep in data]
        steps_list = [ep['steps'] for ep in data]
        all_values = np.concatenate([ep['value_estimates'] for ep in data])

        return {
            'status': 'ok',
            'final_reward': float(np.mean(total_rewards[-10:])),
            'success_rate': float(np.mean(success_flags)),
            'avg_steps': float(np.mean(steps_list)),
            'seed': config['seed'],
            'target_x': config['env']['goal_pos'][0],
            'target_y': config['env']['goal_pos'][1],
            'success_reward': config['reward']['args'].get('goal_bonus', 1.0),
            'step_penalty': config['reward']['args']['step_penalty'],
            'distance_penalty_coef': config['reward']['args']['distance_penalty_coef'],
            'loop_penalty': config['reward']['args']['loop_penalty'],
            'mean_value': float(np.mean(all_values)),
            'min_value': float(np.min(all_values)),
            'max_value': float(np.max(all_values)),
            'episode_rewards': total_rewards
        }
    except Exception as e:
        return {
            'status': 'error',
            'final_reward': None,
            'seed': config['seed'],
            'target_x': config['env']['goal_pos'][0],
            'target_y': config['env']['goal_pos'][1],
            'error': str(e)
        }

# --- Experiment 1: Reward Sweep ---
def experiment_reward_sweep(base_config_path='datachallengeg15/config_dqn.yaml', n_reward_steps=20, output_csv='hpo_dqn_reward.csv', test=False):
    base_config = read_config(base_config_path)
    target = [9.5, 27.5]
    seed = 1

    step_penalties = np.geomspace(0.0001, 0.01, n_reward_steps)
    distance_penalties = np.linspace(0, 0.1, n_reward_steps)
    loop_penalties = np.linspace(0, 1.0, n_reward_steps)

    configs = []
    for i in range(n_reward_steps):
        cfg = copy.deepcopy(base_config)
        cfg['seed'] = seed
        cfg['env']['goal_pos'] = target
        cfg['trainer']['live_tracking'] = False
        cfg['reward']['args'] = {
            'goal_bonus': base_config['reward']['args'].get('goal_bonus', 1.0),
            'step_penalty': float(step_penalties[i]),
            'distance_penalty_coef': float(distance_penalties[i]),
            'loop_penalty': float(loop_penalties[i])
        }
        cfg['reward_fn'] = make_reward_fn(**cfg['reward']['args'])
        configs.append(cfg)

    if test:
        configs = configs[:2]

    results = [run_trial(cfg) for cfg in tqdm(configs, desc='Reward Sweep')]
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Reward sweep results saved to {output_csv}")

# --- Experiment 2: Target/Seed Sweep ---
def experiment_target_seed_sweep(base_config_path='datachallengeg15/config_dqn.yaml', n_seeds=10, output_csv='hpo_dqn_target.csv', test=False):
    base_config = read_config(base_config_path)
    reward_args = {
        'goal_bonus': base_config['reward']['args'].get('goal_bonus', 1.0),
        'step_penalty': 0.01,
        'distance_penalty_coef': 0.1,
        'loop_penalty': 1.0
    }
    targets = [
        [9.5, 27.5], [7.5, 17.5], [16.5, 14], [15.5, 35.5], [16.5, 1.5]
    ]

    configs = []
    for targ in targets:
        for seed in range(1, n_seeds + 1):
            cfg = copy.deepcopy(base_config)
            cfg['seed'] = seed
            cfg['env']['goal_pos'] = targ
            cfg['trainer']['live_tracking'] = False
            cfg['reward']['args'] = reward_args
            cfg['reward_fn'] = make_reward_fn(**reward_args)
            configs.append(cfg)

    if test:
        configs = configs[:2]

    results = [run_trial(cfg) for cfg in tqdm(configs, desc='Target/Seed Sweep')]
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Target/seed sweep results saved to {output_csv}")

# --- CLI Entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DQN HPO experiments.")
    parser.add_argument('--phase', choices=['reward', 'target'], required=True,
                        help='Which experiment to run: reward or target')
    parser.add_argument('-t', '--test', action='store_true', help='Run experiment in test mode')
    args = parser.parse_args()

    if args.phase == 'reward':
        experiment_reward_sweep(test=args.test)
    else:
        experiment_target_seed_sweep(test=args.test)
