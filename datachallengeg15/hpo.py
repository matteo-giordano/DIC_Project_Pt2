import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from main import read_config, set_random_seeds
from trainer import PPOTrainer

# --- Utility for running a single trial ---
def run_trial(config: dict) -> dict:
    set_random_seeds(config['seed'])
    try:
        trainer = PPOTrainer(config)
        _, episode_rewards, _ = trainer.train()
        # Run evaluation and collect metrics
        eval_metrics = trainer.eval()
        result = {
            'status': 'success',
            'final_reward': np.mean(episode_rewards[-10:]) if episode_rewards else 0,
            'seed': config['seed'],
            'target_x': config['env']['goal_pos'][0],
            'target_y': config['env']['goal_pos'][1],
            **config['reward']['args']
        }
        # Add eval metrics to result (flatten if needed)
        for k, v in eval_metrics.items():
            if isinstance(v, (list, np.ndarray)):
                result[k] = str(v)
            else:
                result[k] = v
        # Store the full episode rewards from training
        result['episode_rewards'] = str(episode_rewards)
        return result
    except Exception as e:
        return {
            'status': 'failure',
            'final_reward': None,
            'seed': config['seed'],
            'target_x': config['env']['goal_pos'][0],
            'target_y': config['env']['goal_pos'][1],
            'error': str(e),
            **config['reward']['args']
        }

# --- Experiment 1: Reward Sweep ---
def experiment_reward_sweep(base_config_path='datachallengeg15/config.yaml', n_reward_steps=20, output_csv='hpo_ppo_reward.csv'):
    base_config = read_config(base_config_path)
    # Fixed target and seed
    target = [9.5, 27.5]
    seed = 1
    # Sweep reward parameters
    min_step_penalty = 0.0001
    max_step_penalty = 0.01
    max_distance_penalty_coef = 0.1
    max_loop_penalty = 1.0
    step_penalties = np.geomspace(min_step_penalty, max_step_penalty, n_reward_steps)
    distance_penalties = np.linspace(0, max_distance_penalty_coef, n_reward_steps)
    loop_penalties = np.linspace(0, max_loop_penalty, n_reward_steps)
    results = []
    for i in tqdm(range(n_reward_steps), desc='Reward Sweep'):
        config = copy.deepcopy(base_config)
        config['seed'] = seed
        config['env']['goal_pos'] = target
        config['trainer']['enable_live_tracking'] = False
        config['reward']['args'] = {
            'success_reward': base_config['reward']['args']['success_reward'],
            'step_penalty': step_penalties[i],
            'distance_penalty_coef': distance_penalties[i],
            'loop_penalty': loop_penalties[i]
        }
        results.append(run_trial(config))
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Reward sweep results saved to {output_csv}")

# --- Experiment 2: Target/Seed Sweep ---
def experiment_target_seed_sweep(base_config_path='datachallengeg15/config.yaml', n_targets=5, n_seeds=10, output_csv='hpo_ppo_target_seed.csv'):
    base_config = read_config(base_config_path)
    # Use the densest reward config (last in sweep)
    reward_args = {
        'success_reward': base_config['reward']['args']['success_reward'],
        'step_penalty': 0.01,
        'distance_penalty_coef': 0.1,
        'loop_penalty': 1.0
    }
    # Targets from the original HPOConfig
    targets = np.array([[9.5, 27.5], [7.5, 17.5], [16.5, 14], [15.5, 35.5], [16.5, 1.5]])
    results = []
    for target in tqdm(targets[:n_targets], desc='Target Loop'):
        for seed in range(1, n_seeds + 1):
            config = copy.deepcopy(base_config)
            config['seed'] = seed
            config['env']['goal_pos'] = target.tolist()
            config['trainer']['enable_live_tracking'] = False
            config['reward']['args'] = reward_args
            results.append(run_trial(config))
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Target/seed sweep results saved to {output_csv}")

# --- CLI Entrypoint ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run HPO experiments.")
    parser.add_argument('-e', '--exp', type=str, choices=['reward', 'target'], required=False, help='Which experiment to run: reward or target')
    parser.add_argument('-t', '--test', action='store_true', help='Run both experiments in test mode (2 runs each)') 
    args = parser.parse_args()
    if args.test:
        print("Running both experiments in test mode (2 runs each)...")
        experiment_reward_sweep(n_reward_steps=2, output_csv='hpo_reward_sweep_test.csv')
        experiment_target_seed_sweep(n_targets=2, n_seeds=1, output_csv='hpo_target_seed_sweep_test.csv')
    elif args.exp == 'reward':
        experiment_reward_sweep()
    elif args.exp == 'target':
        experiment_target_seed_sweep()

if __name__ == "__main__":
    main()
