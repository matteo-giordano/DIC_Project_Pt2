from dataclasses import dataclass, field
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import copy
import pandas as pd

# Import necessary components from other files
from main import read_config, set_random_seeds
from trainer import PPOTrainer


def run_trial(config: dict) -> dict:
    """
    Executes a single training trial based on a given complete configuration.
    This function is designed to be called by a multiprocessing Pool.
    """
    # Set seeds for reproducibility
    set_random_seeds(config['seed'])
    
    # Initialize and run the trainer
    try:
        trainer = PPOTrainer(config)
        _, episode_rewards, _ = trainer.train()
        
        # Return results, logging only the parts of the config that changed
        return {
            'status': 'success',
            'config': {
                'reward': config['reward']['args'],
                'seed': config['seed'],
                'target': config['env']['goal_pos']
            },
            'final_reward': np.mean(episode_rewards[-10:]) if episode_rewards else 0
        }
    except Exception as e:
        return {
            'status': 'failure',
            'config': {
                'reward': config['reward']['args'],
                'seed': config['seed'],
                'target': config['env']['goal_pos']
            },
            'error': str(e)
        }


@dataclass
class HPOConfig:
    n_seeds: int = 10
    entropy_coef: np.ndarray = field(default_factory=lambda: np.linspace(0.001, 0.03, 20))
    targets: np.ndarray = field(default_factory=lambda: np.array([[9.5, 27.5], [7.5, 17.5], [16.5, 14], [15.5, 35.5], [16.5, 1.5]]))
    n_reward_steps: int = 20
    
    # Parameter ranges for reward components
    min_step_penalty: float = 0.0001  # Very small penalty
    max_step_penalty: float = 0.01    # Significant penalty per step
    max_distance_penalty_coef: float = 0.1
    max_loop_penalty: float = 1.0
    
    base_config_path: str = 'datachallengeg15/config.yaml'

    def __post_init__(self):
        """Load the base configuration after the dataclass is initialized."""
        self.base_config = read_config(self.base_config_path)

    def generate_reward_configs(self) -> List[Dict]:
        """Generate reward configurations exploring different reward components."""
        configs = []
        
        # Generate exponential spacing for step penalty (better for exploring different scales)
        step_penalties = np.geomspace(self.min_step_penalty, self.max_step_penalty, self.n_reward_steps)
        
        # Generate linear spacing for dense reward parameters
        distance_penalties = np.linspace(0, self.max_distance_penalty_coef, self.n_reward_steps)
        loop_penalties = np.linspace(0, self.max_loop_penalty, self.n_reward_steps)
        
        # Create configurations with different combinations
        for i in range(self.n_reward_steps):
            # Base configuration (sparse)
            config = {
                "success_reward": self.base_config['reward']['args']['success_reward'],
                "step_penalty": step_penalties[i],
                "distance_penalty_coef": distance_penalties[i],
                "loop_penalty": loop_penalties[i]
            }
            configs.append(config)
            
            # Add a variation with higher step penalty but lower dense rewards
            if i > 0:  # Skip first iteration to keep at least one purely sparse config
                variation = {
                    "success_reward": self.base_config['reward']['args']['success_reward'],
                    "step_penalty": step_penalties[-(i+1)],  # Use penalties from the other end
                    "distance_penalty_coef": distance_penalties[i] / 2,  # Reduced dense rewards
                    "loop_penalty": loop_penalties[i] / 2
                }
                configs.append(variation)
        
        return configs

    def generate_hpo_configs(self) -> List[Dict]:
        """Generates the full set of HPO configurations by overriding the base config."""
        full_configs = []
        reward_configs = self.generate_reward_configs()

        for reward_cfg in reward_configs:
            for seed in range(1, self.n_seeds + 1):
                for target in self.targets:
                    # Create a deep copy to avoid modifying the base config object
                    config = copy.deepcopy(self.base_config)
                    
                    # Override settings
                    config['seed'] = seed
                    config['reward']['args'] = reward_cfg
                    config['env']['goal_pos'] = target.tolist()
                    
                    # Disable live tracking for parallel runs to avoid display conflicts
                    config['trainer']['enable_live_tracking'] = False
                    
                    full_configs.append(config)
        
        return full_configs


class HPORunner:
    def __init__(self, configs: List[Dict]):
        self.configs = configs

    def run(self, test: bool = False):
        """
        Runs all HPO experiments sequentially.
        If test is True, only runs a small subset of experiments.
        """
        configs_to_run = self.configs
        if test:
            print("--- Running in TEST mode: only 2 experiments will be run. ---")
            configs_to_run = self.configs[:2]

        results = []
        for config in tqdm(configs_to_run, desc="Running HPO Trials"):
            result = run_trial(config)
            results.append(result)
        
        return results


def main():
    hpo_config = HPOConfig()
    configs = hpo_config.generate_hpo_configs()
    print(f"Generated {len(configs)} configurations")

    runner = HPORunner(configs=configs)

    # Add a command-line argument for test mode
    import argparse
    parser = argparse.ArgumentParser(description="Run HPO experiments.")
    parser.add_argument('-t', '--test', action='store_true', help="Run in test mode (only 4 experiments).")
    args = parser.parse_args()

    results = runner.run(test=args.test)

    print("\n--- HPO Run Finished ---")
    print(f"Total results collected: {len(results)}")
    
    if not results:
        print("No results to save.")
        return

    # Flatten and save results to CSV
    flattened_data = []
    for r in results:
        flat_dict = {
            'status': r.get('status'),
            'final_reward': r.get('final_reward'),
            'error': r.get('error')
        }
        
        config_part = r.get('config', {})
        # Merge the reward dictionary keys as top-level columns
        flat_dict.update(config_part.get('reward', {}))
        flat_dict['seed'] = config_part.get('seed')
        
        target = config_part.get('target', [None, None])
        if target:
            flat_dict['target_x'] = target[0]
            flat_dict['target_y'] = target[1]
        
        flattened_data.append(flat_dict)

    df = pd.DataFrame(flattened_data)
    results_path = "hpo_results.csv"
    # Ensure reward columns are float for proper sorting later
    for col in df.columns:
        if 'reward' in col or 'penalty' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Find and print the best trial from the DataFrame
    successful_df = df[df['status'] == 'success'].copy()
    if not successful_df.empty:
        best_trial = successful_df.loc[successful_df['final_reward'].idxmax()]
        
        print("\n--- Best Performing Trial ---")
        print(f"Final Reward: {best_trial['final_reward']:.4f}")
        print("Configuration (from CSV):")
        print(best_trial.drop(['status', 'error']).to_string())


if __name__ == "__main__":
    main()
