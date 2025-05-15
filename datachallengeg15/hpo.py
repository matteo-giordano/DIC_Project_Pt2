from itertools import product
import yaml
import pandas as pd
from tqdm import tqdm
from train import Trainer
from grid import Grid
import numpy as np
from agent import TabularQLearningAgent, MonteCarloAgent, ValueIterationAgent
from reward import reward_fn, reward_dont_revisit
import concurrent.futures
import random
from datetime import datetime
import os

class HPO:
    def __init__(self, cfg_path: str):
        self.cfg = yaml.safe_load(open(cfg_path))
        self.algorithm = self.cfg["algorithm"]
        self.map = self.load_map()
        self.map_name = self.cfg["map"].split("/")[-1].split(".")[0]
        self.reward_fn = self.cfg["reward_fn"]
        self.early_stopping_threshold = self.cfg["early_stopping_threshold"]
        self.results = []
        self.max_workers = self.cfg.get("max_workers", 10)
        self.n_trials = self.cfg.get("n_trials", 20)
        self.n_seeds = self.cfg.get("n_seeds", 1)
        self.sigma = self.cfg.get("sigma", [0, 0.1, 1])

    def run_experiment(self):
        # Generate random parameter combinations based on min/max values in config
        param_combinations = self._generate_random_params(self.n_trials)
        
        # Create tasks for each parameter combination and seed
        sigmas = np.linspace(self.sigma[0], self.sigma[1], self.sigma[2])
        tasks = []
        for params in param_combinations:
            for seed in range(self.n_seeds):
                for sigma in sigmas:
                    # Copy params and add seed
                    params_with_seed = params.copy()
                    params_with_seed['seed'] = seed
                    params_with_seed['sigma'] = sigma
                    tasks.append(params_with_seed)
        
        # Run trials in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for params in tasks:
                futures.append(executor.submit(self.run_instance, params))
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                self.results.append(result)

        # Flatten the results for better DataFrame representation
        flattened_results = []
        for result in self.results:
            flat_result = {}
            # Add all parameters with their names
            for param_name, param_value in result["params"].items():
                flat_result[param_name] = param_value
            # Add other metrics
            flat_result["iters"] = result["iters"]
            flat_result["optimal_path_length"] = result["optimal_path_length"]
            flat_result["valid_path"] = result["valid_path"]
            flat_result["seed"] = result["seed"]
            flat_result["sigma"] = result["sigma"]
            flat_result["rewards"] = result["rewards"]
            flattened_results.append(flat_result)

        # Save results to CSV
        df = pd.DataFrame(flattened_results)
        timestamp = datetime.now().strftime('%d_%H-%M')
        save_path = f"../results/{self.algorithm}_{self.map_name}_{self.reward_fn}__{timestamp}.csv"
        print(f"Saving results to {save_path}")
        df.to_csv(save_path, index=False)
        return df

    def _generate_random_params(self, n_trials):
        param_combinations = []
        for _ in range(n_trials):
            params = {}
            for param_name, param_range in self.cfg["algorithm_params"].items():
                # Get min and max values from the range provided in config
                min_val, max_val = min(param_range), max(param_range)
                
                # Generate random value based on parameter type
                if isinstance(min_val, int):
                    params[param_name] = random.randint(min_val, max_val)
                elif isinstance(min_val, float):
                    params[param_name] = random.uniform(min_val, max_val)
                else:
                    # For non-numeric parameters, choose randomly from the list
                    params[param_name] = random.choice(param_range)
            
            param_combinations.append(params)
        return param_combinations

    def run_instance(self, params):
        # Extract seed from params for reproducibility
        seed = params.pop('seed', 0)  # Remove seed from params to avoid passing it to the agent
        sigma = params.pop('sigma', 0.0)
        np.random.seed(seed)
        random.seed(seed)
        
        # Create a copy of params to avoid modifying the original
        agent_params = params.copy()

    # Convert gamma and theta to float if present
        if 'gamma' in agent_params:
            agent_params['gamma'] = float(agent_params['gamma'])
        if 'theta' in agent_params:
            agent_params['theta'] = float(agent_params['theta'])    
                
        agent_cls = eval(self.algorithm)
        trainer = Trainer(eval(self.algorithm), eval(self.reward_fn), agent_kwargs=agent_params, early_stopping_threshold=self.early_stopping_threshold)
        iters, _ = trainer.train_on_map(self.map, 10_000, 10_000, sigma=sigma)
        optimal_path = (trainer.agent.extract_policy_path(self.map.start_cell, self.map.target_cell))
        valid_path = self.validate_path(optimal_path)
        optimal_path_length = len(optimal_path)
        rewards = trainer.evaluate_on_map(self.map, 100, sigma=sigma)
        
        
        return {
            "params": params,
            "iters": iters,
            "rewards": rewards,
            "optimal_path_length": optimal_path_length,
            "valid_path": valid_path,
            "seed": seed,
            "sigma": sigma
        }
    
    def validate_path(self, path):
        valid = True
        valid = len(path) > 0
        valid = valid and path[0] == self.map.start_cell
        valid = valid and path[-1] == self.map.target_cell
        valid = valid and len(path) == len(set(path))
        return valid
    
    def load_map(self):
        grid_path = os.path.join(base_dir, "grid_configs", self.cfg["map"])
        arr = np.load(grid_path)
        arr[1, 49] = 3 # Target cell. 
        # Note that the position of target cell is [row, col], while the start cell is [col, rol]
        # The environment will finally deal with and show positions in [col, row], like (x, y)
        # return Grid(array=np.load(self.cfg["map"]), start_cell=tuple(self.cfg["start_cell"]))
        return Grid(array=arr, start_cell=tuple(self.cfg["start_cell"]))


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__) # Absolute path of main.py
    grid_path = os.path.join(base_dir, "ql-hpo.yaml")
    # hpo = HPO("ql-hpo.yaml")
    hpo = HPO(grid_path)
    hpo.run_experiment()