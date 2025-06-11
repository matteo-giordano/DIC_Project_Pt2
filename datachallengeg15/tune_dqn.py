import yaml
import itertools
import copy
from dqn import train_dqn, test_dqn_agent

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)

def run_tuning(base_cfg_path, search_cfg_path, output_path='tuning_results.yaml'):
    base_cfg = load_yaml(base_cfg_path)
    search_space = load_yaml(search_cfg_path)['search_space']

    keys, values = zip(*search_space.items())
    all_combinations = list(itertools.product(*values))

    results = []

    for combo in all_combinations:
        test_cfg = copy.deepcopy(base_cfg)
        combo_dict = dict(zip(keys, combo))
        test_cfg['DQN'].update(combo_dict)

        print(f"\nTesting config: {combo_dict}")

        # Train agent and get the trained environment
        agent, env = train_dqn(config=test_cfg)
        success, avg_steps = test_dqn_agent(agent, env, episodes=5, max_steps=250, return_metrics=True)

        results.append({
            'params': combo_dict,
            'success_rate': success,
            'avg_steps': avg_steps
        })

    save_yaml(output_path, results)
    print(f"\nTuning complete. Results saved to: {output_path}")

if __name__ == "__main__":
    run_tuning(
        base_cfg_path='datachallengeg15/config_dqn.yaml',
        search_cfg_path='datachallengeg15/tune_dqn_config.yaml'
    )
