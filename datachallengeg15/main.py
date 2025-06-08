import yaml
import argparse
from trainer import PPOTrainer
import random
import numpy as np
import torch


def read_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description='Train or test PPO agent')
    parser.add_argument('-c', '--config', type=str, default='datachallengeg15/config.yaml', help='Path to the config file')
    parser.add_argument('-t', '--test', action='store_true', help='Test the agent')
    return parser.parse_args()


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args: dict):
    config = read_config(args.config)
    set_random_seeds(config['seed'])
    trainer = PPOTrainer(config)
    if args.test:
        trainer.test(config['test']['episodes'], config['test']['max_steps'])
    else:
        trainer.train()
        trainer.test(config['test']['episodes'], config['test']['max_steps'])


if __name__ == '__main__':
    args = parse_args()
    main(args)