import yaml
import argparse
import random
import numpy as np
import torch
from dqn_trainer import DQNTrainer

def read_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-t', '--test', action='store_true', help='Run in test mode')
    return parser.parse_args()

def main():
    args = parse_args()
    config = read_config(args.config)
    set_random_seeds(config['seed'])
    trainer = DQNTrainer(config)
    if args.test:
        trainer.test(config['trainer']['test_episodes'], config['trainer']['max_steps'])
    else:
        trainer.train()
        trainer.test(config['trainer']['test_episodes'], config['trainer']['max_steps'])

if __name__ == '__main__':
    main()