# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)
    # Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)# Based on https://github.com/thunlp/OpenBackdoor/blob/main/demo_attack.py

import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.trainers import load_trainer
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/syntactic_config.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main(config):
    torch.cuda.set_device(0)
    print('GPU:', torch.cuda.current_device())
    
    # initialize attacker with default parameters 
    attacker = load_attacker(config["attacker"])
    target_dataset = load_dataset(**config["target_dataset"]) 
    poison_dataset = load_dataset(**config["poison_dataset"])
    
    print('LEN DATA:', len(poison_dataset['train']), len(poison_dataset['dev']), len(poison_dataset['test']))
    attacker.poison(None, poison_dataset, 'eval')
    attacker.poison(None, poison_dataset, 'train')


if __name__=='__main__':
    args = parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    config = set_config(config)
    set_seed(args.seed)
    
    main(config)
