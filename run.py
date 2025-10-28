import argparse
import os
import random
import numpy as np
import torch

from src import config
from src.ESLAM import ESLAM
import wandb

from eval_utils import MAPPING_DEPTH_SOURCES

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running ESLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--force', action='store_true', help='force to run')
    args = parser.parse_args()

    mapping_depth_source = args.config.split("/")[-2]
    if mapping_depth_source.endswith('_random_nbv'):
        mapping_depth_source = mapping_depth_source.replace('_random_nbv', '')

    if mapping_depth_source not in MAPPING_DEPTH_SOURCES:
        raise ValueError(f"Invalid mapping depth source: {mapping_depth_source}, must be one of: " + ", ".join(MAPPING_DEPTH_SOURCES))

    cfg = config.load_config(args.config, 'configs/ESLAM.yaml')
    # wandb_run = wandb.init(
    #     # Set the project where this run will be logged
    #     project=cfg['project_name'],
    #     # Track hyperparameters and run metadata
    #     config={
    #         "algo": "ActiveNGF"
    #     },
    #     name="ActiveNGF"
    # )
    wandb_run = None
    cfg['mapping_depth_source'] = mapping_depth_source
    cfg['force'] = args.force
    eslam = ESLAM(cfg, args, wandb_run)
    eslam.run()

if __name__ == '__main__':
    os.environ['WANDB_DISABLED'] = "true"
    main()
