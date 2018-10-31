from dataloaders import dataloader
from models import RefExpVanilla
import argparse
from utils import global_parameters
from src.models import config as model_config
import torch
import numpy as np
import random
import json
import logging

logger = logging.getLogger()

def init_model():
    model = RefExpVanilla(args)

def train_epochs(args, dataset, model):
    raise NotImplementedError


def main():
    ## DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    # TODO: Load dataset in memory using N workers
    dataset = dataloader.get_dataset(args, logger)

    ## MODEL
    logger.info('-' * 100)
    # TODO: Checkpoint, Pre-trained Load
    logger.info('Building model from scratch...')
    model = init_model(args, dataset)

    # Setup optimizer
    model.init_optimizer()

    # Use GPU
    if args.use_cuda:
        model.cuda()

    # Train Model
    train_epochs(args, dataset, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Multitask Multimodal Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General Arguments + File Paths + Dataset Paths
    global_parameters.add_args(parser)

    # Model Arguments
    model_config.add_args(parser)

    args = parser.parse_args()

    # Set CUDA
    if args.cuda and torch.cuda.is_available():
        vars(args)['use_cuda'] = True
    else:
        vars(args)['use_cuda'] = False

    # Set Random Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set Logging
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)

    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    main(args)