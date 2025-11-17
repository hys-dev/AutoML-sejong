import os
import argparse
import sys

import torch

from lib import utils
from engine.evolution_multimodal import get_args_parser, evolution_search_experiment
from lib.utils import get_random_string
from lib.config import EVOLUTION_SEARCH_METRIC_LOG, EvolutionSearchStartConfig, NAS_SEARCH_METRIC_LOG, \
    update_retrain_config


# evo.py
def main():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    nas_search_exp_key = sys.argv[1]
    dataset_name = sys.argv[2]

    max_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    min_param_limits = int(sys.argv[5])
    param_limits = int(sys.argv[6])
    select_num = int(sys.argv[7])
    population_num = int(sys.argv[8])
    crossover_num = int(sys.argv[9])
    mutation_num = int(sys.argv[10])

    hyperparameters = {
        'dataset_name': dataset_name,
        'max_epochs': max_epochs, 'batch_size': batch_size,
        'min_param_limits': min_param_limits, 'param_limits': param_limits,
        'select_num': select_num, 'population_num': population_num,
        'crossover_num': crossover_num, 'mutation_num': mutation_num,
    }

    hyperparameters.update({'mode': 'retrain'})
    config = EvolutionSearchStartConfig(hyperparameters)

    if utils.is_main_process():
        exp_key = get_random_string()
        obj_to_broadcast = [exp_key]
    else:
        obj_to_broadcast = [None]

    if args.distributed:
        torch.distributed.broadcast_object_list(obj_to_broadcast, src=0)

    exp_key = obj_to_broadcast[0]
    config.set_exp_key(exp_key)
    print(exp_key)

    for key in config.keys():
        setattr(args, key, config[key])

    output_dir = os.path.join(EVOLUTION_SEARCH_METRIC_LOG, config.exp_key)
    setattr(args, 'output_dir', output_dir)

    resume_dir = os.path.join(NAS_SEARCH_METRIC_LOG, nas_search_exp_key, "best_model.pth")
    setattr(args, 'resume', resume_dir)

    retrain_yaml_path = './experiments/subnet/AutoFormer-T.yaml'

    best_arch = evolution_search_experiment(args)

    if utils.is_main_process():
        update_retrain_config(best_arch, retrain_yaml_path)


if __name__ == '__main__':
    main()
