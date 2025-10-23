import os
import argparse
import torch

from lib import utils
from engine.supernet_train import get_args_parser, nas_search_experiment
from lib.utils import get_random_string
from lib.config import NAS_RETRAIN_METRIC_LOG, NasRetrainStartConfig, update_retrain_config


def main():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    dataset_name = "MMIMDB"

    arch = (12,
            3.5, 4, 4, 4, 3.5, 3.5, 4, 3.5, 4, 3.5, 3.5, 3.5,
            4, 4, 4, 3, 3, 3, 4, 3, 4, 3, 4, 4,
            192)
    max_epochs = 500
    learning_rate = 3e-5
    min_learning_rate = 1e-6
    warmup_epochs = 10
    batch_size = 16
    weight_decay = 0.05
    optimizer = "adamw"
    lr_scheduler = "cosine"

    hyperparameters = {
        "dataset_name": dataset_name,
        'max_epochs': max_epochs, 'learning_rate': learning_rate,
        'min_learning_rate': min_learning_rate, 'warmup_epochs': warmup_epochs,
        'batch_size': batch_size, 'weight_decay': weight_decay,
        'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
    }

    hyperparameters.update({'mode': 'retrain'})
    config = NasRetrainStartConfig(hyperparameters)

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

    retrain_yaml_path = './experiments/subnet/AutoFormer-T.yaml'

    output_dir = os.path.join(NAS_RETRAIN_METRIC_LOG, config.exp_key)
    setattr(args, 'output_dir', output_dir)
    setattr(args, 'cfg', retrain_yaml_path)
    setattr(args, 'resume', './checkpoint/supernet-tiny.pth')

    if utils.is_main_process():
        update_retrain_config(arch, retrain_yaml_path)

    nas_search_experiment(args)


if __name__ == '__main__':
    main()
