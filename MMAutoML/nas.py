import os
import argparse
import sys

import torch
from lib import utils
from engine.supernet_train import get_args_parser, nas_search_experiment
from lib.utils import get_random_string
from lib.config import NAS_SEARCH_METRIC_LOG, NasSearchStartConfig


# nas.py
def main():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    utils.init_distributed_mode(args)
    """
    dataset_name = "MMIMDB"
    max_epochs = 500
    learning_rate = 3e-5
    min_learning_rate = 1e-6
    warmup_epochs = 10
    batch_size = 16
    weight_decay = 0.05
    optimizer = "adamw"
    lr_scheduler = "cosine"
    """
    dataset_name = sys.argv[1]
    max_epochs = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    min_learning_rate = float(sys.argv[4])
    warmup_epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    weight_decay = float(sys.argv[7])
    optimizer = sys.argv[8]
    lr_scheduler = sys.argv[9]
    print(sys.argv)

    hyperparameters = {
        "dataset_name": dataset_name,
        'max_epochs': max_epochs, 'learning_rate': learning_rate,
        'min_learning_rate': min_learning_rate, 'warmup_epochs': warmup_epochs,
        'batch_size': batch_size, 'weight_decay': weight_decay,
        'optimizer': optimizer, 'lr_scheduler': lr_scheduler,
    }

    hyperparameters.update({'mode': 'super'})
    config = NasSearchStartConfig(hyperparameters)

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
    print(f"[EXP_KEY]{exp_key}")

    for key in config.keys():
        setattr(args, key, config[key])

    output_dir = os.path.join(NAS_SEARCH_METRIC_LOG, config.exp_key)
    setattr(args, 'output_dir', output_dir)
    setattr(args, 'resume', './checkpoint/supernet-tiny.pth')

    nas_search_exp_key = nas_search_experiment(args)
    print(f"[NAS_SEARCH_EXP_KEY]{nas_search_exp_key}")


if __name__ == '__main__':
    main()
