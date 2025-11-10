import json
import sys

from common.utils import get_random_string, GPUManager
from engine.nas.darts.main2 import nas_search_experiment
from lib.config import NasSearchStartConfig

def main():
    print("start image_nas")

    dataset_name = sys.argv[1]
    layer_operations = json.loads(sys.argv[2])
    max_epochs = int(sys.argv[3])
    strategy = sys.argv[4]
    batch_size = int(sys.argv[5])
    learning_rate = float(sys.argv[6])
    momentum = float(sys.argv[7])
    weight_decay = float(sys.argv[8])
    gradient_clip_val = float(sys.argv[9])
    width = int(sys.argv[10])
    num_cells = int(sys.argv[11])
    auxiliary_loss_weight = float(sys.argv[12])

    print(sys.argv)

    """
    dataset_name = 'cifar10'
    max_epochs = 10
    strategy = 'Darts'
    batch_size = 64
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    auxiliary_loss_weight = 0.
    gradient_clip_val = 5.
    width = 16
    num_cells = 8

    layer_operations = [
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ]
    """

    hyperparameters = {
        'dataset_name': dataset_name, 'max_epochs': max_epochs,
        'strategy': strategy, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'momentum': momentum,
        'weight_decay': weight_decay, 'auxiliary_loss_weight': auxiliary_loss_weight,
        'gradient_clip_val': gradient_clip_val, 'width': width,
        'num_cells': num_cells, 'layer_operations': layer_operations,
    }

    
    config = NasSearchStartConfig(hyperparameters)

    exp_key = get_random_string()
    gpu_id = GPUManager.get_available_gpu_id()
    config.set_exp_key(exp_key)
    config.set_gpu_id(gpu_id)
    print(exp_key)
    print(f"[EXP_KEY]{exp_key}")

    arch = nas_search_experiment(config)
    print(arch)


if __name__ == '__main__':
    main()
