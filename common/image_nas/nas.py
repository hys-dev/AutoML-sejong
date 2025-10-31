from common.utils import get_random_string, GPUManager
from engine.nas.darts.main2 import nas_search_experiment
from lib.config import NasSearchStartConfig
import argparse


def main(dataset_name, layer_operations, max_epochs, strategy, batch_size, learning_rate, momentum, weight_decay, auxiliary_loss_weight, gradient_clip_val, width, num_cells):
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

    '''
    config = NasSearchStartConfig(hyperparameters)

    exp_key = get_random_string()
    gpu_id = GPUManager.get_available_gpu_id()
    config.set_exp_key(exp_key)
    config.set_gpu_id(gpu_id)
    print(exp_key)
    print(f"RANDOM_STRING:{exp_key}")

    arch = nas_search_experiment(config)
    print(arch)
    '''

    print(hyperparameters['dataset_name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, required=True)
    parser.add_argument('--strategy', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--auxiliary_loss_weight', type=float, required=True)
    parser.add_argument('--gradient_clip_val', type=float, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--num_cells', type=int, required=True)
    parser.add_argument('--layer_operations', type=str, required=True)
    args = parser.parse_args()

    main(args.dataset_name, args.layer_operations, args.max_epochs, args.strategy, args.batch_size,
         args.learning_rate, args.momentum, args.weight_decay, args.auxiliary_loss_weight,
         args.gradient_clip_val, args.width, args.num_cells)
