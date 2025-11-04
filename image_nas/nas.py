import json
import sys

from common.utils import get_random_string, GPUManager
from engine.nas.darts.main2 import nas_search_experiment
from lib.config import NasSearchStartConfig

def main():
    dataset_name = sys.argv[1]
    layer_operations = json.loads(sys.argv[2])
    max_epochs = sys.argv[3]
    strategy = sys.argv[4]
    batch_size = sys.argv[5]
    learning_rate = sys.argv[6]
    momentum = sys.argv[7]
    weight_decay = sys.argv[8]
    auxiliary_loss_weight = sys.argv[9]
    gradient_clip_val = sys.argv[10]
    width = sys.argv[11]
    num_cells = sys.argv[12]

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
    print(f"RANDOM_STRING:{exp_key}")

    arch = nas_search_experiment(config)
    print(arch)


if __name__ == '__main__':
    main()
