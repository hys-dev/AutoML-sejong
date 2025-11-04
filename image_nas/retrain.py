from common.utils import get_random_string, GPUManager
from engine.nas.darts.main2 import nas_retrain_experiment
from lib.config import NasRetrainStartConfig


def main():
    arch = {
        'normal/op_2_0': 'sep_conv_3x3', 'normal/input_2_0': 0, 'normal/op_2_1': 'sep_conv_3x3', 'normal/input_2_1': 1,
        'normal/op_3_0': 'sep_conv_3x3', 'normal/input_3_0': 1, 'normal/op_3_1': 'skip_connect', 'normal/input_3_1': 0,
        'normal/op_4_0': 'sep_conv_3x3', 'normal/input_4_0': 0, 'normal/op_4_1': 'max_pool_3x3', 'normal/input_4_1': 1,
        'normal/op_5_0': 'sep_conv_3x3', 'normal/input_5_0': 0, 'normal/op_5_1': 'sep_conv_3x3', 'normal/input_5_1': 1,
        'reduce/op_2_0': 'max_pool_3x3', 'reduce/input_2_0': 0, 'reduce/op_2_1': 'sep_conv_5x5', 'reduce/input_2_1': 1,
        'reduce/op_3_0': 'dil_conv_5x5', 'reduce/input_3_0': 2, 'reduce/op_3_1': 'max_pool_3x3', 'reduce/input_3_1': 0,
        'reduce/op_4_0': 'max_pool_3x3', 'reduce/input_4_0': 0, 'reduce/op_4_1': 'sep_conv_3x3', 'reduce/input_4_1': 2,
        'reduce/op_5_0': 'max_pool_3x3', 'reduce/input_5_0': 0, 'reduce/op_5_1': 'skip_connect', 'reduce/input_5_1': 2
    }

    dataset_name = 'cifar10'
    max_epochs = 900
    batch_size = 64
    width = 36
    num_cells = 20
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 0.0003
    auxiliary_loss_weight = 0.4
    drop_path_prob = 0.2

    hyperparameters = {
        'dataset_name': dataset_name, 'max_epochs': max_epochs,
        'arch': arch,
        'batch_size': batch_size, 'width': width,
        'num_cells': num_cells, 'learning_rate': learning_rate,
        'momentum': momentum, 'weight_decay': weight_decay,
        'auxiliary_loss_weight': auxiliary_loss_weight, 'drop_path_prob': drop_path_prob
    }

    config = NasRetrainStartConfig(hyperparameters)
    exp_key = get_random_string()
    gpu_id = GPUManager.get_available_gpu_id()
    config.set_exp_key(exp_key)
    config.set_gpu_id(gpu_id)
    print(exp_key)

    nas_retrain_experiment(config)


if __name__ == '__main__':
    main()
