import os
import json
import time

from common.config import HPO_CACHE_PATH
from common.utils import get_random_string, GPUManager, save_to_json
from engine.experiment import Experiment
from engine.hpo.experiment import hpo_experiment
from engine.log.hpo import HPOLog
from lib.config import HpoStartConfig
from lib.lib import CustomJSONEncoder


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
    search_space = {
        'max_epochs': {'_type': 'randint', '_value': [1, 2]},
        'batch_size': {'_type': 'choice', '_value': [32, 64]},
        'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'momentum': {'_type': 'choice', '_value': [0.8, 0.9, 0.95]},
        'weight_decay': {'_type': 'loguniform', '_value': [0.0001, 0.001]},
        'auxiliary_loss_weight': {'_type': 'choice', '_value': [0.1 * i for i in range(1, 10)]},
        'width': {'_type': 'choice', '_value': [16, 20, 28, 32]},
        'num_cells': {'_type': 'choice', '_value': [4, 5, 6, 7, 8]},
        'drop_path_prob': {'_type': 'choice', '_value': [0.1, 0.2, 0.3, 0.4]}
    }
    dataset_name = 'cifar10'
    tuner = 'Random'
    trial_number = 2

    hyperparameters = {
        'dataset_name': dataset_name, 'tuner': tuner,
        'trial_number': trial_number, 'arch': arch,
        'search_space': search_space,
    }

    config = HpoStartConfig(hyperparameters)

    exp_key = get_random_string()
    gpu_id = GPUManager.get_available_gpu_id()
    config.set_exp_key(exp_key)
    config.set_gpu_id(gpu_id)
    print(exp_key)

    cache = {exp_key: Experiment('hpo', 'running', config)}
    json_str = json.dumps(cache, cls=CustomJSONEncoder)
    json_dict = json.loads(json_str)
    if not os.path.exists(HPO_CACHE_PATH):
        os.mkdir(HPO_CACHE_PATH)
    save_to_json(os.path.join(HPO_CACHE_PATH, 'cache.json'), json_dict)

    experiment = hpo_experiment(config)
    log = HPOLog(experiment)

    hpo_log_path = os.path.join(HPO_CACHE_PATH, exp_key)
    if not os.path.exists(hpo_log_path):
        os.mkdir(hpo_log_path)

    while log.get_status() != 'DONE':
        time.sleep(10)
        result = {"job_status": log.get_job_statistic(),
                  "max_trial_num": log.get_max_trial_num(),
                  "start_time": log.get_start_time(),
                  "jobs": log.get_log()}
        print(result)
        result_str = json.dumps(result)
        with open(os.path.join(hpo_log_path, 'log.txt'), 'a') as f:
            f.write(result_str + '\n')
    print("FINISHED")


if __name__ == '__main__':
    main()
