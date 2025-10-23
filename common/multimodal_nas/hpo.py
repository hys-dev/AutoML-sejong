import os
import json
import time

from lib import utils
from engine.experiment import Experiment
from engine.hpo.experiment import hpo_experiment
from engine.log.hpo import HPOLog
from lib.utils import get_random_string, save_to_json
from lib.config import HPO_CACHE_PATH, HpoStartConfig, update_retrain_config
from lib.lib import CustomJSONEncoder


def main():
    dataset_name = "MMIMDB"
    arch = (12,
            3.5, 4, 4, 4, 3.5, 3.5, 4, 3.5, 4, 3.5, 3.5, 3.5,
            4, 4, 4, 3, 3, 3, 4, 3, 4, 3, 4, 4,
            192)

    tuner = 'Random'
    trial_number = 2
    search_space = {
        'max_epochs': {'_type': 'randint', '_value': [1, 2]},
        'batch_size': {'_type': 'choice', '_value': [8, 16]},
        'learning_rate': {'_type': 'loguniform', '_value': [0.00001, 0.0005]},
        'min_learning_rate': {'_type': 'loguniform', '_value': [0.000001, 0.00001]},
        'warmup_epochs': {'_type': 'choice', '_value': [5, 10, 15, 20]},
        'weight_decay': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
        'optimizer': {'_type': 'choice', '_value': ['adamw', 'sgd', 'adam', 'rmsprop']},
        'lr_scheduler': {'_type': 'choice', '_value': ['cosine', 'plateau', 'tanh', 'step']},
    }

    hyperparameters = {
        'dataset_name': dataset_name, 'tuner': tuner,
        'trial_number': trial_number,
        'search_space': search_space,
    }

    hyperparameters.update({'mode': 'retrain'})
    retrain_yaml_path = './experiments/subnet/AutoFormer-T.yaml'

    if utils.is_main_process():
        update_retrain_config(arch, retrain_yaml_path)

    config = HpoStartConfig(hyperparameters)

    exp_key = get_random_string()
    config.set_exp_key(exp_key)
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
