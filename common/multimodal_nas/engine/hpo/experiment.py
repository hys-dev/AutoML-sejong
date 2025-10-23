import os.path
import torch
from nni.experiment import Experiment
from lib.utils import PortManager


def hpo_experiment(config):
    default_search_space = {
        'max_epochs': {'_type': 'choice', '_value': [5]},
        'learning_rate': {'_type': 'choice', '_value': [0.00003]},
        'batch_size': {'_type': 'choice', '_value': [16]},
        'min_learning_rate': {'_type': 'choice', '_value': [0.00001]},
        'weight_decay': {'_type': 'choice', '_value': [0.1]},
        'warmup_epochs': {'_type': 'choice', '_value': [10]},
        'optimizer': {'_type': 'choice', '_value': ['adamw']},
        'lr_scheduler': {'_type': 'choice', '_value': ['cosine']},
    }

    search_space = {}
    for key in set(default_search_space.keys()).union(config.search_space.keys()):
        if key in config.search_space:
            search_space[key] = config.search_space[key]
        else:
            search_space[key] = default_search_space[key]

    exp_key = config.exp_key
    n_gpus = torch.cuda.device_count()

    experiment = Experiment('local')
    experiment_dir = os.path.join(experiment.config.experiment_working_directory, experiment.id)
    print('\033[92mExperiment Log Directory: ', experiment_dir + '\033[0m')

    # configure trial code
    experiment.config.trial_command = ('set PYTHONPATH=. && set EXP_KEY={} && python -m torch.distributed.launch --nproc_per_node={} '
                                   '--use_env engine/hpo/supernet_train2.py --cfg ./experiments/subnet/AutoFormer-T.yaml ' 
                                   '--resume ./checkpoint/supernet-tiny.pth').format(exp_key, n_gpus)
    experiment.config.trial_code_directory = './'

    # configure search space
    experiment.config.search_space = search_space

    # configure tuning algorithm
    experiment.config.tuner.name = config.tuner  # 'GridSearch' or 'Random' or 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    # configure how many trials to run
    experiment.config.max_trial_number = config.trial_no
    experiment.config.trial_concurrency = 1

    port = PortManager.get_free_port()

    experiment.start(port)

    return experiment
