import os.path

from nni.experiment import Experiment
from common.utils import PortManager


def hpo_experiment(config):
    default_search_space = {
        'max_epochs': {'_type': 'choice', '_value': [5]},
        'learning_rate': {'_type': 'choice', '_value': [0.025]},
        'batch_size': {'_type': 'choice', '_value': [32]},
        'momentum': {'_type': 'choice', '_value': [0.9]},
        'weight_decay': {'_type': 'choice', '_value': [3e-4]},
        'auxiliary_loss_weight': {'_type': 'choice', '_value': [0.4]},
        'width': {'_type': 'choice', '_value': [16]},
        'num_cells': {'_type': 'choice', '_value': [8]},
        'drop_path_prob': {'_type': 'choice', '_value': [0.2]}
    }

    search_space = {}
    for key in set(default_search_space.keys()).union(config.search_space.keys()):
        if key in config.search_space:
            search_space[key] = config.search_space[key]
        else:
            search_space[key] = default_search_space[key]

    exp_key = config.exp_key

    experiment = Experiment('local')
    experiment_dir = os.path.join(experiment.config.experiment_working_directory, experiment.id)
    print('\033[92mExperiment Log Directory: ', experiment_dir + '\033[0m')

    # configure trial code
    experiment.config.trial_command = 'python -m engine.hpo.darts2 {}'.format(exp_key)
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
