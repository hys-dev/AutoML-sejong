from easydict import EasyDict as edict
import yaml
import os

NAS_SEARCH_METRIC_LOG = './storage/nas/search'
EVOLUTION_SEARCH_METRIC_LOG = './storage/nas/evolution'
NAS_RETRAIN_METRIC_LOG = './storage/nas/retrain'
HPO_CACHE_PATH = './storage/hpo'

cfg = edict()

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return

def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, edict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            base_cfg[k] = v
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)

def decode_cand_tuple(cand_tuple):
    depth = cand_tuple[0]
    return depth, list(cand_tuple[1:depth+1]), list(cand_tuple[depth + 1: 2 * depth + 1]), cand_tuple[-1]

def update_retrain_config(arch, yaml_path):
    if not os.path.exists(yaml_path):
        print(f"Error: YAML file not found at {yaml_path}")
        return

    print(f"Updating YAML config at: {yaml_path}")

    depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(arch)

    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)

    if 'RETRAIN' not in config_data:
        config_data['RETRAIN'] = {}

    config_data['RETRAIN']['MLP_RATIO'] = mlp_ratio
    config_data['RETRAIN']['NUM_HEADS'] = num_heads
    config_data['RETRAIN']['DEPTH'] = depth
    config_data['RETRAIN']['EMBED_DIM'] = embed_dim

    with open(yaml_path, 'w') as f:
        yaml.dump(config_data, f, indent=2, sort_keys=False)

    print("Retrain YAML config updated successfully with the new architecture!")
    print(f"  - DEPTH: {depth}")
    print(f"  - MLP_RATIO: {mlp_ratio}")
    print(f"  - NUM_HEADS: {num_heads}")
    print(f"  - EMBED_DIM: {embed_dim}")


class NasStartConfig:
    def __init__(self, data):
        self.data = data
        self.dataset_name = data['dataset_name']
        self.max_epochs = data['max_epochs']
        self.mode = data['mode']
        self.exp_key = ''

    def set_exp_key(self, key):
        self.exp_key = key


class NasSearchStartConfig(NasStartConfig):
    def __init__(self, data):
        super(NasSearchStartConfig, self).__init__(data)
        self.batch_size = data.get('batch_size', 16)
        self.learning_rate = data.get('learning_rate', 3e-5)
        self.min_learning_rate = data.get('min_learning_rate', 1e-6)
        self.weight_decay = data.get('weight_decay', 0.05)
        self.warmup_epochs = data.get('warmup_epochs', 10)
        self.optimizer = data.get('optimizer', 'adamw')
        self.lr_scheduler = data.get('lr_scheduler', 'cosine')

    def keys(self):
        return ['dataset_name', 'max_epochs', 'batch_size', 'mode', 'learning_rate', 'weight_decay',
                'min_learning_rate', 'warmup_epochs', 'optimizer', 'lr_scheduler']

    def __getitem__(self, item):
        return getattr(self, item)

class EvolutionSearchStartConfig(NasStartConfig):
    def __init__(self, data):
        super(EvolutionSearchStartConfig, self).__init__(data)
        self.batch_size = data.get('batch_size', 64)
        self.min_param_limits = data.get('min_param_limits', 10)
        self.param_limits = data.get('param_limits', 20)
        self.select_num = data.get('select_num', 10)
        self.population_num = data.get('population_num', 50)
        self.crossover_num = data.get('crossover_num', 25)
        self.mutation_num = data.get('mutation_num', 25)

    def keys(self):
        return ['dataset_name', 'max_epochs', 'batch_size', 'mode', 'min_param_limits', 'param_limits',
                'select_num', 'population_num', 'crossover_num', 'mutation_num']

    def __getitem__(self, item):
        return getattr(self, item)

class NasRetrainStartConfig(NasStartConfig):
    def __init__(self, data):
        super(NasRetrainStartConfig, self).__init__(data)
        self.batch_size = data.get('batch_size', 16)
        self.learning_rate = data.get('learning_rate', 3e-5)
        self.min_learning_rate = data.get('min_learning_rate', 1e-6)
        self.weight_decay = data.get('weight_decay', 0.05)
        self.warmup_epochs = data.get('warmup_epochs', 10)
        self.optimizer = data.get('optimizer', 'adamw')
        self.lr_scheduler = data.get('lr_scheduler', 'cosine')

    def keys(self):
        return ['dataset_name', 'max_epochs', 'batch_size', 'mode', 'learning_rate', 'weight_decay',
                'min_learning_rate', 'warmup_epochs', 'optimizer', 'lr_scheduler']

    def __getitem__(self, item):
        return getattr(self, item)


class HpoStartConfig:
    def __init__(self, data):
        self.dataset_name = data['dataset_name']
        self.tuner = data['tuner']
        self.trial_no = data['trial_number']
        self.search_space = data['search_space']
        self.mode = data['mode']
        self.exp_key = ''

    def set_exp_key(self, key):
        self.exp_key = key

    def keys(self):
        return ['dataset_name', 'tuner', 'trial_no', 'mode', 'exp_key', 'search_space']

    def __getitem__(self, item):
        return getattr(self, item)
