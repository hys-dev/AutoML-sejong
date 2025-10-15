class NasStartConfig:
    def __init__(self, data):
        self.data = data
        self.dataset_name = data['dataset_name']
        self.max_epochs = data['max_epochs']
        self.exp_key = ''
        self.gpu_id = 0

    def set_exp_key(self, key):
        self.exp_key = key

    def set_gpu_id(self, gpu_id):
        self.gpu_id = gpu_id


class NasSearchStartConfig(NasStartConfig):
    def __init__(self, data):
        super(NasSearchStartConfig, self).__init__(data)
        self.strategy = data.get('strategy', 'darts')
        self.batch_size = data.get('batch_size', 64)
        self.learning_rate = data.get('learning_rate', 0.025)
        self.momentum = data.get('momentum', 0.9)
        self.weight_decay = data.get('weight_decay', 3e-4)
        self.auxiliary_loss_weight = data.get('auxiliary_loss_weight', 0.)
        self.gradient_clip_val = data.get('gradient_clip_val', 5.)
        self.width = data.get('width', 16)
        self.num_cells = data.get('num_cells', 8)
        self.layer_operations = data.get('layer_operations', [])

    def keys(self):
        return ['dataset_name', 'max_epochs', 'batch_size', 'learning_rate', 'momentum', 'weight_decay',
                'auxiliary_loss_weight', 'gradient_clip_val', 'width', 'num_cells', 'layer_operations']

    def __getitem__(self, item):
        return getattr(self, item)


class NasRetrainStartConfig(NasStartConfig):
    def __init__(self, data):
        super(NasRetrainStartConfig, self).__init__(data)
        self.arch = data['arch']
        self.batch_size = data.get('batch_size', 64)

        self.width = data.get('width', 36)
        self.num_cells = data.get('num_cells', 20)
        self.learning_rate = data.get('learning_rate', 0.025)
        self.momentum = data.get('momentum', 0.9)
        self.weight_decay = data.get('weight_decay', 3e-4)
        self.auxiliary_loss_weight = data.get('auxiliary_loss_weight', 0.4)
        self.drop_path_prob = data.get('drop_path_prob', 0.2)

    def keys(self):
        return ['dataset_name', 'max_epochs', 'arch', 'width', 'num_cells', 'learning_rate',
                'momentum', 'weight_decay', 'auxiliary_loss_weight', 'drop_path_prob']

    def __getitem__(self, item):
        return getattr(self, item)


class HpoStartConfig:
    def __init__(self, data):
        self.dataset_name = data['dataset_name']
        self.tuner = data['tuner']
        self.trial_no = data['trial_number']
        self.arch = data['arch']
        self.search_space = data['search_space']
        self.exp_key = ''
        self.gpu_id = 0

    def set_exp_key(self, key):
        self.exp_key = key

    def set_gpu_id(self, gpu_id):
        self.gpu_id = gpu_id

    def keys(self):
        return ['dataset_name', 'tuner', 'trial_no', 'arch', 'exp_key', 'gpu_id', 'search_space']

    def __getitem__(self, item):
        return getattr(self, item)
