import sys
import nni
from nni.nas import fixed_arch
from nni.retiarii.evaluator.pytorch import DataLoader
from nni.retiarii.evaluator.pytorch import Lightning, Trainer

from common.config import FAST_DEV_RUN, LOG_EVERY_N_STEPS
from common.utils import load_from_json
from engine.dataset import build_dataset
from engine.nas.nas.nasnet import DARTS as DartsSpace
from engine.hpo.classification import DartsClassificationModule

exp_key = sys.argv[1]

# searchable parameter
# 1. max_epochs
# 2. learning_rate
# 3. batch_size
# 4. momentum
# 5. weight_decay
# 6. auxiliary_loss_weight
# 7. width
# 8. num_cells
# 9. drop_path_prob


params = {
    'max_epochs': 3,
    'learning_rate': 0.025,
    'batch_size': 64,
    'momentum': 0.9,
    'weight_decay': 3e-4,
    'auxiliary_loss_weight': 0.4,
    'width': 16,
    'num_cells': 8,
    'drop_path_prob': 0.2
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

cache_file = './storage/hpo/cache.json'


def load_cache(filename):
    data = load_from_json(filename)
    return data


cache = load_cache(cache_file)
config = cache[exp_key]['config']
dataset_name = config['dataset_name']
gpu_id = config['gpu_id']
arch = config['arch']

train_dataset, valid_dataset = build_dataset(dataset_name)

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=0, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], num_workers=0)

num_labels = len(train_dataset.classes)

with fixed_arch(arch):
    final_model = DartsSpace(num_labels=num_labels,
                             width=params['width'],
                             num_cells=params['num_cells'],
                             dataset=dataset_name,
                             auxiliary_loss=True,
                             drop_path_prob=params['drop_path_prob'])

gpus = [gpu_id, ] if gpu_id is not None else gpu_id

trainer = Trainer(
    gpus=gpus,
    max_epochs=params['max_epochs'],
    fast_dev_run=FAST_DEV_RUN,
    log_every_n_steps=LOG_EVERY_N_STEPS
)

evaluator = Lightning(
    DartsClassificationModule(params['learning_rate'],
                              params['momentum'],
                              params['weight_decay'],
                              params['auxiliary_loss_weight'],
                              params['max_epochs'],
                              cache, exp_key),
    trainer=trainer,
    train_dataloaders=train_loader,
    val_dataloaders=valid_loader
)

evaluator.fit(final_model)
