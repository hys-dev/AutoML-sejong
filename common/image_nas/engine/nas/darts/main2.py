import nni
import os
import json
import numpy as np
from nni.nas import fixed_arch
from torch.utils.data import SubsetRandomSampler
from nni.retiarii.evaluator.pytorch import DataLoader
from nni.retiarii.evaluator.pytorch import Lightning, Trainer
from nni.retiarii.strategy import DARTS as DartsStrategy
from nni.retiarii.strategy import GumbelDARTS as GumbelDartsStrategy
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig

from lib.config import NasSearchStartConfig, NasRetrainStartConfig
from engine.nas.nas.nasnet import DARTS as DartsSpace
from pytorch_lightning.loggers import CSVLogger
from engine.dataset import build_dataset
from common.config import NAS_SEARCH_METRIC_LOG, FAST_DEV_RUN, LOG_EVERY_N_STEPS, NAS_RETRAIN_METRIC_LOG
from engine.nas.darts.classification import DartsClassificationModule


def nas_search_experiment(config: NasSearchStartConfig):
    dataset_name = config.dataset_name
    max_epochs = config.max_epochs
    strategy = config.strategy
    gpu_id = config.gpu_id
    exp_key = config.exp_key
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay
    auxiliary_loss_weight = config.auxiliary_loss_weight
    gradient_clip_val = config.gradient_clip_val
    width = config.width
    num_cells = config.num_cells
    layer_operations = config.layer_operations
    train_dataset, valid_dataset = build_dataset(dataset_name)

    num_samples = len(train_dataset)
    indices = np.random.permutation(num_samples)
    split = num_samples // 2

    search_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
        sampler=SubsetRandomSampler(indices[:split])
    )

    search_valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0,
        sampler=SubsetRandomSampler(indices[split:])
    )
    csv_logger = nni.trace(CSVLogger)(NAS_SEARCH_METRIC_LOG, exp_key)

    gpus = [gpu_id, ] if gpu_id is not None else gpu_id

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        fast_dev_run=FAST_DEV_RUN,
        logger=csv_logger,
        val_check_interval=0.1,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )

    evaluator = Lightning(
        DartsClassificationModule(learning_rate, momentum, weight_decay,
                                  auxiliary_loss_weight, max_epochs, exp_key),
        trainer,
        train_dataloaders=search_train_loader,
        val_dataloaders=search_valid_loader
    )

    if strategy.lower() == 'DARTS'.lower():
        strategy = DartsStrategy(gradient_clip_val=gradient_clip_val)
    else:
        strategy = GumbelDartsStrategy(gradient_clip_val=gradient_clip_val)

    num_labels = len(train_dataset.classes)

    model_space = DartsSpace(num_labels=num_labels, width=width, num_cells=num_cells,
                             dataset=dataset_name, auxiliary_loss=auxiliary_loss_weight > 0,
                             darts_ops=layer_operations)

    retiarii_config = RetiariiExeConfig(execution_engine='oneshot')
    retiarii_config.training_service.use_active_gpu = True
    experiment = RetiariiExperiment(model_space, evaluator=evaluator, strategy=strategy)
    experiment.run(retiarii_config)

    exported_arch = experiment.export_top_models()[0]

    arch_filename = 'arch_final.json'
    arch_filepath = os.path.join(NAS_SEARCH_METRIC_LOG, exp_key, arch_filename)
    with open(arch_filepath, 'w') as f:
        json.dump(exported_arch, f)

    return exported_arch


def nas_retrain_experiment(config: NasRetrainStartConfig):
    dataset_name = config.dataset_name
    max_epochs = config.max_epochs
    gpu_id = config.gpu_id
    exp_key = config.exp_key

    arch = config.arch
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    momentum = config.momentum
    weight_decay = config.weight_decay
    auxiliary_loss_weights = config.auxiliary_loss_weight
    drop_path_prob = config.drop_path_prob
    width = config.width
    num_cells = config.num_cells

    train_dataset, valid_dataset = build_dataset(dataset_name, with_cutout=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=6, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=6)

    num_labels = len(train_dataset.classes)

    with fixed_arch(arch):
        final_model = DartsSpace(num_labels=num_labels, width=width, num_cells=num_cells, dataset=dataset_name,
                                 auxiliary_loss=auxiliary_loss_weights > 0, drop_path_prob=drop_path_prob)

    csv_logger = nni.trace(CSVLogger)(NAS_RETRAIN_METRIC_LOG, exp_key)

    gpus = [gpu_id, ] if gpu_id is not None else gpu_id

    trainer = Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        gradient_clip_val=5.,
        fast_dev_run=FAST_DEV_RUN,
        logger=csv_logger,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )

    evaluator = Lightning(
        DartsClassificationModule(learning_rate, momentum, weight_decay, auxiliary_loss_weights,
                                  max_epochs, exp_key),
        trainer=trainer,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )

    evaluator.fit(final_model)
