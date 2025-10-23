import math
import sys
from typing import Iterable, Optional

import numpy as np
from timm.utils.model import unwrap_model
import torch

from sklearn.metrics import f1_score
from lib.Mixup import Mixup
from timm.utils import accuracy, ModelEma

from torch import distributed as dist
from lib import utils
import random
import time

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, (tuple, list)):
        return type(obj)(move_to_device(x, device) for x in obj)
    else:
        return obj

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,
                    csv_writer=None, metrics_file=None, args=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train_f1_score', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    pred_list, tgt_list = [], []
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = move_to_device(samples, device)
        targets = targets.to(device, non_blocking=True)

        text, segment, img = samples

        origin_targets = targets.clone()

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            img, targets = mixup_fn(img, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(text, segment, img)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(text, segment, img)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(text, segment, img)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(text, segment, img)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(text, segment, img)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        batch_size = img.shape[0]

        if args.task_type == "multilabel":
            pred = torch.sigmoid(outputs).cpu().detach().numpy()
            tgt = origin_targets.cpu().detach().numpy()

            pred_list.append(pred)
            tgt_list.append(tgt)

            tgts = np.vstack(tgt_list)
            preds = np.vstack(pred_list)

            if args.distributed and args.dist_eval:
                gather_preds = [None for _ in range(dist.get_world_size())]
                gather_tgts = [None for _ in range(dist.get_world_size())]

                dist.all_gather_object(gather_preds, preds)
                dist.all_gather_object(gather_tgts, tgts)

                preds_global = np.vstack(gather_preds)
                tgts_global = np.vstack(gather_tgts)
            else:
                preds_global = preds
                tgts_global = tgts

            search_space = np.linspace(0.05, 0.95, 19)
            best_thr, best_f1 = 0.0, 0.0
            for threshold in search_space:
                f1 = f1_score(tgts_global, preds_global > threshold, average='weighted')
                if f1 > best_f1:
                    best_f1, best_thr = f1, threshold

            global_f1 = f1_score(tgts_global, preds_global > best_thr, average='weighted')
            metric_logger.meters['train_f1_score'].update(global_f1.item()*100)

        else :
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


        if csv_writer is not None and utils.is_main_process():
            step = epoch * len(data_loader) + i
            csv_writer.writerow([loss_value, global_f1.item()*100, epoch,  step, "", "", "", ""])
            metrics_file.flush()

        i = i + 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, args, amp=True, choices=None, mode='super', retrain_config=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if args.task_type == "multilabel":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    preds, tgts = [], []
    for samples, target in metric_logger.log_every(data_loader, 10, header):
        samples = move_to_device(samples, device)
        target = target.to(device, non_blocking=True)

        text, segment, img = samples
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(text, segment, img)
                loss = criterion(output, target)
        else:
            output = model(text, segment, img)
            loss = criterion(output, target)

        batch_size = img.shape[0]
        metric_logger.update(loss=loss.item())

        if args.task_type == "multilabel":
            pred = torch.sigmoid(output).cpu().detach().numpy()
            tgt = target.cpu().detach().numpy()

            preds.append(pred)
            tgts.append(tgt)
        else :
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)

        if args.distributed and args.dist_eval:
            gather_preds = [None for _ in range(dist.get_world_size())]
            gather_tgts = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(gather_preds, preds)
            dist.all_gather_object(gather_tgts, tgts)

            preds_global = np.vstack(gather_preds)
            tgts_global = np.vstack(gather_tgts)
        else:
            preds_global = preds
            tgts_global = tgts

        search_space = np.linspace(0.05, 0.95, 19)
        best_thr, best_f1 = 0.0, 0.0
        for threshold in search_space:
            f1 = f1_score(tgts_global, preds_global > threshold, average='weighted')
            if f1 > best_f1:
                best_f1, best_thr = f1, threshold

        global_f1 = f1_score(tgts_global, preds_global > best_thr, average='weighted')
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    if args.task_type == "multilabel":
        print('* F1@weighted {f1_weighted:.3f} loss {losses.global_avg:.3f}'
              .format(f1_weighted=global_f1.item()*100, losses=metric_logger.loss))
    else :
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.task_type == "multilabel":
        stats["f1_score"] = global_f1.item()*100
    return stats
