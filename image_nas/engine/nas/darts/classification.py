import json
import os
import re
from collections import defaultdict
import torch
from nni.retiarii.evaluator.pytorch import ClassificationModule
from common.config import NAS_SEARCH_METRIC_LOG


class DartsClassificationModule(ClassificationModule):
    def __init__(
            self,
            learning_rate: float = 0.001,
            momentum: float = 0.9,
            weight_decay: float = 0.,
            auxiliary_loss_weight: float = 0.4,
            max_epochs: int = 600,
            exp_key: str = None,
    ):
        self.momentum = momentum
        self.auxiliary_loss_weight = auxiliary_loss_weight
        # Training length will be used in LR scheduler
        self.max_epochs = max_epochs
        self.exp_key = exp_key
        super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, export_onnx=False)

    def configure_optimizers(self):
        """Customized optimizer with momentum, as well as a scheduler."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            momentum=self.momentum,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-3)
        }

    def training_step(self, batch, batch_idx):
        """Training step, customized with auxiliary loss."""
        x, y = batch
        if self.auxiliary_loss_weight:
            y_hat, y_aux = self(x)
            loss_main = self.criterion(y_hat, y)
            loss_aux = self.criterion(y_aux, y)
            self.log('train_loss_main', loss_main)
            self.log('train_loss_aux', loss_aux)
            loss = loss_main + self.auxiliary_loss_weight * loss_aux
        else:
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('train_' + name, metric(y_hat, y), prog_bar=True)

        return loss

    def on_train_epoch_start(self):
        # Set drop path probability before every epoch. This has no effect if drop path is not enabled in model.
        self.model.set_drop_path_prob(self.model.drop_path_prob * self.current_epoch / self.max_epochs)

        # Logging learning rate at the beginning of every epoch
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        arch = self.export_arch(self.model)
        arch_filename = f'arch_{self.current_epoch + 1}.json'
        arch_filepath = os.path.join(NAS_SEARCH_METRIC_LOG, self.exp_key, arch_filename)
        with open(arch_filepath, 'w') as f:
            json.dump(arch, f)

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        for name, metric in self.metrics.items():
            self.log('val_' + name, metric(y_hat, y), prog_bar=True)

        return loss

    def parse_edge_index_from_name(self, name: str):
        m = re.search(r"_arch_alpha\.(normal|reduce)/(\d+)_(\d+)$", name)
        if not m:
            raise ValueError(f"Unrecognized arch alpha name: {name}")
        cell_type = m.group(1)
        i = int(m.group(2))
        j = int(m.group(3))
        return cell_type, i, j

    def op_names_for_this_edge(self, name: str = None, model=None, fallback_ops=None):
        return self.model.op_candidates

    def input_index_for_this_edge(self, name: str, idx: int = None):
        _, _, j = self.parse_edge_index_from_name(name)
        return j

    def export_arch(self, model):

        alphas = []
        for name, param in model.named_parameters():
            if "_arch_alpha." in name:
                try:
                    cell, i, j = self.parse_edge_index_from_name(name)
                except ValueError:
                    continue
                alphas.append((cell, i, j, name, param))

        result = {}
        buckets = defaultdict(list)

        for cell, i, j, name, alpha in alphas:
            op_list = self.op_names_for_this_edge(name, model=model)
            K = len(op_list)

            logits = alpha.detach().cpu()
            logits = logits[:K]
            # score = float(F.softmax(logits, dim=-1).max().item())
            score = float(logits.max().item())
            op_idx = int(logits.argmax().item())
            op_name = op_list[op_idx] if op_idx < len(op_list) else f"op{op_idx}"

            buckets[(cell, i)].append({
                "j": j,
                "score": score,
                "op_idx": op_idx,
                "op_name": op_name,
                "name": name
            })

        for (cell, i), edges in buckets.items():
            edges_sorted = sorted(edges, key=lambda e: e["score"], reverse=True)
            topk = edges_sorted[:2]

            for rank, e in enumerate(topk):
                result[f"{cell}/op_{i}_{rank}"] = e["op_name"]
                result[f"{cell}/input_{i}_{rank}"] = int(e["j"])

        return result
