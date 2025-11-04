import json
from pathlib import Path
from typing import Any, Tuple
import torch
from nni.nas.strategy import ENAS as ENASStrategyBase
from nni.nas.execution.common import Model
from nni.nas.evaluator.pytorch.lightning import Lightning
import pytorch_lightning as pl


class _GreedyExportCallback(pl.Callback):
    """
    At the end of each epoch's validation phase, call the oneshot Lightning module's export(),
    and save the resulting DARTS-style dictionary into a JSON file.
    """

    def __init__(self, export_callable, out_dir: Path):
        super().__init__()
        self.export_callable = export_callable
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _do_export(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", tag: str):
        # Skip sanity check
        if getattr(trainer, "sanity_checking", False):
            return

        # Before exporting, ensure the module is in eval mode
        # and randomness (e.g., DropPath) is disabled
        was_training = pl_module.training
        try:
            pl_module.eval()
            if hasattr(pl_module, "model") and hasattr(pl_module.model, "set_drop_path_prob"):
                pl_module.model.set_drop_path_prob(0.0)
            torch.manual_seed(0)

            arch = self.export_callable()  # Same source as export_top_models()
            if arch:
                ep = int(getattr(trainer, "current_epoch", 0))
                out = self.out_dir / f"arch_{ep + 1}_bb.json"
                with out.open("w", encoding="utf-8") as f:
                    json.dump(arch, f, ensure_ascii=False, indent=2)

        finally:
            if was_training:
                pl_module.train()

    def on_validation_epoch_end(self, trainer, pl_module):
        print("on_validation_epoch_end")
        self._do_export(trainer, pl_module, tag="valend")

    def on_validation_end(self, trainer, pl_module):
        print("on_validation_end")
        self._do_export(trainer, pl_module, tag="valend")


class ENASWithEpochExport(ENASStrategyBase):
    def __init__(self, export_dir: str = "enas_epoch_exports", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._export_dir = Path(export_dir)

    def preprocess_dataloader(self, train_dataloaders, val_dataloaders) -> Tuple[Any, Any]:
        # Keep the original ENAS behavior (use ConcatLoader to merge train/val)
        from nni.nas.oneshot.pytorch.dataloader import ConcatLoader
        return ConcatLoader({
            'train': train_dataloaders,
            'val': val_dataloaders
        }), None

    def run(self, base_model: Model, applied_mutators):
        """
        Almost identical to the base class, but attaches a Callback before training.
        """
        if applied_mutators:
            raise ValueError('Mutator is not empty. Set engine to `oneshot` for one-shot strategy.')

        if not isinstance(base_model.evaluator, Lightning):
            raise TypeError('Evaluator must be a Lightning evaluator for one-shot strategy.')

        # Reuse the base class logic for attach_model / dataloader
        self.attach_model(base_model)
        evaluator: Lightning = base_model.evaluator
        if evaluator.train_dataloaders is None or evaluator.val_dataloaders is None:
            raise TypeError('Training and validation dataloaders are both required.')

        train_loader, val_loader = self.preprocess_dataloader(evaluator.train_dataloaders,
                                                              evaluator.val_dataloaders)

        # Here self.model is the EnasLightningModule (oneshot Lightning module),
        # which has the export() method (export_top_models calls this).
        if not hasattr(self.model, "export"):
            raise RuntimeError("ENAS oneshot module has no export(); cannot do greedy export per epoch.")

        #  Install the Callback: export() will be called at the end of each epoch's validation phase
        cb = _GreedyExportCallback(export_callable=self.model.export,
                                   out_dir=self._export_dir)
        # Directly append to trainer.callbacks
        evaluator.trainer.callbacks.append(cb)

        # Start training (fit will execute the Callback at each epoch's val_end)
        evaluator.trainer.fit(self.model, train_loader, val_loader)

    def export_top_models(self, top_k: int = 1):
        # Keep the same behavior as the base class: do a final export
        return super().export_top_models(top_k=top_k)
