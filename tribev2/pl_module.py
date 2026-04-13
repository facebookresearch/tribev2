# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom lightning module that wraps a pytorch model.
"""

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
from einops import rearrange
from neuralset.dataloader import SegmentData
from neuraltrain.optimizers import BaseOptimizer
from torch import nn
from torchmetrics import Metric


class BrainModule(pl.LightningModule):
    """Torch-lightning module for fMRI encoding model training."""

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optim_config: BaseOptimizer,
        metrics: dict[str, Metric],
        checkpoint_path: Path | None = None,
        config: dict[str, tp.Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.config = config

        # Optimizer
        self.optim_config = optim_config

        self.loss = loss
        self.metrics = metrics

    def forward(self, batch):
        return self.model(batch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_build_args"] = {
            "feature_dims": self.model.feature_dims,
            "n_outputs": self.model.n_outputs,
            "n_output_timesteps": self.model.n_output_timesteps,
        }

    def _run_step(
        self, batch: SegmentData, batch_idx, step_name, dataloader_idx: int = 0
    ):
        y_true = batch.data["fmri"]  # B, D, T
        y_pred = self.forward(batch)  # B, D, T
        if step_name == "val":
            y_true = y_true[:, :, self.config["data.overlap_trs_val"] :]
            y_pred = y_pred[:, :, self.config["data.overlap_trs_val"] :]
        subject_ids_flat = batch.data["subject_id"].repeat_interleave(
            y_pred.shape[2], 0
        )

        y_pred_flat = rearrange(y_pred, "b d t -> (b t) d")
        y_true_flat = rearrange(y_true, "b d t -> (b t) d")
        bad_samples = None
        if not self.config["data.stride_drop_incomplete"]:
            bad_indices = (y_true_flat == 0).all(dim=1)
            y_pred_flat = y_pred_flat[~bad_indices]
            y_true_flat = y_true_flat[~bad_indices]
            subject_ids_flat = subject_ids_flat[~bad_indices]
            # sample-level mask for retrieval metrics (B,): True where entire segment is zero
            bad_samples = (y_true == 0).all(dim=1).all(dim=1)

        loss = self.loss(y_pred_flat, y_true_flat).mean()
        log_kwargs = {
            "on_step": True if step_name == "train" else False,
            "on_epoch": True,
            "logger": True,
            "prog_bar": True,
            "batch_size": y_pred.shape[0],
        }

        self.log(
            f"{step_name}/loss",
            loss,
            **log_kwargs,
        )

        # Compute metrics
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                if "grouped" in metric.__class__.__name__.lower():
                    metric.update(y_pred_flat, y_true_flat, groups=subject_ids_flat)
                else:
                    if "retrieval" in metric_name:
                        y_pred_ret = y_pred.mean(dim=-1)
                        y_true_ret = y_true.mean(dim=-1)
                        if bad_samples is not None:
                            y_pred_ret = y_pred_ret[~bad_samples]
                            y_true_ret = y_true_ret[~bad_samples]
                        metric.update(y_pred_ret, y_true_ret)
                    else:
                        metric.update(y_pred_flat, y_true_flat)
                    self.log(
                        metric_name,
                        metric,
                        **log_kwargs,
                    )
        return loss, y_pred.detach().cpu(), y_true.detach().cpu()

    def on_val_or_test_epoch_end(self, step_name: str) -> None:
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                if "grouped" in metric.__class__.__name__.lower():
                    subject_id_to_name = {
                        v: k
                        for k, v in self.config[
                            "data.subject_id.predefined_mapping"
                        ].items()
                    }
                    metric_dict = {
                        metric_name + "/" + subject_id_to_name[int(k)]: v
                        for k, v in metric.compute().items()
                    }
                    self.log_dict(metric_dict)
                    metric.reset()

    def on_validation_epoch_end(self) -> None:
        self.on_val_or_test_epoch_end("val")
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.on_val_or_test_epoch_end("test")
        return super().on_test_epoch_end()

    def training_step(self, batch: SegmentData, batch_idx):
        loss, _, _ = self._run_step(batch, batch_idx, step_name="train")
        return loss

    def validation_step(self, batch: SegmentData, batch_idx, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(
            batch, batch_idx, step_name="val", dataloader_idx=dataloader_idx
        )
        return y_pred, y_true

    def test_step(self, batch: SegmentData, batch_idx, dataloader_idx: int = 0):
        _, y_pred, y_true = self._run_step(
            batch, batch_idx, step_name="test", dataloader_idx=dataloader_idx
        )
        return y_pred, y_true

    def configure_optimizers(self):
        optim_config = self.optim_config.copy()
        unfrozen_params = [p for p in self.parameters() if p.requires_grad]
        if self.config["max_steps"] > 0:
            total_steps = self.config["max_steps"]
        else:
            total_steps = self.trainer.estimated_stepping_batches
        optimizer = optim_config.build(unfrozen_params, total_steps=total_steps)
        return optimizer
