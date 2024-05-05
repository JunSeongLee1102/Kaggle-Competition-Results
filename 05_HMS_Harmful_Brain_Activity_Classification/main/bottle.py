import os
import sys
import gc
import logging
import random
import time
from glob import glob
import warnings
from argparse import Namespace
from typing import Any

import rich
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import lightning.pytorch as pl
from module import HMS_Model

from utils import FocalLoss


class HMS_Lightning(pl.LightningModule):

    def __init__(self, hp: Namespace, origin_ckpt_path: str = ""):
        super().__init__()
        self.model = HMS_Model(hp, origin_ckpt_path)
        self.save_hyperparameters(hp)
        self.hp = hp
        self.x_batch = None
        self.y_batch = None
        self.best_loss = 10000

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.model = AWP(
            self.model,
            self.optimizers().optimizer,
            scaler=torch.cuda.amp.GradScaler(),
            **self.hp.awp,
            grad_clip=self.hp.grad_clip,
        )

        self.n_steps = self.trainer.estimated_stepping_batches
        self.n_warmup_steps = self.n_steps * self.hp.lr_warmup
        self.hp_metric = 1.0
        self.val_cache = []
        self.logger.log_hyperparams(self.hp, {"hp/metric": self.hp_metric})

    def on_train_epoch_start(self):
        t = self.trainer
        self.logger.log_metrics({"hp/epoch": t.current_epoch}, t.global_step)

    def training_step(self, batch, batch_idx):
        self.update_LR()
        x, y, category = batch["x"], batch["y"], batch["category"]
        self.x_batch = x
        self.y_batch = y
        self.category_batch = category
        out = self.forward(x)
        out = F.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss_kl = kl_loss(out, y)
        log_d = {"loss/kl/T": loss_kl}
        self.log_dict(log_d, on_step=True, on_epoch=True, add_dataloader_idx=False)

        return loss_kl

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            loss.backward(*args, **kwargs)
        # AWP
        if loss.item() < self.hp.awp_metric_thres:
            self.model.attack_backward(
                self.x_batch, self.y_batch, self.trainer.current_epoch
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, category = batch["x"], batch["y"], batch["category"]
        out = self.forward(x)
        out = F.log_softmax(out, dim=1)
        if dataloader_idx:
            out = (out + self.val_cache.pop(0)) / 2
        elif self.hp.flip_val:
            return self.val_cache.append(out)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        loss = kl_loss(out, y)
        log_d = {"loss/V": loss}
        for i in range(0, 5):
            part_out, part_y = out[category[:, 0] == i, :], y[category[:, 0] == i, :]
            if len(part_out) > 0:
                part_loss = kl_loss(part_out, part_y)
                log_d[f"loss/V_{i}"] = part_loss

        self.log_dict(log_d, on_step=False, on_epoch=True, add_dataloader_idx=False)
        return loss

    def on_validation_end(self):
        l = self.trainer.logged_metrics["loss/V"].item()
        if l < self.hp_metric:
            self.hp_metric = l
            self.logger.log_metrics({"hp/metric": l}, self.trainer.global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return F.softmax(self(batch), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=self.hp.wt_decay
        )
        return optimizer

    def update_LR(self):
        lr_m = inv_sqrt_sched(self.trainer.global_step, self.n_warmup_steps)
        self.log("hp/lr", (lr := self.hp.lr * lr_m))
        for p in self.optimizers().optimizer.param_groups:
            p["lr"] = lr


def inv_sqrt_sched(current_step: int, num_warmup_steps: int, timescale=None) -> float:
    timescale = num_warmup_steps if timescale is None else timescale
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / (((current_step + shift) / timescale) ** 0.5)
    return decay


# https://www.kaggle.com/code/wht1996/feedback-nn-train
class AWP(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None,
        grad_clip=1.0,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def attack_backward(self, x, y, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                out = self.model(x)
                adv_loss = self.kl_loss(F.log_softmax(out, dim=1), y)
            # self.optimizer.zero_grad()
            adv_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(), max_norm=self.grad_clip
            )

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1) and not torch.isnan(norm2):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    # r_at = torch.clip(r_at, min=-self.grad_clip, max=self.grad_clip)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(
        self,
    ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

    def forward(self, x):
        return self.model(x)
