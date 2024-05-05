import os
import collections
from argparse import Namespace
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error
from types import FunctionType
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)
from itertools import repeat

import torch
from lightning.pytorch.callbacks import Callback
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import TensorBoardLogger


def grid_search(hp: dict, hp_skips: list) -> list:
    def search(hp: dict) -> list:
        kl = [k for k, v in hp.items() if type(v) == list]
        if not kl:
            args = Namespace()
            for k, v in hp.items():
                setattr(args, k, v)
            return [args]
        out = []
        for item in hp[kl[0]]:
            hp_ = hp.copy()
            hp_[kl[0]] = item
            out += search(hp_)
        return out

    def skip(hp: Namespace, hp_skips: list) -> bool:
        if not hp_skips:
            return False
        for hp_skip in hp_skips:
            for k, v in hp_skip.items():
                v = [v] if not isinstance(v, list) else v
                if not getattr(hp, k) in v:
                    match = False
                    break
                match = True
            if match:
                return True
        return False

    return [_ for _ in search(hp) if not skip(_, hp_skips)]


# https://www.kaggle.com/code/seshurajup/eegs-train-split-cv
def kfold(
    train_meta_df: pd.DataFrame,
    n_folds: int,
    cache_dir: str = "cache",
):
    fname = f"{cache_dir}/{n_folds}.parquet"
    try:
        return pd.read_parquet(fname).values.tolist()
    except:
        pass

    gkf = GroupKFold(n_splits=n_folds)
    best_distribution = None
    best_balance_score = float("inf")
    best_iteration = -1

    iterations = len(train_meta_df) * 10
    for iteration in tqdm(range(iterations), total=iterations):
        shuffled_patient_ids = (
            train_meta_df["patient_id"].sample(frac=1).reset_index(drop=True)
        )
        train_meta_df["shuffled_patient_id"] = shuffled_patient_ids

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(
                train_meta_df,
                train_meta_df["target"],
                groups=train_meta_df["shuffled_patient_id"],
            )
        ):
            train_meta_df.loc[val_idx, "fold"] = fold
        current_folds = train_meta_df["fold"].values
        fold_distributions = (
            train_meta_df.groupby("fold")["target"]
            .value_counts(normalize=True)
            .unstack()
            .fillna(0)
        )
        balance_score = calculate_balance_score(fold_distributions)

        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_distribution = fold_distributions
            best_iteration = iteration
            best_folds = current_folds
            # print("Improved", best_balance_score)

    ind_list = np.arange(0, len(train_meta_df))
    fold_arrs = []
    for i in range(0, n_folds):
        train_index = ind_list[best_folds != i]
        valid_index = ind_list[best_folds == i]
        fold_arrs.append([train_index, valid_index])
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame(columns=["train", "valid"], data=fold_arrs).to_parquet(fname)
    return fold_arrs


def calculate_balance_score(distribution):
    flat_distribution = distribution.values.flatten()
    mean_distribution = np.mean(flat_distribution)
    balance_score = mean_absolute_error(
        flat_distribution, [mean_distribution] * len(flat_distribution)
    )
    return balance_score


class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step=None) -> None:
        metrics.pop("epoch", None)
        return super().log_metrics(metrics, step)

    @property
    def log_dir(self) -> str:
        version = (
            self.version
            if isinstance(self.version, str)
            else f"version_{self.version:02}"
        )
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir


class ExCB(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            raise exception


def inv_sqrt_sched(current_step: int, num_warmup_steps: int, timescale=None) -> float:
    timescale = num_warmup_steps if timescale is None else timescale
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / (((current_step + shift) / timescale) ** 0.5)
    return decay


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def _log_api_usage_once(obj: Any) -> None:
    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;

    Args:
        obj (class instance or method): an object to extract info from.
    """
    module = obj.__module__
    if not module.startswith("torchvision"):
        module = f"torchvision.internal.{module}"
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{module}.{name}")


V = TypeVar("V")


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value: V) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead."
            )
    else:
        kwargs[param] = new_value


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2.0,
        num_classes=6,
        alpha=None,
        max_batch_size=256,
        eps=1e-4,
        mode="class",
    ):
        super().__init__()
        self.gamma = gamma
        if alpha == None:
            alpha = torch.ones(num_classes)
        self.alpha = alpha.unsqueeze(0).repeat(max_batch_size, 1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.eps = eps
        self.mode = mode

    def forward_log_alpha(self, x1, x2):
        x = x1 * torch.log(x2)
        return x

    def forward(self, pred, target):
        pred = torch.nn.functional.softmax(pred, dim=1)
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        inv_pred = 1 - pred
        inv_pred = torch.clamp(inv_pred, self.eps, 1 - self.eps)
        target = torch.clamp(target, self.eps, 1 - self.eps)
        pred_log = self.forward_log_alpha(target, pred)
        target_log = self.forward_log_alpha(target, target)
        loss = -(inv_pred**self.gamma) * self.alpha[: pred.shape[0]].to(
            pred.get_device()
        )
        loss *= (pred_log - target_log) if self.mode == "kl" else pred_log
        loss = torch.mean(torch.sum(loss, dim=1))
        return loss
