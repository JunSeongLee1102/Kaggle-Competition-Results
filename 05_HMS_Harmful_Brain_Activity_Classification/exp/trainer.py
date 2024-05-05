import os
import sys
import gc
import logging
import random
import time
import warnings

import rich
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.model_selection import KFold, GroupKFold

if os.path.isdir("/kaggle"):
    ROOT = "/kaggle"
else:
    ROOT = ".."
INPUT = f"{ROOT}/input"
sys.path.append(f"{ROOT}/main/")
from data import HMS_DM
from utils import grid_search, TBLogger, ExCB
from bottle import HMS_Lightning

n_trials = 0
debug = 0
n_workers = 16
run_folds = [0]
swa_params = []
seed = 42
early_stop = 15
cache_data = False if not debug else False
log_dir = "e016"
pred_dir = "subs"
pred = False
metric = "loss/V"
hp_conf = {
    "n_epochs": 100,
    "lr": 0.0005,
    "lr_warmup": 0.001,
    "wt_decay": 0.01,
    "n_grad_accum": 1,
    "augment": True,
    "augment_species": [
        {"hor_flip": {"p": 0.5}},
    ],
    "dropout": 0.2,
    "stochastic_prob": 0.2,
    "stochastic_weight_average": {"swa_lrs": 0.01, "swa_epoch_start": 20},
    "grad_clip": 10.0,
    "awp": {
        "adv_param": "weight",
        "adv_lr": 0.005,
        "adv_eps": 0.05,
        "start_epoch": 1000,
        "adv_step": 1,
    },
    "awp_metric_thres": 0.6,
    "seed": 5,
    "n_folds": 5,
    "fold": 0,
    "read_spec_files": False,
    "read_eeg_spec_files": False,
    "use_kaggle_spectrograms": True,
    "use_eeg_spectrograms": True,
    "flip_val": False,
    "batch_size": 32,
    "cache_data": cache_data,
}
hp_skip = []
ckpt = True
origin_ckpt_path = f"{ROOT}/input/hms-efficientnetb0-pt-ckpts/efficientnet_b0_rwightman-7f5810bc.pth"  # ""

train_meta_csv_path = f"{INPUT}/hms-harmful-brain-activity-classification/train.csv"
train_spec_path = (
    f"{INPUT}/hms-harmful-brain-activity-classification/train_spectrograms/"
)
# train_eeg_path = f"{INPUT}/brain-eeg-spectrograms/"
train_eeg_path = f"{INPUT}/custom-eeg-spectrograms/"
test_meta_csv_path = f"{INPUT}/hms-harmful-brain-activity-classification/test.csv"
test_spec_path = f"{INPUT}/hms-harmful-brain-activity-classification/test_spectrograms/"
test_eeg_path = f"{INPUT}/hms-harmful-brain-activity-classification/test_eegs/"


try:
    with rich.get_console().status("Reticulating Splines"):
        if not debug:
            warnings.filterwarnings("ignore")
            for n in logging.root.manager.loggerDict:
                logging.getLogger(n).setLevel(logging.WARN)
        torch.set_float32_matmul_precision("medium")
        torch.manual_seed(seed)
        random.seed(seed)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        trials = grid_search(hp_conf, hp_skip)
        n_trials = len(trials) if not n_trials else n_trials
        n_trials = len(trials) if len(trials) < n_trials else n_trials

    print(f"Log: {log_dir} | EStop: {early_stop} | Ckpt: {ckpt} | Pred: {pred}")
    for i, hp in enumerate(trials[:n_trials]):
        for j, f in enumerate(run_folds):
            print(f"Trial {i + 1}/{n_trials} Fold {j + 1}/{len(run_folds)} ({f})")
            hp.fold = f
            tbl = TBLogger(os.getcwd(), log_dir, default_hp_metric=False)
            cb = [RichProgressBar(), ExCB()]
            cb += [ModelCheckpoint(tbl.log_dir, None, metric)] if ckpt else []
            cb += [EarlyStopping(metric, 0, early_stop)] if early_stop else []
            if hp.stochastic_weight_average:
                cb += [StochasticWeightAveraging(**hp.stochastic_weight_average)]
            hp.dirpath = cb[2].dirpath
            dm = HMS_DM(
                hp,
                n_workers,
                train_meta_csv_path,
                train_spec_path,
                train_eeg_path,
                test_meta_csv_path,
                test_spec_path,
                test_eeg_path,
            )
            model = HMS_Lightning(hp, origin_ckpt_path)
            # if load_ckpt_path != "":
            #    model.model.base_model.load_state_dict(
            #        torch.load(load_ckpt_path), strict=False
            #    )
            trainer = pl.Trainer(
                precision="bf16-mixed",
                accelerator="gpu",
                benchmark=True,
                max_epochs=hp.n_epochs,
                accumulate_grad_batches=hp.n_grad_accum,
                gradient_clip_val=hp.grad_clip,
                fast_dev_run=debug,
                num_sanity_val_steps=0,
                enable_model_summary=False,
                logger=tbl,
                callbacks=cb,
            )
            gc.collect()
            try:
                trainer.fit(model, datamodule=dm)
            except KeyboardInterrupt:
                print("Fit Interrupted")
                if i + 1 < n_trials:
                    with rich.get_console().status("Quit?") as s:
                        for k in range(3):
                            s.update(f"Quit? {3-k}")
                            time.sleep(1)
                continue
            if pred:
                try:
                    cp = None if debug else "best"
                    preds = trainer.predict(model, datamodule=dm, ckpt_path=cp)
                except KeyboardInterrupt:
                    print("Prediction Interrupted")
                    continue

except KeyboardInterrupt:
    print("Goodbye")
    sys.exit()
