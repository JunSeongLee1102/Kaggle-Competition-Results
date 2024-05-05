import os
import sys
import gc
import logging
import random
import time
import warnings
from argparse import Namespace
from dataclasses import dataclass
import copy
import math
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import rich
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)
from torchvision.models._api import register_model, Weights, WeightsEnum

import lightning.pytorch as pl

# import timm
# from transformers import ConvNextImageProcessor, ConvNextForImageClassification
# from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

from utils import _make_ntuple, _log_api_usage_once, _ovewrite_named_param

root_path = os.getcwd().split("/")[:-1]
root_path = "/" + "/".join(root_path)
sys.path.append(f"{root_path}/models")
from efficientnet import _efficientnet, _efficientnet_conf, Conv2dNormActivation


class HMS_Model(nn.Module):
    def __init__(self, hp: Namespace, origin_ckpt_path: str):
        super().__init__()
        # self.ch_conv = nn.Conv2d(8, 3, 1)
        inverted_residual_setting, last_channel = _efficientnet_conf(
            "efficientnet_b0", width_mult=1.0, depth_mult=1.0
        )

        self.base_model = _efficientnet(
            inverted_residual_setting,
            dropout=hp.dropout,
            last_channel=last_channel,
            weights=None,
            progress=True,
            stochastic_depth_prob=hp.stochastic_prob,
        )

        if origin_ckpt_path != "":
            self.base_model.load_state_dict(torch.load(origin_ckpt_path), strict=False)
        self.base_model.features[0][0] = nn.Conv2d(
            3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, 6, dtype=torch.float32
        )

        self.prob_out = nn.Softmax()
        self.hp = hp
        self.global_avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x1 = [x[:, :, :, i : i + 1] for i in range(4)]
        # x1 = torch.concat(x1, dim=1)
        # x2 = [x[:, :, :, i + 4 : i + 5] for i in range(4)]
        # x2 = torch.concat(x2, dim=1)
        # if self.hp.use_kaggle_spectrograms & self.hp.use_eeg_spectrograms:
        #    x = torch.concat([x1, x2], dim=2)
        # elif self.hp.use_eeg_spectrograms:
        #    x = x2
        # else:
        #    x = x1
        x = torch.concat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        out = self.base_model(x)
        return out
