import sys
import os
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
import optuna

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import optuna

sys.path.append(f"{os.getcwd()[:-4]}/main")
from utils import load_pickle, OptunaEarlyStoppingCallback
from modules import model_objectives
from params import select_param_type

wandb.login(key="f2d4d498ee5f11b3e3503afd7f00f4cf52589e2e")


DEBUG = False
N_FOLDS = 10
FOLD_FOR_ENSEMBLE = 9
INPUT_TYPE = "sentence"  # sentence, bpe
MODEL = "passiveAggresive"  # LGBM, XGB, CatBoost, SGD, complementNB, categoricalNB, ridge, passiveAggresive
ROOT = "../input"
SEED = 99
LOWERCASE = False
VOCAB_SIZE = 30522
PROCESSED_PATH = f"{ROOT}/230110_0.05_v2_10folds"
N_ESTIMATORS = 2000
N_OPTUNA_TRIALS = 3000
OPTUNA_EARLY_STOP_COUNT = 100
BALANCE_CLASS_WEIGHTS = False
configs = {
    "model": MODEL,
    "N_FOLDS": N_FOLDS,
    "INPUT_TYPE": INPUT_TYPE,
    "N_OPTUNA_ITERATIONs": N_OPTUNA_TRIALS,
    "OPTUNA_EARLY_STOP_COUNT": OPTUNA_EARLY_STOP_COUNT,
    "PROCESSED_PATH": PROCESSED_PATH,
}

INITIAL_PARAMS = None

if DEBUG:
    N_ESTIMATORS = 10
    N_OPTUNA_TRIALS = 10
    OPTUNA_EARLY_STOP_COUNT = 10


def select_model_objectives(
    trial,
    opt_mode="coarse",
    model=MODEL,
    n_folds=N_FOLDS,
    data_path=PROCESSED_PATH,
    input_type=INPUT_TYPE,
    seed=SEED,
):
    params = select_param_type(model)

    return model_objectives(
        trial,
        N_ESTIMATORS,
        opt_mode,
        model,
        n_folds,
        data_path,
        input_type,
        seed,
        params,
        BALANCE_CLASS_WEIGHTS,
        DEBUG,
    )


def select_model_objectives_coarse(trial):
    return select_model_objectives(trial, opt_mode="coarse")


def select_model_objectives_finetuning(trial):
    return select_model_objectives(trial, opt_mode="finetune")


def optuna_optimization(opt_mode="coarse", first_trial_param=None):
    wandb_kwargs = {
        "project": "Detect AI-Generated Text",
        "group": f"{MODEL}_params",
        "name": f"{MODEL}_{INPUT_TYPE}_{opt_mode}",
        "config": configs,
    }
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
    early = OptunaEarlyStoppingCallback(early_stop_count=OPTUNA_EARLY_STOP_COUNT)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="maximize", study_name="Classifier", sampler=sampler
    )
    if first_trial_param is not None:
        study.enqueue_trial(first_trial_param)
    study.optimize(
        select_model_objectives_coarse
        if opt_mode == "coarse"
        else select_model_objectives_finetuning,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
        callbacks=[wandbc, early],
    )
    wandb.log(study.best_params)
    wandb.log({"Best value": study.best_value})
    wandb.finish()
    print(f"{wandb_kwargs['name']}_Best parameters")
    print(study.best_params)
    print(f"Best auc: {study.best_value}")
    return study


# Coarse optimization with only 1 fold
# coarse_study = optuna_optimization("coarse", INITIAL_PARAMS)
# Finetuning optimization with all N_FOLDS
finetune_study = optuna_optimization("finetune", None)
