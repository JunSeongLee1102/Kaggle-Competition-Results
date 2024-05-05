import gc
import os
import shutil
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, sum_models
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier
from scipy import sparse
from utils import save_pickle, load_pickle, average_preds

from joblib import Parallel, delayed

# split dataset and do consecutive training of the catboost to support GPU!
N_SPLIT_CATBOOST_TRAIN = 8
# split catboost partial models to N group to avoid memory issue again
N_SPLIT_CATBOOST_GROUP = 2


def model_objectives(
    trial,
    n_estimators,
    opt_mode,
    model_name,
    n_folds,
    data_path,
    input_type,
    seed,
    model_params,
    use_prev_error,
    debug,
):
    if opt_mode == "coarse" or opt_mode == "infer_one_fold":
        folds = [0]
    elif opt_mode == "finetune" or opt_mode == "infer_all_folds":
        folds = np.arange(0, n_folds - 1)  # to save fold for ensemble
    params = model_params if trial is None else model_params(trial)
    if debug:
        print(params)
    sample_count = 0
    sum_scores = 0
    (
        tf_train,
        y_train,
        fold_info,
        tf_org_train,
        y_org_train,
        tf_test,
        prev_best,
        prev_errs,
    ) = load_data(data_path, input_type, seed, model_name, use_prev_error)
    err_valids = []
    pred_tests = []
    pred_for_ensembles = []
    for fold in folds:
        (
            tf_train_fold,
            y_train_fold,
            tf_valid_fold,
            y_valid_fold,
            tf_for_ensemble,
            y_for_ensemble,
        ) = get_train_valid_folds(
            fold, fold_info, tf_train, y_train, tf_org_train, y_org_train
        )
        # prev_errs_fold = None if prev_errs is None else prev_errs[fold_info != fold]
        prev_errs_fold = None
        pred_valid, pred_for_ensemble, pred_test = subprocess_inference(
            model_name,
            n_estimators,
            params,
            tf_train_fold,
            y_train_fold,
            tf_valid_fold,
            y_valid_fold,
            tf_for_ensemble,
            tf_test,
            prev_errs_fold,
        )
        err_valids.append(np.abs(pred_valid - y_valid_fold))
        sum_scores += roc_auc_score(y_valid_fold, pred_valid) * len(y_valid_fold)
        sample_count += len(y_valid_fold)
        pred_tests.append(pred_test)
        pred_for_ensembles.append(pred_for_ensemble)
        del (
            pred_valid,
            pred_test,
            tf_train_fold,
            y_train_fold,
            tf_valid_fold,
            y_valid_fold,
            tf_for_ensemble,
        )
        gc.collect()
    del tf_train, y_train, tf_org_train, y_org_train, tf_test
    gc.collect()
    score = sum_scores / sample_count
    process_error(score, err_valids, data_path, input_type, seed, model_name, prev_best)
    # pred_test = average_preds(pred_tests)
    # del pred_tests
    gc.collect()
    if "infer" in opt_mode:
        return pred_tests, pred_for_ensembles, y_for_ensemble
    else:
        return score


def load_data(data_path, input_type, seed, model_name, use_prev_error):
    base_path = f"{data_path}/{input_type}_seed{seed}_"
    tf_train, y_train, fold_info = load_pickle(f"{base_path}train.pkl")
    tf_org_train, y_org_train = load_pickle(f"{base_path}org_train.pkl")
    tf_test = load_pickle(f"{base_path}test.pkl")
    err_path = f"{base_path}{model_name}_errs.pkl"
    prev_errs = None
    prev_best = 0
    if use_prev_error:
        if os.path.isfile(err_path):
            prev_best_errs = load_pickle(err_path)
            prev_best = prev_best_errs[0]
            prev_errs = prev_best_errs[1:]
    return (
        tf_train,
        y_train,
        fold_info,
        tf_org_train,
        y_org_train,
        tf_test,
        prev_best,
        prev_errs,
    )


def model_selector(
    model_name,
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    model_inputs = {
        "n_estimators": n_estimators,
        "params": params,
        "tf_train": tf_train,
        "y_train": y_train,
        "tf_valid": tf_valid,
        "y_valid": y_valid,
        "tf_for_ensemble": tf_for_ensemble,
        "tf_test": tf_test,
        "prev_errs_fold": prev_errs_fold,
    }

    if model_name == "LGBM":
        return lgbm_fit_pred(**model_inputs)
    elif model_name == "XGB":
        return xgb_fit_pred(**model_inputs)
    elif model_name == "CatBoost":
        return catboost_split_fit_pred(**model_inputs)
    elif model_name == "SGD":
        return sgd_fit_pred(**model_inputs)
    elif model_name == "multinomialNB":
        return multinomialNB_fit_pred(**model_inputs)
    elif model_name == "complementNB":
        return complementNB_fit_pred(**model_inputs)
    elif model_name == "ridge":
        return ridge_fit_pred(**model_inputs)
    elif model_name == "passiveAggresive":
        return passiveAggresive_fit_pred(**model_inputs)


def subprocess_inference(
    model_name,
    n_estimators,
    params,
    tf_train_fold,
    y_train_fold,
    tf_valid_fold,
    y_valid_fold,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    def fit_pred_wrappers(
        model_name=model_name,
        n_estimators=n_estimators,
        params=params,
        tf_train_fold=tf_train_fold,
        y_train_fold=y_train_fold,
        tf_valid_fold=tf_valid_fold,
        y_valid_fold=y_valid_fold,
        tf_for_ensemble=tf_for_ensemble,
        tf_test=tf_test,
        prev_errs_fold=prev_errs_fold,
    ):
        return model_selector(
            model_name,
            n_estimators,
            params,
            tf_train_fold,
            y_train_fold,
            tf_valid_fold,
            y_valid_fold,
            tf_for_ensemble,
            tf_test,
            prev_errs_fold,
        )

    return Parallel(n_jobs=1)(delayed(fit_pred_wrappers)() for i in range(1))[0]


def lgbm_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    fit_params = {
        "X": tf_train,
        "y": y_train,
        "eval_set": [(tf_valid, y_valid)],
        "verbose": False,  # params["verbosity"],
    }
    if prev_errs_fold is not None:
        fit_params["sample_weight"] = prev_errs_fold
    model = LGBMClassifier(n_estimators=n_estimators, **params)
    model.fit(**fit_params)
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model, fit_params
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def xgb_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    fit_params = {
        "X": tf_train,
        "y": y_train,
        "eval_set": [(tf_valid, y_valid)],
        "verbose": 0,
    }
    model = XGBClassifier(n_estimators=n_estimators, **params)
    model.fit(**fit_params)
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model, fit_params
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def catboost_split_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    rng_samples = np.linspace(0, len(y_train), N_SPLIT_CATBOOST_TRAIN + 1, dtype=int)
    preds = []
    for grp_cnt in range(0, N_SPLIT_CATBOOST_GROUP):
        models = []
        for i in range(0, N_SPLIT_CATBOOST_TRAIN // N_SPLIT_CATBOOST_GROUP):
            cur_ind = grp_cnt * N_SPLIT_CATBOOST_TRAIN // N_SPLIT_CATBOOST_GROUP + i

            params["early_stopping_rounds"] = 100
            fit_params = {
                "X": tf_train[rng_samples[cur_ind] : rng_samples[cur_ind + 1]],
                "y": np.array(y_train[rng_samples[cur_ind] : rng_samples[cur_ind + 1]]),
                "eval_set": [(tf_valid, y_valid)],
            }
            tmp_model = CatBoostClassifier(iterations=1000, **params)
            tmp_model.fit(**fit_params)
            models.append(tmp_model)
            del tmp_model, fit_params
            gc.collect()
        model = sum_models(models)
        preds.append(
            model.predict(tf_valid, prediction_type="Probability")[:, 1]
        )  # predict_proba(tf_test)[:,1]
        del models, model
        gc.collect()
    pred = average_preds(preds)
    del preds
    gc.collect()
    return pred


def sgd_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    fit_params = {"X": tf_train, "y": y_train}
    if prev_errs_fold is not None:
        fit_params["sample_weight"] = prev_errs_fold
    model = SGDClassifier(**params)
    model.fit(**fit_params)
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def multinomialNB_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    model = MultinomialNB(**params)
    model.fit(tf_train, y_train)
    pred_valid = model.predict_proba(tf_valid)[:, 1]
    pred_test = model.predict_proba(tf_test)[:, 1]
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def complementNB_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    model = ComplementNB(**params)
    fit_params = {"X": tf_train, "y": y_train}
    if prev_errs_fold is not None:
        fit_params["sample_weight"] = prev_errs_fold
    model.fit(**fit_params)
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model, fit_params
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def ridge_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    model = RidgeClassifier(**params)
    fit_params = {"X": tf_train, "y": y_train}
    if prev_errs_fold is not None:
        fit_params["sample_weight"] = prev_errs_fold
    model.fit(**fit_params)
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model, fit_params
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def passiveAggresive_fit_pred(
    n_estimators,
    params,
    tf_train,
    y_train,
    tf_valid,
    y_valid,
    tf_for_ensemble,
    tf_test,
    prev_errs_fold,
):
    model = PassiveAggressiveClassifier(**params)
    fit_params = {"X": tf_train, "y": y_train}
    model.fit(**fit_params)
    pred_valid, pred_for_ensemble, pred_test = pred_all(
        model, tf_valid, tf_for_ensemble, tf_test
    )
    del model, fit_params
    gc.collect()
    return pred_valid, pred_for_ensemble, pred_test


def get_train_valid_folds(
    fold, fold_info, tf_train, y_train, tf_org_train, y_org_train
):
    max_fold = np.max(fold_info)
    tf_train_fold = tf_train[(fold_info != fold) & (fold_info != max_fold)]
    y_train_fold = y_train[(fold_info != fold) & (fold_info != max_fold)]
    tf_valid_fold = tf_train[fold_info == fold]
    y_valid_fold = y_train[fold_info == fold]
    tf_for_ensemble = tf_train[fold_info == max_fold]
    y_for_ensemble = y_train[fold_info == max_fold]
    merged_tf_valid_fold = sparse.vstack([tf_valid_fold, tf_org_train])
    merged_y_valid_fold = np.hstack([y_valid_fold, y_org_train])
    return (
        tf_train_fold,
        y_train_fold,
        merged_tf_valid_fold,
        merged_y_valid_fold,
        tf_for_ensemble,
        y_for_ensemble,
    )


def pred_all(model, tf_valid, tf_for_ensemble, tf_test):
    try:
        pred_valid = model.predict_proba(tf_valid)[:, 1]
        pred_for_ensemble = model.predict_proba(tf_for_ensemble)[:, 1]
        pred_test = model.predict_proba(tf_test)[:, 1]
    except:  # for models like Ridge
        pred_valid = model.predict(tf_valid)
        pred_for_ensemble = model.predict(tf_for_ensemble)
        pred_test = model.predict(tf_test)
    return pred_valid, pred_for_ensemble, pred_test


def process_error(
    score, err_valids, data_path, input_type, seed, model_name, prev_best
):
    score_err_valids = np.hstack([[score], np.hstack(err_valids)])
    # If best, save prev err because that was used.
    base_path = f"{data_path}/{input_type}_seed{seed}_"
    err_path = f"{base_path}{model_name}_errs.pkl"
    if (prev_best > 0) and (score > prev_best):
        shutil.copy(err_path, f"{err_path[:-4]}_best.pkl")
    save_pickle(err_path, score_err_valids)
    del err_valids
    gc.collect()
