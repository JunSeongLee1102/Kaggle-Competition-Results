import numpy as np
import pickle
import optuna


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


class OptunaEarlyStoppingCallback:
    def __init__(self, early_stop_count: int):
        self.best_score = 0.0
        self.not_improved_counter = 0
        self.early_stop_count = early_stop_count

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if study.best_value > self.best_score:
            self.best_score = study.best_value
            self.not_improved_counter = 0
        else:
            self.not_improved_counter += 1
        if self.not_improved_counter >= self.early_stop_count:
            print("Optuna study early stopped by early_stop_count")
            study.stop()


def average_preds(preds):
    pred = preds[0].astype(float)
    for i in range(1, len(preds)):
        pred += preds[i].astype(float)
    pred /= len(preds)
    return pred


def weighted_average_preds(preds, weights):
    assert len(preds) == len(weights)
    preds = np.vstack(preds)
    weights = np.hstack(weights)
    pred = np.average(preds, axis=0, weights=weights)
    return pred


def last_fold_changer(train, model_name_last_fold):
    folds = sorted(np.unique(train.fold))
    target_model_fold = 0
    last_fold = np.max(folds)
    for i in range(0, len(folds)):
        part_train = train.loc[train.fold == i]
        model_list = np.unique(part_train.model)
        for model in model_list:
            if model != "human":
                if model == model_name_last_fold:
                    target_model_fold = i
    train.loc[
        (train.fold == last_fold) & (train.model != "human"), "fold"
    ] = target_model_fold
    train.loc[train.model == model_name_last_fold, "fold"] = last_fold
    return train


def models_excluder(train, models_to_exclude):
    for model in models_to_exclude:
        train = train.loc[train.model != model]
    return train


def append_preds(cur_preds, to_append_preds):
    for to_append_pred in to_append_preds:
        cur_preds.append(to_append_pred)
    return cur_preds
