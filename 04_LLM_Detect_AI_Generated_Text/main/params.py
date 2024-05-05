def select_param_type(model):
    if model == "LGBM":
        return lgbm_params_sweep_range
    elif model == "XGB":
        return xgb_params_sweep_range
    elif model == "CatBoost":
        return catboost_params_sweep_range
    elif model == "SGD":
        return sgd_params_sweep_range
    elif model == "multinomialNB":
        return multinomialnb_params_sweep_range
    elif model == "complementNB":
        return complementnb_params_sweep_range
    elif model == "ridge":
        return ridge_params_sweep_range
    elif model == "passiveAggresive":
        return passive_aggresive_params_sweep_range


def lgbm_params_sweep_range(trial):
    params = {
        "verbosity": trial.suggest_int("verbosity", -1, -1),
        "random_state": trial.suggest_int("random_state", 42, 42),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 100),
        "metric": trial.suggest_categorical("metric", ["auc"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 10.0),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 32, 2048),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.0, 1e-2),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 40),
        "max_bin": trial.suggest_int("max_bin", 128, 1024),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 1),
    }
    return params


def xgb_params_sweep_range(trial):
    params = {
        "tree_method": trial.suggest_categorical("tree_method", ["hist"]),
        "device": trial.suggest_categorical("device", ["cuda"]),
        "random_state": trial.suggest_int("random_state", 42, 42),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 100),
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "max_bin": trial.suggest_int("max_bin", 128, 2048),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    return params


def catboost_params_sweep_range(trial):
    params = {
        "verbose": trial.suggest_int("verbose", 0, 0),
        "eval_metric": trial.suggest_categorical("eval_metric", ["AUC"]),
        "depth": trial.suggest_int("depth", 3, 6),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain"]),
        "border_count": trial.suggest_int("border_count", 1, 512),
        "feature_border_type": trial.suggest_categorical(
            "feature_border_type",
            [
                "Median",
                # "Uniform",
                # "UniformAndQuantiles",
                "GreedyLogSum",
                "MaxLogSum",
                "MinEntropy",
            ],
        ),
        # "grow_policy": trial.suggest_categorical(
        #    "grow_policy", ["SymmetricTree", "Lossguide"]
        # ),
        "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 100, 100),
        "random_strength": trial.suggest_float("random_strength", 0.5, 1.5),
        "fold_permutation_block": trial.suggest_int("fold_permutation_block", 1, 256),
        "leaf_estimation_method": trial.suggest_categorical(
            "leaf_estimation_method", ["Newton", "Gradient"]
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10),
        "model_size_reg": trial.suggest_float("model_size_reg", 1e-2, 10),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 1, 4),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1),
        "subsample": trial.suggest_float("subsample", 5e-2, 1.0),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bernoulli", "Poisson", "MVS"]
        ),
        "allow_const_label": trial.suggest_categorical("allow_const_label", [True]),
        "task_type": trial.suggest_categorical("task_type", ["GPU"]),
    }
    return params


def sgd_params_sweep_range(trial):
    params = {
        "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        "loss": trial.suggest_categorical(
            "loss",
            [
                "log_loss",
                "modified_huber",
            ],
        ),
        "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
        "alpha": trial.suggest_float("alpha", 1e-4, 9e-1),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", ["constant", "optimal", "invscaling", "adaptive"]
        ),
        "eta0": trial.suggest_float("eta0", 0.0, 1.0),
        "early_stopping": trial.suggest_categorical("early_stopping", [True]),
        "n_iter_no_change": trial.suggest_int("n_iter_no_change", 100, 100),
        "warm_start": trial.suggest_categorical("warm_start", [True]),
        "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
    }
    return params


def multinomialnb_params_sweep_range(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
    }
    return params


def bernoullinb_params_sweep_range(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0),
        "force_alpha": trial.suggest_categorical("force_alpha", [True, False]),
        "binarize": trial.suggest_float("binarize", 1e-2, 0.99),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
    }
    return params


def complementnb_params_sweep_range(trial):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0),
        "force_alpha": trial.suggest_categorical("force_alpha", [True, False]),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
        "norm": trial.suggest_categorical("norm", [True, False]),
    }
    return params


def ridge_params_sweep_range(trial):
    params = {
        "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0),
    }
    return params


def passive_aggresive_params_sweep_range(trial):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0),
        "max_iter": trial.suggest_int("max_iter", 1000, 10000),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "early_stopping": trial.suggest_categorical("early_stopping", [True]),
        "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
        "average": trial.suggest_int("average", 0, 10),
    }
    return params
