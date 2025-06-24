import numpy as np
import pandas as pd
import pickle as pkl
import os
from scipy.stats import wilcoxon
from utils import fit_eval_folds,


if __name__ == "__main__":
    PATH = os.path.join('./data/20230601/processed', 'data_for_ML.csv')
    sex_diary = pd.read_csv(PATH)
    sex_diary = sex_diary.sort_values(['sexdate', 'sextime'])
    results = {}

    # Random forest (11 features)
    features = [
        'sextime', 'pre_date', 'condom', 'position',
        'mood', 'partnerage', 'partnerold', 'partneryoung',
        'partnersame', 'partnerHIV', 'icon_default'
    ]
    assert len(features) == 11

    hyperparams_rf = {
        'random_state': 1234,
    }
    results["rf"] = fit_eval_folds(
        features=features,
        n_fold=10,
        positive_label='1',
        clf_name='rf',
        hyperparams=hyperparams_rf,
        sex_diary=sex_diary.fillna(9999),
        label_name='protect_our',
        print_train=True,
    )

    # XGBoost (11 features)

    features = [
        'sextime', 'pre_date', 'condom', 'position',
        'mood', 'partnerage', 'partnerold', 'partneryoung',
        'partnersame', 'partnerHIV', 'icon_default'
    ]
    assert len(features) == 11

    hyperparams_xgb = {
        'random_state': 1234,
    }
    results["xgb"] = fit_eval_folds(
        features=features,
        n_fold=10,
        positive_label='1',
        clf_name='xgb',
        hyperparams=hyperparams_xgb,
        sex_diary=sex_diary.fillna(9999),
        label_name='protect_our',
        print_train=False,
    )

    # Catboost (11 features)
    features = [
        'sextime', 'pre_date', 'condom', 'position',
        'mood', 'partnerage', 'partnerold', 'partneryoung',
        'partnersame', 'partnerHIV', 'icon_default'
    ]
    cat_features = [
        'pre_date', 'condom', 'position', 'partnerHIV', 'icon_default',
        'partnerold', 'partneryoung', 'partnersame',
    ]
    assert len(features) == 11
    assert len(cat_features) == 8

    hyperparams_cat = {
        'random_state': 1234,
        'verbose': False,
        'cat_features': cat_features,
        'has_time': True
    }
    results["cat"] = fit_eval_folds(
        features=features,
        n_fold=10,
        positive_label='1',
        clf_name='cat',
        hyperparams=hyperparams_cat,
        sex_diary=sex_diary,
        label_name='protect_our',
        print_train=True,
    )

    # Wilcoxon signed-rank test: Random Forest vs Catboost
    wilcoxon_compare(results, "rf", "cat")

    # Wilcoxon signed-rank test: XGBoost vs Catboost
    wilcoxon_compare(results, "xgb", "cat")
