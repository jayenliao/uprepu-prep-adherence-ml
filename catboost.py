import numpy as np
import pandas as pd
import pickle as pkl
import os
from utils import (
    run,
    concat_feature_importance,
    plot_agg_feature_importance,
    wilcoxon_compare
)

if __name__ == "__main__":
    PATH = os.path.join('./data/20230601/processed', 'data_for_ML.csv')
    sex_diary = pd.read_csv(PATH)
    sex_diary = sex_diary.sort_values(['sexdate', 'sextime'])
    results_cat = {}

    # Model 1 (11 features)
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
    results_cat = run(sex_diary, results_cat, features, cat_features, n_fold=10, random_state=1234)

    # Model 2 (30 features)
    features = [
        'pre_date', 'condom', 'position', 'partnerHIV', 'mood', 'icon_default',
        'partnerold', 'partneryoung', 'partnersame',
        'sextime', 'partnerage', 'user_duration', 'age',
        'sex_cumsum', 'dating_cumsum', 'dose_cumsum', 'switch_cumsum',
        'condom_m', 'pre_date_m', 'position_0_m', 'position_1_m', 'position_10_m',
        'partnerold_m', 'partneryoung_m', 'partnersame_m', 'partnerHIV_unknown_m',
        'partnerHIV_neg_prep_m', 'partnerHIV_neg_noprep_m', 'partnerHIV_posi_uvl_m', 'partnerHIV_posi_other_m'
    ]
    cat_features = [
        'pre_date', 'condom', 'position', 'partnerHIV', 'icon_default',
        'partnerold', 'partneryoung', 'partnersame',
    ]
    assert len(features) == 30
    assert len(cat_features) == 8
    results_cat = run(sex_diary, results_cat, features, cat_features, n_fold=10, random_state=1234)

    df_fi_concat_cat = concat_feature_importance(results_cat["30f_1_10-fold__1234"])
    df_fi_concat_cat['avg'] = df_fi_concat_cat.mean(axis=1)
    df_fi_concat_cat = df_fi_concat_cat.sort_values('avg', ascending=False)

    title = 'Feature Importance of Catboost with 30 Features'
    df_fi_agg_cat_30f = plot_agg_feature_importance(
        df_fi_concat_cat, figsize_fi=(4,6), labelsize=8, title=title)

    ## Model 2-a (25 features)
    k = df_fi_agg_cat_30f.shape[0] - 25
    features_remained = list(df_fi_agg_cat_30f.iloc[k:].index)
    cat_features_remained = list(set(cat_features) & set(features_remained))
    assert len(features_remained) == 25
    assert len(cat_features_remained) <= 8
    results_cat = run(sex_diary, results_cat, features_remained, cat_features_remained, n_fold=10, random_state=1234)

    ## Model 2-b (20 features)
    k = df_fi_agg_cat_30f.shape[0] - 20
    features_remained = list(df_fi_agg_cat_30f.iloc[k:].index)
    cat_features_remained = list(set(cat_features) & set(features_remained))
    assert len(features_remained) == 20
    assert len(cat_features_remained) <= 8
    results_cat = run(sex_diary, results_cat, features_remained, cat_features_remained, n_fold=10, random_state=1234)

    ## Model 2-c (15 features)
    k = df_fi_agg_cat_30f.shape[0] - 15
    features_remained = list(df_fi_agg_cat_30f.iloc[k:].index)
    cat_features_remained = list(set(cat_features) & set(features_remained))
    assert len(features_remained) == 15
    assert len(cat_features_remained) <= 8
    results_cat = run(sex_diary, results_cat, features_remained, cat_features_remained, n_fold=10, random_state=1234)

    ## Model 2-d (10 features)
    k = df_fi_agg_cat_30f.shape[0] - 10
    features_remained = list(df_fi_agg_cat_30f.iloc[k:].index)
    cat_features_remained = list(set(cat_features) & set(features_remained))
    assert len(features_remained) == 10
    assert len(cat_features_remained) <= 8
    results_cat = run(sex_diary, results_cat, features_remained, cat_features_remained, n_fold=10, random_state=1234)

    ## Model 2-e (5 features)
    k = df_fi_agg_cat_30f.shape[0] - 5
    features_remained = list(df_fi_agg_cat_30f.iloc[k:].index)
    cat_features_remained = list(set(cat_features) & set(features_remained))
    assert len(features_remained) == 5
    assert len(cat_features_remained) <= 8
    results_cat = run(sex_diary, results_cat, features_remained, cat_features_remained, n_fold=10, random_state=1234)

    df_fi_concat_cat_5f = concat_feature_importance(results_cat["5f_1_10-fold__1234"])
    df_fi_concat_cat_5f['avg'] = df_fi_concat_cat_5f.mean(axis=1)
    df_fi_concat_cat_5f = df_fi_concat_cat_5f.sort_values('avg', ascending=False)

    title = 'Feature Importance of Catboost with Five Features'
    df_fi_agg_cat_5f = plot_agg_feature_importance(
        df_fi_concat_cat_5f, figsize_fi=(4, 2), markersize=4, labelsize=8, title=title)

    # Model 1 vs. Model 2
    wilcoxon_compare(results_cat, "11f_1_10-fold__1234", "30f_1_10-fold__1234")
    # Model 2 vs. Model 2-a
    wilcoxon_compare(results_cat, "30f_1_10-fold__1234", "25f_1_10-fold__1234")
    # Model 2 vs. Model 2-b
    wilcoxon_compare(results_cat, "30f_1_10-fold__1234", "20f_1_10-fold__1234")
    # Model 2 vs. Model 2-c
    wilcoxon_compare(results_cat, "30f_1_10-fold__1234", "15f_1_10-fold__1234")
    # Model 2 vs. Model 2-d
    wilcoxon_compare(results_cat, "30f_1_10-fold__1234", "10f_1_10-fold__1234")
    # Model 2 vs. Model 2-e
    wilcoxon_compare(results_cat, "30f_1_10-fold__1234", "5f_1_10-fold__1234")

