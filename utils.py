import pickle as pkl
import numpy as np
import pandas as pd
import os, time
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import wilcoxon

def _get_classifier_and_params(clf_name:str, hyperparams:dict):
    if clf_name.lower() in ['rf', 'randomforest']:
        return RandomForestClassifier(**hyperparams)
    elif clf_name.lower() in ['xgb', 'xgboost']:
        return XGBClassifier(**hyperparams)
    elif clf_name.lower() in ['cat', 'catboost']:
        return CatBoostClassifier(**hyperparams)
    else:
        raise Exception(f"We don't support the given classifier {clf_name} currently.")


def _get_eval_values(
    eval_dict:dict,
    positive_label:str='1',
    indices:list=['accuracy', 'precision', 'recall', 'f1-score']
):
    values = {}
    for idx in indices:
       values[idx] = eval_dict[idx] if idx == 'accuracy' else eval_dict[positive_label][idx]
    return values


def fit_eval_folds(
    sex_diary:pd.DataFrame,
    features:list,
    n_fold:int=5,
    positive_label:str='1',
    label_name:str='protect_our',
    clf_name:str='rf',
    hyperparams:dict={},
    indices:list=['accuracy', 'precision', 'recall', 'f1-score'],
    print_train:bool=False,
    print_test:bool=False,
    print_overall:bool=True,
    printed_n_digits:int=4,
):
    n_rows = sex_diary.shape[0]
    n_rows_fold = int(n_rows/(n_fold+1))

    results = {}
    results['all_folds_tr'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    results['all_folds_te'] = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}

    data = sex_diary[features]
    if clf_name.lower() in ['cat', 'catboost'] and 'cat_features' in hyperparams.keys():
        for cat_feature in hyperparams['cat_features']:
            data[cat_feature] = data[cat_feature].astype(str)

    for fold in range(n_fold):
        results[fold] = {}
        if print_train or print_test:
            s = '-'*10 + f' Fold {fold} ' + '-'*10
            print(s)

        # Split the data
        idx_middle = n_rows_fold * (fold+1)
        idx_end = n_rows_fold * (fold+2) if fold < n_fold-1 else n_rows
        results[fold]['idx_middle'] = idx_middle
        results[fold]['idx_end'] = idx_end

        X_train = data.iloc[:idx_middle,:]
        y_train = sex_diary[label_name].values[:idx_middle]
        X_test  = data.iloc[idx_middle:idx_end,:]
        y_test  = sex_diary[label_name].values[idx_middle:idx_end]
        if print_train or print_test:
            print(f'Shapes: {X_train.shape} (train set) {X_test.shape} (test set)')

        # Fit the model
        if print_train or print_test:
            print('Training the model ...', end=' ')
            t0 = time.time()
        model = _get_classifier_and_params(clf_name, hyperparams)
        model.fit(X_train, y_train)
        if print_train or print_test:
            tDiff = time.time() - t0
            print(f'Done! Time cost: {tDiff:.2f} seconds.')

        # Get the predictions
        y_pred_tr = model.predict(X_train)
        y_pred_te = model.predict(X_test)
        eval_tr = metrics.classification_report(y_train, y_pred_tr, output_dict=True)
        eval_te = metrics.classification_report(y_test, y_pred_te, output_dict=True)

        # Collect (and print) the results
        s = ['Training |']
        values = _get_eval_values(eval_tr, positive_label, indices)
        for idx in indices:
            v = values[idx]
            results['all_folds_tr'][idx].append(v)
            if print_train:
                v = round(v, printed_n_digits)
                s.append(f'{idx}={v}')
        if print_train:
            print(' '.join(s))

        s = ['Test     |']
        values = _get_eval_values(eval_te, positive_label, indices)
        for idx in indices:
            v = values[idx]
            results['all_folds_te'][idx].append(v)
            if print_test:
                v = round(v, printed_n_digits)
                s.append(f'{idx}={v}')
        if print_test:
            print(' '.join(s))

        results[fold]['model'] = model
        results[fold]['y_pred_tr'] = y_pred_tr
        results[fold]['y_pred_te'] = y_pred_te
        results[fold]['eval_tr'] = eval_tr
        results[fold]['eval_te'] = eval_te
        if print_train or print_test:
            print()

    if print_overall:
        s = '='*20 + ' OVERALL ' + '='*20
        print(s)
        if print_train:
            print(f'Training |', end=' ')
            for idx, v in results['all_folds_tr'].items():
                print(f'{idx}={np.mean(v):.2f}+/-{np.std(v):.2f},', end=' ')
            print()
        print(f'Test     |', end=' ')
        for idx, v in results['all_folds_te'].items():
            print(f'{idx}={np.mean(v):.2f}+/-{np.std(v):.2f},', end=' ')
        print()

    return results


def get_results_df(results:dict, save_path:str):
    df = pd.DataFrame(results)
    if save_path is not None:
        df.to_csv(save_path)
    return df


def get_feature_importance(classifier, features_used, sort:bool=True, save_path:str=None):
    df_fi = pd.DataFrame(classifier.feature_importances_, index=features_used, columns=['Importance'])
    if sort:
        df_fi = df_fi.sort_values(by='Importance', ascending=False)
    if save_path is not None:
        df_fi.to_csv(save_path)
    return df_fi


def plot_feature_importance(df:pd.DataFrame, figsize_fi=(6, 4), savePATH:str=None, k:float=1):
    assert k > 0
    if k == 1:
        k = df.shape[0]
    elif k < 1:
        k = int(df.shape[0]*k)
    else:
        k = int(k)
    df = df.sort_values('Importance', ascending=True).iloc[-k:, :]
    plt.figure(figsize=figsize_fi)
    plt.hlines(range(k), 0, df['Importance'])
    plt.plot(df['Importance'], range(k), 'o')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    try:
        plt.yticks(range(k), df['feature'])
    except:
        plt.yticks(range(k), df.index)
    plt.rc('ytick', labelsize=6)
    plt.grid(linestyle=':')
    if savePATH is None:
        plt.show()
    else:
        plt.savefig(savePATH)
        print('The plot of feature importance is saved as', savePATH)

def concat_feature_importance(results:dict, features:list=None):
    out = []
    for k in results.keys():
        if isinstance(k, int):
            clf = results[k]['model']
            if features is None:
                try:
                    feature_names_ = clf.feature_names_
                except:
                    feature_names_ = clf.feature_names_in_
            else:
                feature_names_ = clf.feature_names_in_ if features is None else features

            df_fi = get_feature_importance(clf, feature_names_, sort=False, save_path=None)
            df_fi.columns = [f'fold_{k}']
            out.append(df_fi)
    out = pd.concat(out, axis=1)
    return out


def plot_agg_feature_importance(
    df_fi_concat:pd.DataFrame,
    figsize_fi=(6, 4),
    markersize:float=3,
    labelsize:float=6,
    savePATH:str=None,
    k:float=1,
    title:str=None,
):
    assert k > 0
    n_features = df_fi_concat.shape[0]
    if k == 1:
        k = n_features
    elif k < 1:
        k = int(n_features*k)
    else:
        k = int(k)

    df_fi_agg = pd.DataFrame({
        'mean': df_fi_concat.mean(axis=1),
        'std':  df_fi_concat.std(axis=1),
    })
    df_fi_agg = df_fi_agg.sort_values('mean', ascending=True).iloc[-k:, :]
    plt.figure(figsize=figsize_fi)
    plt.errorbar(df_fi_agg['mean'], range(k), xerr=df_fi_agg['std'], fmt='o', markersize=markersize)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.yticks(range(k), df_fi_agg.index)
    plt.rc('ytick', labelsize=labelsize)
    plt.grid(linestyle=':')
    if title is not None:
        plt.title(title)

    if savePATH is None:
        plt.show()
    else:
        plt.savefig(savePATH)
        print('The plot of feature importance is saved as', savePATH)

    return df_fi_agg


def run(sex_diary, results_cat:dict, features:list, cat_features:list, n_fold:int, random_state:int):
    n_features = len(features)
    hyperparams_cat = {
        'random_state': random_state,
        'verbose': False,
        'cat_features': cat_features,
        'has_time': True
    }

    exp_name = f"cat_{n_features}f"
    folder = os.path.join('./outputs/', exp_name)
    os.makedirs(folder, exist_ok=True)

    for postive_label in ['1', '0']:
        key = f"{n_features}f_{postive_label}_{n_fold}-fold__{hyperparams_cat['random_state']}"
        results_cat[key] = fit_eval_folds(
            features=features,
            n_fold=n_fold,
            positive_label=postive_label,
            clf_name='cat',
            hyperparams=hyperparams_cat,
            sex_diary=sex_diary,
            label_name='protect_our',
            print_train=False,
        )

        fn = f"results_{n_fold}-folds_label-{postive_label}__{hyperparams_cat['random_state']}"
        PATH = os.path.join(folder, f"{fn}.pkl")
        with open(PATH, 'wb') as f:
            pkl.dump(results_cat[key], f, pkl.HIGHEST_PROTOCOL)
        print(f"Results saved to {PATH}")
        PATH = os.path.join(folder, f"{fn}.csv")
        _ = get_results_df(results_cat[key]['all_folds_te'], PATH)
        print(f"Results saved to {PATH}")

    return results_cat

def wilcoxon_test(scores_A, scores_B):
    statistic, p_value = wilcoxon(scores_A, scores_B, zero_method='wilcox', correction=False)
    print(f"Wilcoxon statistic = {statistic:.3f}, p-value = {p_value:.4f}")
    return statistic, p_value

def wilcoxon_compare(results, model_A, model_B):
    scores_A = results[model_A]['test_scores']
    scores_B = results[model_B]['test_scores']
    assert len(scores_A) == len(scores_B), "Scores lists must have the same length."
    return wilcoxon_test(scores_A, scores_B)
