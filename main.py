import os
import yaml
import joblib
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from pprint import pprint
from shutil import copyfile
from sklearn import metrics, dummy, linear_model, ensemble, svm, neural_network
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from nilearn.datasets import fetch_atlas_aal


def get_yaml(f):
    with open(f) as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def main(config_path):
    config = get_yaml(config_path)

    # prepare result path
    if not os.path.isdir(config['result_dir']):
        os.mkdir(config['result_dir'])
    filename = 'fmri-radiomic-ml-sad-'
    fileid = 0
    while True:
        expname = filename+str(fileid)
        if os.path.isdir(os.path.join(config['result_dir'], expname)):
            fileid += 1
            continue
        break
    os.mkdir(os.path.join(config['result_dir'], expname, 'fig'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'fig', 'confusion_matrix'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'fig', 'importance'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'fig', 'importance', 'weights'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'fig', 'importance', 'shap'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'model'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'result'))
    os.mkdir(os.path.join(config['result_dir'], expname, 'result', 'cv'))
    copyfile(os.path.abspath(__file__), os.path.join(config['result_dir'], expname, os.path.basename(__file__)))
    copyfile(config_path, os.path.join(config['result_dir'], expname, os.path.basename(config_path)))

    # get aal atlas
    atlas = fetch_atlas_aal()
    atlas_map = {}
    for index, label in zip(atlas['indices'], atlas['labels']):
        atlas_map[index] = label

    # prepare crf dataframe
    df = pd.read_csv(config['crf_path'], index_col=0)
    df = df.drop(['lsas_anx','lsas_avoid'], axis=1)
    df = df.loc[~df['age'].isna() & ~df['sex'].isna()]
    df['age'] = df['age'].astype(int)

    # prepare functional radiomic feature dataframe
    for _a in config['atlases']:
        for _f in config['features']:
            _df = pd.read_csv(os.path.join(config['data_dir'], f'{_a}_{_f}.csv'), index_col=0)
            if _f=='dc':
                _df = _df.loc[_df['Sub-brick']=='0[Binary De]']
            _df['sub'] = _df['File'].map(lambda x: x.split('/')[-2])
            _df = _df.drop(['File', 'Sub-brick'], axis=1)
            _df = _df.set_index('sub')
            _df.columns = [col.rstrip(' ') for col in _df.columns.to_list()]
            _df.columns = [col.replace('Mean', f'{_f}_{_a}') for col in _df.columns.to_list()]
            _df.columns = [col.replace(col[-4:], atlas_map[col[-4:]]) for col in _df.columns.to_list()]
            df = df.join(_df)
    for column in df.columns.to_list()[17:]:
        df = df.loc[~df[column].isnull()]
        _drop = True
        for region in config['regions']:
            if region in column:
                _drop = False
        if _drop:
            df = df.drop(column, axis=1)


    ### start analysis

    # define models
    models = [
        ('Dummy', dummy.DummyClassifier()),
        ('LogReg', linear_model.LogisticRegression()),
        ('SVM', svm.SVC()),
        ('RandomForest', ensemble.RandomForestClassifier()),
        ('MLP', neural_network.MLPClassifier()),
        ('XGBoost', XGBClassifier()),
    ]

    # define hyperparameter search space
    param_grid = {
        'Dummy': {
        
        },
        'LogReg': {
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [float(10**x) for x in range(-4, 4)],
            'l1_ratio': np.random.uniform(size=5),
            'max_iter': [10000]
        },
        'SVM': {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [float(10**x) for x in range(-3, 3)],
            'gamma': [float(10**x) for x in range(-3, 4)],
            'probability': [True],
        },
        'RandomForest': {
            'n_estimators': [10, 25, 100],
            'max_config['features']': ['auto', 3, 9, 15],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'MLP': {
            'hidden_layer_sizes': [10, 50, 100],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [float(10**x) for x in range(-2,2)],
            'batch_size': [int(2**x) for x in range(3, 5)],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [1000],
            'warm_start': [False, True]
        },
        'XGBoost': {
            'booster': ['gbtree', 'gblinear', 'dart'],

            'learning_rate': [float(0.1**x) for x in range(1, 5)],
            'n_estimators': [10, 25, 100],
            'sampling_method': ['uniform', 'gradient_based'],
            'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
            'max_leaves': [int(x) for x in np.linspace(0, 50, num = 6)],

            'lambda': [float(10**x) for x in range(-2,3)],
            'alpha': [float(10**x) for x in range(-2,3)],
        },
    }

    # prepare input-label data pairs
    df = df.loc[~df['lsas_total'].isnull()]
    X = df[df.columns[17:]]
    y = df['lsas_total']
    y = pd.qcut(y, q=len(config['qcuts']), labels=config['qcuts'])
    config['features'] = list(X.columns)

    # split data pairs into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=23, random_state=config['seed'], shuffle=True, stratify=y)

    # standardize input data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # prepare category label encoding with integers
    le = LabelEncoder()
    le.fit(config['qcuts'])

    # plot demographics
    sns.set_theme(context='paper', style="white", font='Freesans', font_scale=1.5, palette='muted')

    df_demo = df.rename(columns = {'lsas_total': 'LSAS', 'age': 'Age', 'hads_a': 'HADS-Anxiety', 'hads_d': 'HADS-Depression'})
    fig, ax = plt.subplots(ncols=4, figsize=(15,3.5), sharey=True)

    for i, x in enumerate(['Age', 'LSAS', 'HADS-Anxiety', 'HADS-Depression']):
        sns.histplot(x=x, data=df_demo, ax=ax[i], kde=True)
        if x=='LSAS':
            ax[i].axvline(x=df_demo[x].median(), color='r', linestyle='--', linewidth=2.0)

    plt.tight_layout()
    plt.savefig(os.path.join(config['result_dir'], expname, 'fig', 'demographics.png'))
    plt.close()

    # start model training
    result_dict = {}
    weight_importance_dict = {}
    shap_importance_dict = {}

    for model_name, model in MODELS:
        # train 5-fold cross validation with exhaustive grid search
        model_hyperparamsearch = GridSearchCV(estimator=model, param_grid=param_grid[model_name], cv=NUM_FOLDS, verbose=0, n_jobs=-1, config['scoring']=config['scoring'])
        model_hyperparamsearch.fit(X_train, y_train if not model_name=='XGBoost' else le.transform(y_train))
        best_estimator = model_hyperparamsearch.best_estimator_

        # get prediction on the test dataset
        pred_test = model_hyperparamsearch.predict(X_test)
        pred_test = pred_test if not model_name=='XGBoost' else le.inverse_transform(pred_test)

        # keep test performance results
        result_dict[model_name] = {}
        result_dict[model_name]['uncertainty_mean'] = best_estimator.predict_proba(X_test).max(1).mean()
        result_dict[model_name]['uncertainty_std'] = best_estimator.predict_proba(X_test).max(1).std()
        result_dict[model_name]['test_acc'] = metrics.accuracy_score(y_test, pred_test)
        result_dict[model_name]['test_acc_bal'] = metrics.balanced_accuracy_score(y_test, pred_test)
        result_dict[model_name]['test_f1'] = metrics.f1_score(y_test, pred_test, average='binary' if y.nunique()==2 else "macro", pos_label=y.unique()[-1] if y.nunique()==2 else None)
        result_dict[model_name]['test_cohen_kappa'] = metrics.cohen_kappa_score(y_test, pred_test, labels=y.unique())
        result_dict[model_name]['val_'+config['scoring']] = model_hyperparamsearch.best_score_
        pprint(result_dict)

        # plot confusion matrix
        cm = metrics.confusion_matrix(y_test, pred_test, labels=y.unique())
        cm /= cm.sum() # convert to ratio
        ax = sns.heatmap(cm, annot=True, xticklabels=y.unique(), yticklabels=y.unique(), cmap='Blues', fmt='.2f', square=True)
        plt.suptitle(f'{model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(config['result_dir'], expname, 'fig', 'confusion_matrix', model_name+'.png'))
        plt.close()

        # compute feature importance for XGBoost
        if 'XGBoost' in model_name:
            # get coefficient weight feature importance
            importance = {'weights': np.abs(best_estimator.feature_importances_)}

            # compute shap importance
            background = shap.maskers.Partition(X_test)
            explainer = shap.Explainer(lambda x: shap.links.identity(best_estimator.predict_proba(x, validate_config['features']=False))[:,1], background, link=shap.links.logit) if model_name=='XGBoost' else shap.Explainer(lambda x: shap.links.identity(best_estimator.predict_proba(x))[:,1], background, link=shap.links.logit)
            shap_values = explainer(X_test)
            importance['shap'] = shap_values.abs.mean(0).values

            # keep feature importance scores
            weight_importance_dict[model_name] = {}
            shap_importance_dict[model_name] = {}
            for feature, weight_imp, shap_imp in zip(config['features'], importance['weights'], importance['shap']):
                weight_importance_dict[model_name][feature] = weight_imp
                shap_importance_dict[model_name][feature] = shap_imp

            # plot feature importance
            for k in importance.keys():
                sort_order = importance[k].argsort()[::-1]
                fig, ax = plt.subplots(figsize=(6,4+len(config['features'])//6))
                sns.barplot(x=importance[k][sort_order], y=[config['features'][idx] for idx in sort_order], ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(config['result_dir'], expname, 'fig', 'importance', k, model_name+'.png'))
                plt.close()

        # save model and results
        joblib.dump(best_estimator, os.path.join(config['result_dir'], expname, 'model', model_name+'.joblib'))
        pd.DataFrame.from_dict(model_hyperparamsearch.cv_results_).to_csv(os.path.join(config['result_dir'], expname, 'result', 'cv', model_name+'.csv'))
        pd.DataFrame.from_dict(result_dict).T.to_csv(os.path.join(config['result_dir'], expname, 'result', 'result.csv'))
        pd.DataFrame.from_dict(weight_importance_dict).T.to_csv(os.path.join(config['result_dir'], expname, 'result', 'importance_weight.csv'))
        pd.DataFrame.from_dict(shap_importance_dict).T.to_csv(os.path.join(config['result_dir'], expname, 'result', 'importance_shap.csv'))


if __name__ == '__main__':
    main('config.yaml')
    exit(0)
