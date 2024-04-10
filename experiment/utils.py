"""
This module contains several functions that are used in various stages of the process
"""
import json
import operator as op
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: Dict[str, Union[int, float, str]], save_dir: str = "./"):
    with open(os.path.join(save_dir, "results.json"), mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path) -> Dict[str, Union[int, float, str]]:
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_object(obj, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def cal_auc_score(model, data, feature_cols, label_col):
    pred_proba = model.predict_proba(data[feature_cols])
    if data[label_col].nunique() == 2:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba[:, 1])
    else:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba, multi_class="ovo")
    return auc


def cal_acc_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[label_col], pred)
    return acc


def cal_metrics(model, data, feature_cols, label_col):
    acc = cal_acc_score(model, data, feature_cols, label_col)
    auc = cal_auc_score(model, data, feature_cols, label_col)
    return {"ACC": acc, "AUC": auc}

def cal_mse_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    mse = mean_squared_error(data[label_col], pred)
    return mse

def cal_rmse_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    mse = mean_squared_error(data[label_col], pred)
    rmse = np.sqrt(mse)
    return rmse

def cal_mae_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    mae = mean_absolute_error(data[label_col], pred)
    return mae

def cal_metrics_regression(model, data, feature_cols, label_col):
    mse = cal_mse_score(model, data, feature_cols, label_col)
    mae = cal_mae_score(model, data, feature_cols, label_col)
    rmse = cal_rmse_score(model, data, feature_cols, label_col)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}

def set_categories_in_rule(ruleset, categories_dict):
    ruleset.set_categories(categories_dict)

def feature_importance(feature_importance_list, columns, name):
    if name == "xgblgbm":
        xgb_feature_importance_list = []
        lgbm_feature_importance_list = []
        for i in range(len(feature_importance_list)):
            xgb_feature_importance_list.append(feature_importance_list[i][0])
            lgbm_feature_importance_list.append(feature_importance_list[i][1])
        xgb_average_importance = np.mean(xgb_feature_importance_list, axis=0)
        lgbm_average_importance = np.mean(lgbm_feature_importance_list, axis=0)
        xgb_feature_importance = pd.DataFrame({'feature': columns, 'importance': xgb_average_importance})
        lgbm_feature_importance = pd.DataFrame({'feature': columns, 'importance': lgbm_average_importance})
        plt_importance(xgb_feature_importance, "xgb_importance")
        plt_importance(lgbm_feature_importance, "lgbm_importance")
    else:
        average_importance = np.mean(feature_importance_list, axis=0)
        feature_importance = pd.DataFrame({'feature': columns, 'importance': average_importance})
        plt_importance(feature_importance)

def plt_importance(feature_importance, name="importance"):
    feature_importance = feature_importance.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='skyblue')
    plt.xlabel('Average Importance')
    plt.title('Average Feature Importance')
    plt.gca().invert_yaxis()
    plt_name = f"{name}.pdf"
    plt.savefig(plt_name)