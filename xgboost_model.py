# -*- coding: utf-8 -*-
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=10)
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


"""
Build XGBoost classifier. Is able to receive important arguments such as max_depth and
number of trees, however are not being utilised in the Root().

Returns auc and y_pred_proba used to compute all the evaluation metrics, as well as a simple output
giving an indication of performance based on mislabeled classes.
"""
def buildXGBoost(dataset,columns_to_exclude,exclude_columns,max_depth=7,n_est=100):
    ds = dataset

    # Remove the columns to to be checked.
    if exclude_columns:
        ds.dprint("Excluding following columns: " + str(columns_to_exclude))
        X_train = ds.X_train.drop(columns_to_exclude, inplace=False, axis=1)
        X_test = ds.X_test.drop(columns_to_exclude, inplace=False, axis=1)
    else:
        X_train = ds.X_train
        X_test = ds.X_test

    xgb = XGBClassifier(max_depth=max_depth, n_estimators=n_est, silent=False, learning_rate=0.01)
    xgb.fit(X_train,ds.y_train)

    y_pred = xgb.predict(X_test)
    y_pred_proba = xgb.predict_proba(X_test)
    #y_t_pred_proba = xgb.predict_proba(X_train)

    total_points = ds.y_test.shape[0]
    mislabeled = (ds.y_test != y_pred).sum()
    auc = roc_auc_score(ds.y_test, y_pred_proba[:, 1])
    mislabeled = "Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (ds.y_test != y_pred).sum())

    return auc, y_pred_proba, mislabeled