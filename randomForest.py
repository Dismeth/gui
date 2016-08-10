# -*- coding: utf-8 -*-
import pandas as pd                                 # Pandas and Numpy
import numpy as np                                  #
from sklearn import ensemble                        #

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=7)
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# Cross Validation
#from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score


def feature_importance_RandomForest(dataset,columns_to_exclude,exclude_columns):
    ds = dataset
    forest = RandomForestClassifier(n_estimators=5,random_state=0, criterion='gini')
    columns = list(ds.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)
        ds.X_train.drop(columns_to_exclude,inplace=True,axis=1)
    #print(columns_to_exclude)
    #print(list(ds.X_train.columns.values))
    forest.fit(ds.X_train, ds.y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    #featureimportances =
    columns = list(ds.X_train.columns.values)
    neworder = []
    info = pd.DataFrame(columns=['Column', 'Feature Importance'])
    for f in range(ds.X_train.shape[1]):
        neworder.insert(f, columns[indices[f]])
        print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
        info.loc[f] = [columns[indices[f]],importances[indices[f]]]
    #prediction = forest.predict(ds.X_test)
    #score = forest.predict(ds.X_test) # 1 and 0s
    #score2 = forest.predict_proba(ds.X_test) # Probabilities for 0 and 1
    # print(prediction)
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(ds.X_train.shape[1]), importances[indices],
            color="g", yerr=std[indices], align="center")
    plt.xticks(range(ds.X_train.shape[1]), neworder)
    plt.xlim([-1, ds.X_train.shape[1]])
    plt.ylabel("Mean Impurity")
    plt.xlabel("Columns / Features")
    plt.show()
    #print(score)
    #print(score2)
    return info

def buildRandomForest(dataset,columns_to_exclude,exclude_columns):
    ds = dataset
    randforest = RandomForestClassifier(n_estimators=100, max_depth= 7, oob_score=True,n_jobs=-1)

    """
    For own usage:
    n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True,
    oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None

    """
    # Remove the columns to to be checked.
    if exclude_columns:
        ds.dprint("Excluding following columns: " + str(columns_to_exclude))
        X_train = ds.X_train.drop(columns_to_exclude, inplace=False, axis=1)
        X_test = ds.X_test.drop(columns_to_exclude, inplace=False, axis=1)
    else:
        X_train = ds.X_train
        X_test = ds.X_test

    randforest.fit(X_train, ds.y_train)
    #scores = cross_val_score(randforest, X_train, ds.y_train)

    y_pred = randforest.predict(X_test)
    y_pred_proba = randforest.predict_proba(X_test)
    auc = roc_auc_score(ds.y_test, y_pred_proba[:, 1])
    total_points = X_test.shape[0]
    mislabeled = "Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (ds.y_test != y_pred).sum())

    return auc, y_pred_proba, mislabeled

    #out_of_bag_prediction_for_x = randforest.oob_prediction_

    #print(out_of_bag_prediction_for_x, x))
