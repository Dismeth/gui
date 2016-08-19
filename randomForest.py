# -*- coding: utf-8 -*-
import pandas as pd                                 # Pandas and Numpy
import numpy as np                                  #
from sklearn import ensemble                        #

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rc('xtick', labelsize=10)
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

# Cross Validation
#from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
import random


def feature_importance_RandomForest(dataset,columns_to_exclude,exclude_columns):
    # Update v1.1: Random Forest N_estimators have been set to 10 and only top 15
    # features are shown due to space limitations.
    ds = dataset
    forest = RandomForestClassifier(n_estimators=10,max_depth=5, criterion='gini')
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
    # ds.X_train.shape[1]
    plt.bar(range(15), importances[indices][:15],
            color="g", yerr=std[indices][:15], align="center")
    plt.xticks(range(15), neworder)
    plt.xlim([-1, 15])
    plt.ylim(0,np.amax(importances)+0.1)
    plt.ylabel("Mean Decrease Impurity")
    plt.xlabel("Columns / Features")
    plt.show()
    #print(score)
    #print(score2)
    return info

"""

This is RF Feature Importance version 2 as shown in the beginning of section 7 Results and Evaluation.


NB: the novel method is further down: fi_RandomForest_improved2()
"""

def fi_RandomForest_improved(dataset, columns_to_exclude, exclude_columns, estimators=10, maximum_depth=7):
    # Update v1.1: Random Forest N_estimators have been set to 10 and only top 15
    # features are shown due to space limitations.
    ds = dataset
    forest = RandomForestClassifier(n_estimators=estimators, max_depth=maximum_depth, criterion='gini')
    dummy_forest = RandomForestClassifier(n_estimators=estimators, max_depth=maximum_depth, criterion='gini')
    columns = list(ds.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)
        ds.X_train.drop(columns_to_exclude, inplace=True, axis=1)

    dummy_y = ds.y_train.sample(frac=1).reset_index(drop=True)
    forest.fit(ds.X_train, ds.y_train)
    dummy_forest.fit(ds.X_train, dummy_y)
    importances = forest.feature_importances_
    dummy_importances = dummy_forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    dummy_std = np.std([tree.feature_importances_ for tree in dummy_forest.estimators_],
                       axis=0)
    indices = np.argsort(importances)[::-1]
    dummy_indices = np.argsort(dummy_importances)[::-1]

    columns = list(ds.X_train.columns.values)
    neworder = []
    dummy_neworder = []
    info = pd.DataFrame(columns=['Column', 'Feature Importance'])
    for f in range(ds.X_train.shape[1]):
        neworder.insert(f, columns[indices[f]])
        dummy_neworder.insert(f, columns[dummy_indices[f]])
        # print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))
        info.loc[f] = [columns[indices[f]], importances[indices[f]]]

    # Dummy variable y percent wrong:
    dummy_y_percent = ((ds.y_train != dummy_y).sum()) / float(len(ds.y_train)) * 100

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Random Forest Feature Importances (v2)")
    # ds.X_train.shape[1]
    plt.scatter(range(15), importances[indices][:15], color="g", label="Real y")
    plt.scatter(range(15), dummy_importances[dummy_indices][:15], color="r",
                label="Dummy/shuffled y (%.2f %% false)" % dummy_y_percent)
    plt.xticks(range(15), neworder)
    plt.xlim([-1, 15])
    plt.ylim(0, np.amax(importances) + 0.1)
    plt.ylabel("Mean Decrease Impurity")
    plt.xlabel("Columns / Features")
    plt.legend(loc="best")
    plt.show()

    return info

"""

Build a Random Forest, called from the main menu in the Root class (menu_randomforest()).
Room for improvements, e.g. receiving parameters that are able to be modified in the settings window.
n_estimators and max_depth should be implemented as a minimum in future work.

"""
def buildRandomForest(dataset,columns_to_exclude,exclude_columns):
    ds = dataset
    randforest = RandomForestClassifier(n_estimators=100, max_depth= 7, oob_score=True,n_jobs=-1)

    """
    The parameters that can be adjusted:
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


import seaborn as sns

sns.set(style="whitegrid", color_codes=True)


"""

This is RF Feature Importance version 2 as shown in the beginning of section 7 Results and Evaluation.


NB: the novel method is further down: fi_RandomForest_improved2()
"""

def fi_RandomForest_improved2(dataset, exclude_columns, columns_to_exclude, estimators=1, maximum_depth=2):
    # Update v1.1: Random Forest N_estimators have been set to 10 and only top 15
    # features are shown due to space limitations.
    ds = dataset

    columns = list(ds.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)
        ds.X_train.drop(columns_to_exclude, inplace=True, axis=1)

    dummy_y = ds.y_train.sample(frac=1).reset_index(drop=True)

    #Demo purposes!
    #number_times = 100
    number_times = 10
    columns = list(ds.X_train.columns.values)
    info = pd.DataFrame(columns=['Column', 'Value', 'Dummy'])
    # dummy_info = pd.DataFrame(columns=['Column', 'Value'])
    total_columns = pd.Series()

    for i in range(number_times):
        forest = RandomForestClassifier(n_estimators=estimators, max_depth=maximum_depth, criterion='gini', n_jobs=-1)
        dummy_forest = RandomForestClassifier(n_estimators=estimators, max_depth=maximum_depth, criterion='gini',
                                              n_jobs=-1)

        random_cols = pd.Series(random.sample(columns, 3))
        total_columns = total_columns.append(random_cols)

        forest.fit(ds.X_train[random_cols], ds.y_train)
        dummy_forest.fit(ds.X_train[random_cols], dummy_y)

        importances = forest.feature_importances_
        dummy_importances = dummy_forest.feature_importances_

        # std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
        # dummy_std = np.std([tree.feature_importances_ for tree in dummy_forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        dummy_indices = np.argsort(dummy_importances)[::-1]

        neworder = []
        dummy_neworder = []

        for f in range(3):
            neworder.insert(f, random_cols[indices[f]])
            dummy_neworder.insert(f, random_cols[dummy_indices[f]])
            # print("%d. feature %s (%f)" % (f + 1, columns[indices[f]], importances[indices[f]]))

        info.loc[len(info)] = [neworder[0], importances[indices[0]], 0]
        info.loc[len(info)] = [neworder[0], dummy_importances[indices[0]], 1]

    # Dummy variable y percent wrong:
    dummy_y_percent = ((ds.y_train != dummy_y).sum()) / float(len(ds.y_train)) * 100

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("RF Feature Importances (v3) - %.2f%% error in Dummy-set" % dummy_y_percent)
    sns.stripplot(x="Column", y="Value", data=info, hue="Dummy", jitter=True, size=10)
    # sns.stripplot(x="Column", y="Value", data=dummy_info, jitter=True, color = sns.color_palette()[2],size=10)
    # sns.boxplot(x="Column", y="Value", data=info, hue="Dummy");
    plt.ylabel("Relative Feature Importance Score")
    plt.xlabel("Columns / Features")

    plt.show()
    return info, total_columns, dummy_y_percent