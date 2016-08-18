# -*- coding: utf-8 -*-
import pandas as pd                                 # Pandas and Numpy
import numpy as np                                  #
from sklearn import ensemble                        #
from sklearn.naive_bayes import BernoulliNB         # Naive Bayesian Network - Bernoulli
#from sklearn.naive_bayes import MultinomialNB       # Testing..

from scipy import interp
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def naivebayesian(dataset,configFIUse,configFI,alpha,binerize):
    ds = dataset
    ds.dprint("Start Creating BernoulliNB Bayesian Network")
    if configFIUse:
        ds.dprint("Excluding following columns: " + str(configFI))
        X_train = ds.X_train.drop(configFI, inplace=False, axis=1)
        X_test = ds.X_test.drop(configFI, inplace=False, axis=1)
    else:
        X_train = ds.X_train
        X_test = ds.X_test
    bnb = BernoulliNB(alpha=alpha, binarize=binerize)
    y_pred = bnb.fit(X_train, ds.y_train).predict(X_test)
    y_pred_proba = bnb.predict_proba(X_test)
    mislabeled = "Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0] ,(ds.y_test != y_pred).sum())
    auc = roc_auc_score(ds.y_test, y_pred_proba[:, 1])
    return (mislabeled,auc,y_pred_proba)

def nbnfeatureimportance(dataset, column_to_test, target_column_name = 'target_purchase'):
    ds = dataset
    ds.dprint("Testing feature importance of column '" + str(column_to_test) + "'.")
    bnb = BernoulliNB()
    y_pred = bnb.fit(ds.X_train[[column_to_test]], ds.y_train).predict(ds.X_test[[column_to_test]])
    total_points = ds.X_test.shape[0]
    mislabeled = (ds.y_test != y_pred).sum()
    percentage = (mislabeled / total_points)

    return (column_to_test,total_points, mislabeled, percentage)

def findtopfeatures(dataset,columns_to_exclude,exclude_columns,target_column_name = 'target_purchase'):
    columns = list(dataset.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)
    info = pd.DataFrame(columns=['Column','Total_Points','Mislabeled','Percentage'])
    for index, column in enumerate(columns):
        info.loc[index] = nbnfeatureimportance(dataset, column)
    return info


def imp_topten(dataset,exclude_columns,columns_to_exclude,target_column_name = 'target_purchase'):
    ds = dataset
    columns = list(ds.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)

    info = pd.DataFrame(columns=['Column', 'Total_Points', 'Mislabeled', 'Importance'])

    for index, column in enumerate(columns):
        bnb = BernoulliNB()
        y_pred = bnb.fit(ds.X_train[[column]], ds.y_train)
        y_pred = y_pred.predict(ds.X_test[[column]])
        """ Enter the scoring information into a DataFrame named info. """
        total_points = ds.X_test.shape[0]
        mislabeled = (ds.y_test != y_pred).sum()
        info.loc[index] = (column, total_points, mislabeled, mislabeled)
        ds.dprint("FI Column '" + str(column) + "'.: " + str(mislabeled))

    top_ten = info.sort_values(by='Mislabeled', ascending=True)
    #top_ten = top_ten.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    top_ten = top_ten.head(n=10)
    #plt.plot(column[''], mislabeled, lw=1, label='Column %s (mislabeled points: %d)' % (column, mislabeled))
    columns = top_ten['Column']
    importance = top_ten['Mislabeled']
    y_pos = np.arange(len(columns))

    string = """
    ======== ================
    **Col**   **Mislabeled**
    """
    for index, row in top_ten.iterrows():
        string += """
    -------- ----------------
     %s      %0.0f
    """ % (row['Column'], row['Mislabeled'])

    string += """
    ======== ================
    """
    return top_ten, string

"""
Function analysing the smaller dataset in order to find important features
"""

def quickanalyse(dataset,columns_to_exclude,exclude_columns,target_column_name = 'target_purchase'):
    ds = dataset
    targ = target_column_name
    columns = list(ds.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)

    info = pd.DataFrame(columns=['Column', 'Pearson', 'Mislabeled', 'Importance'])
    for index, column in enumerate(columns):
        info.loc[index] = (column,pearsonr(ds.X_train[[column]],ds.y_train),0,0)

    return info
