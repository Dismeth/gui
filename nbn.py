# -*- coding: utf-8 -*-
import pandas as pd                                 # Pandas and Numpy
import numpy as np                                  #
from sklearn import ensemble                        #
from sklearn.naive_bayes import BernoulliNB         # Naive Bayesian Network - Bernoulli

from scipy import interp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

def naivebayesian(dataset):
    ds = dataset
    ds.dprint("Start Creating BernoulliNB Bayesian Network")
    bnb = BernoulliNB()
    y_pred = bnb.fit(ds.X_train, ds.y_train).predict(ds.X_test)
    mislabeled = "Number of mislabeled points out of a total %d points : %d" % (ds.X_test.shape[0] ,(ds.y_test != y_pred).sum())
    log_proba = bnb.score(ds.X_test, ds.y_test)
    return (y_pred,mislabeled,log_proba)

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


def imp_topten(dataset,columns_to_exclude,exclude_columns,target_column_name = 'target_purchase'):
    ds = dataset
    columns = list(ds.X_train.columns.values)
    # Remove the columns to to be checked.
    if exclude_columns:
        for remove_column in columns_to_exclude:
            columns.remove(remove_column)

    info = pd.DataFrame(columns=['Column', 'Total_Points', 'Mislabeled', 'Importance'])

    for index, column in enumerate(columns):
        ds.dprint("Testing feature importance of column '" + str(column) + "'.")
        bnb = BernoulliNB()
        y_pred = bnb.fit(ds.X_train[[column]], ds.y_train)
        y_pred = y_pred.predict(ds.X_test[[column]])
        """ Enter the scoring information into a DataFrame named info. """
        total_points = ds.X_test.shape[0]
        mislabeled = (ds.y_test != y_pred).sum()
        info.loc[index] = (column, total_points, mislabeled, mislabeled)

    top_ten = info.sort_values(by='Mislabeled', ascending=True)
    #top_ten = top_ten.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    top_ten = top_ten.head(n=10)
    #plt.plot(column[''], mislabeled, lw=1, label='Column %s (mislabeled points: %d)' % (column, mislabeled))
    columns = top_ten['Column']
    importance = top_ten['Mislabeled']
    y_pos = np.arange(len(columns))

    plt.barh(y_pos, importance, align='center', alpha=0.4,label='Column (mislabeled points: %0.2f)')
    plt.yticks(y_pos, columns)
    draw(type="importance")
    return top_ten

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

def computeROC():
    # Compute ROC curve and area the curve
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(ds.y_test, y_pred)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (index, roc_auc))

def draw(type="roc"):
    if type== "roc":
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
        title = 'Receiver operating characteristic example'
    elif type=="importance":
        #plt.xlim([-0.05, 1.05])
        #plt.ylim([-0.05, 1.05])
        xlabel = 'Mislabeled Predictions (less is better)'
        ylabel = 'Variable Name'
        title = 'Top 10 Features'
    else:
        title = 'Some Accurate Title - Other Type Graph'
        xlabel = 'X-LABEL'
        ylabel = 'Y-LABEL'
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.title(str(title))
    plt.legend(loc="lower right")
    plt.show()


"""
bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0] ,(y_test != y_pred).sum()))

dprint("Get the parameters")
params = bnb.get_params(deep=True)
print params

dprint("Get the log probabilitites")
log_proba = bnb.score(X_test,y_test)
print log_proba



# Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
"""