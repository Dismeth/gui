# -*- coding: utf-8 -*-
import pandas as pd                                 # Pandas and Numpy
import numpy as np                                  #
from sklearn import ensemble                        #
from sklearn.naive_bayes import BernoulliNB         # Naive Bayesian Network - Bernoulli

def naivebayesian(dataset):
    ds = dataset
    ds.dprint("Start Creating BernoulliNB Bayesian Network")
    bnb = BernoulliNB()
    y_pred = bnb.fit(ds.X_train, ds.y_train).predict(ds.X_test)
    mislabeled = "Number of mislabeled points out of a total %d points : %d" % (ds.X_test.shape[0] ,(ds.y_test != y_pred).sum())
    log_proba = bnb.score(ds.X_test, ds.y_test)
    return (y_pred,mislabeled,log_proba)


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
"""