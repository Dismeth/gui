# coding: utf-8

import pandas as pd                                 # Pandas
import numpy as np
from sklearn import preprocessing                   # Using label encoder to get strings (categories) into numeric values for Bayesian Network
from sklearn import cross_validation                # K-Fold and cross validation
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve

# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 # only handles positive data
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler



from dataset import DataSet
from tqdm import tqdm                               #

########
#
#
#
###########


# Global variables
categorical_columns = ['C2', 'C4', 'C5', 'C6', 'C9', 'C11', 'C13', 'C15', 'C16', 'C17', 'C19', 'C20', 'C21', 'C22', 'C28', 'C30', 'C53', 'C60']
target_value = 'target_purchase'
drop_columns = ['record_ID']
test_set_size = 0.2


def load():
    data = DataSet()
    data.dprint("Import data.")
    data.dataimport("D:\Dropbox\St Andrews\IT\IS5189 MSc Thesis\\02 Data\InnoCentive_Challenge_9933493_training_data.csv")
    data.dprint("Label Encode.")
    data.labelencode(columns=categorical_columns)
    data.dprint("Split data. Test set size: " + str(test_set_size))
    data.split(target_column_name=target_value, test_set_size=test_set_size,random_state_is=True)
    return data



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def report_estimator(estimator, X, Y):
    print("Reporting: ")


if __name__ == '__main__':
    ds = load()
    run_function = 2
    if run_function == 1:
        ds.dprint("Executing run_function: " + str(run_function))
        title = "Learning Curves (Naive Bayes)"
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = cross_validation.ShuffleSplit(ds.X_train.shape[0], n_iter=100,
                                       test_size=0.2, random_state=0)

        estimator = BernoulliNB()
        plot_learning_curve(estimator, title, ds.X_train, ds.y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

        title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = cross_validation.ShuffleSplit(ds.X_train.shape[0], n_iter=10,
                                       test_size=0.2, random_state=0)
        estimator = SVC(gamma=0.001)
        plot_learning_curve(estimator, title, ds.X_train, ds.y_train, (0.7, 1.01), cv=cv, n_jobs=4)

        plt.show()
    elif run_function == 2:
        ds.dprint("Executing run_function: " + str(run_function))
        # ---- Preprocessing ---- #
        skb = SelectKBest(f_classif, k=20)
        new_X_train = skb.fit_transform(ds.X_train,ds.y_train)
        mask = skb.get_support(True)
        mask_cols = list(ds.X_test.iloc[:, mask].columns)
        new_X_test = ds.X_test[mask_cols]
        # ---- Start New Classifier   #

        ds.dprint("Making new classifier")
        estimator = RandomForestClassifier(n_estimators=120,max_depth=5)
        estimator.fit(ds.X_train,ds.y_train)
        pred_y1 = estimator.predict(ds.X_test)
        pred_y_train = estimator.predict(ds.X_train)
        pred_y_train_proba = estimator.predict_proba(ds.X_train)
        points1 = ds.y_test.shape[0]
        mislabeled1 = (ds.y_test != pred_y1).sum()
        print(points1)
        print(mislabeled1)
        # ---- Start New Classifier   #

        scaler = MinMaxScaler(feature_range=(0, 1))
        new_X_train = scaler.fit_transform(new_X_train)
        new_X_test = scaler.fit_transform(new_X_test)

        ds.dprint("Making new classifier 3")
        estimator2 = RandomForestClassifier(n_estimators=120, max_depth=5)
        estimator2.fit(new_X_train, ds.y_train)
        pred_y2 = estimator2.predict(new_X_test)
        pred_y2_proba = estimator2.predict_proba(new_X_test)
        mislabeled2 = (ds.y_test != pred_y2).sum()
        print(mislabeled2)

        #scores = cross_val_score(new_est, new_X_train, ds.y_train)
        #print(scores)


