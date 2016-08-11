"""

A greedy feature selection made by Stubseid using BernoulliNB.
The class is adapted from 'Greedy Feature Selection using Logistic Regression'
to optimize Area Under the ROC Curve by Abhishek.

Credits :
- Abhishek
- Miroslaw @ Kaggle

"""

import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics, preprocessing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("white")


class greedyFF(object):
    def __init__(self, data, labels, scale=1, verbose=0):
        if scale == 1:
            self._data = preprocessing.scale(np.array(data))
        else:
            self._data = np.array(data)
        self._labels = labels
        self._verbose = verbose
        self._total_auc = []
        self._num_feat = []
        self._good_feat = []

    def evaluateScore(self, X, y):
        model = BernoulliNB()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc

    def selectionLoop(self, X, y):
        score_history = []
        good_features = set([])
        num_features = X.shape[1]
        while len(score_history) < 2 or score_history[-1][0] > score_history[-2][0]:
            scores = []
            for feature in range(num_features):
                if feature not in good_features:
                    selected_features = list(good_features) + [feature]

                    Xts = np.column_stack(X[:, j] for j in selected_features)

                    score = self.evaluateScore(Xts, y)
                    scores.append((score, feature))

                    if self._verbose:
                        print "Current AUC : ", np.mean(score)

            good_features.add(sorted(scores)[-1][1])
            score_history.append(sorted(scores)[-1])
            self._total_auc.append(sorted(scores)[-1])
            self._num_feat.append(len(good_features))
            if self._verbose:
                print "Current Features : ", sorted(list(good_features))

        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = sorted(list(good_features))
        if self._verbose:
            print "Selected Features : ", good_features

        return good_features

    def transform(self):
        X = self._data
        y = self._labels
        good_features = self.selectionLoop(X, y)
        self._good_feat = good_features
        return X[:, good_features]

    def get_features(self):
        return self._good_feat

    def plot(self):
        scr = []
        for nr in self._total_auc:
            scr.append(nr[0])

        plt.figure(figsize=(8, 5))
        plt.plot(self._num_feat, scr, c=sns.color_palette()[0])
        plt.axis('tight')
        plt.xlabel('Number of Features')
        plt.ylabel('AUC')
        plt.ylim((min(scr) - 0.02, max(scr) + 0.02))
        plt.xlim((0.0, max(self._num_feat)))
        plt.title("AUC versus Number of Features", fontsize=16.)
        plt.show()