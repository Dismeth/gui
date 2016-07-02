# coding: utf-8
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

def example1():
    ###############################################################################
    # Data IO and generation

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    ###############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(y, n_folds=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    return (mean_tpr, mean_fpr, cv)

def make_graph():
    return 0

def draw(type="roc"):
    if type== "roc":
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        xlabel = 'False Positive Rate'
        ylabel = 'True Positive Rate'
        title = 'Receiver operating characteristic example'
    else:
        title = 'Some Accurate Title - Other Type Graph'
        xlabel = 'X-LABEL'
        ylabel = 'Y-LABEL'
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.title(str(title))
    plt.legend(loc="lower right")
    plt.show()

mean_tpr, mean_fpr, cv = example1()
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

draw(type="not")

