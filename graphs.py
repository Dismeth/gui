# coding: utf-8
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from matplotlib.figure import Figure
from numpy import arange, sin, pi
from kivy.app import App

import numpy as np
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas,\
                                                NavigationToolbar2Kivy
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from matplotlib.transforms import Bbox
from kivy.uix.button import Button
from kivy.graphics import Color, Line, Rectangle

import matplotlib.pyplot as plt

## New stuff

from scipy import interp

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

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

fig,ax = plt.subplots()

for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    ax.legend()

ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

################33



ax.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')
ax.legend(loc="lower right", frameon=False)

####################



canvas = fig.canvas
# canvas.blit(Bbox(np.array([[0, 0], [400, 400]], np.int32)))


def callback(instance):
    canvas.draw()

def savefile(instance):
    print("Saving file")
    canvas.print_png("newfile1.png")

class MatplotlibTest(App):
    title = 'Matplotlib Test'

    def build(self):
        fl = BoxLayout(orientation="vertical")
        a = Button(text="Update Graph", height=40, size_hint_y=None)
        a.bind(on_press=callback)
        b = Button(text="Save File", height=40, size_hint_y=None)
        b.bind(on_press=savefile)
        #nav1 = NavigationToolbar2Kivy(canvas)
        fl.add_widget(canvas)
        fl.add_widget(a)
        fl.add_widget(b)
        return fl

if __name__ == '__main__':
    MatplotlibTest().run()

