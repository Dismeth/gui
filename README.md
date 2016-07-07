# GUI - Predicting Consumer Purchasing Decisions
made by Saavi Stubseid {sgs4@st-andrews.ac.uk}. Last Updated 07.07.2016.

This is the repository for the graphical user interface of the analysis of the data set used, in order to predict the consumer's purchasing decisions.

The application is written in Python 2.7 using Kivy 1.9.1 as GUI library. Anaconda is used as the platform for Python and the other libraries used (such as scikit-learn) preprocessing, analysing and plotting the data set.

Features (work in progress):
- Load and Save the data set.
- Label Encode the data (i.e. convert categorical string entries to numerical values in order to use e.g. Naive Bayesian Networks)
- Split the data set into training set and validation/test set.
- Settings tab to configure the following settings:
  + Label Encoding: which columns to label encode.
  + Split Data: target_column, test set size, seed (or random seed)
  + Feature Importance: which features to exclude testing (such as target_column - does not make sense to test that)
- Naive Bayesian Network. As the application is now, I can make a naive bayesian network based on the data set loaded.
- Graphics/Dataplotting. Using matplotlib the following features will return a plot:
  + Feature importance.
  + Receiver Operator Characteristics of a given model (experimental at this point).
 

Planned Features:
- More settings in order to generalise the use of data set (i.e. any data set can be loaded and analysed).
- Proper Cross Validation: k-fold CV, which can be turned on and off and adjusted in settings.
- Decision Trees (RandomForest) implementation.
- Boosting implemented (preferrably XGBoost, but I am having problems compiling on Windows).
- Make logwindow saveable and improve useability overall.

