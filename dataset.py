# -*- coding: utf-8 -*-
# Module "dataset.py"
# Made by Såvi Stubseid
# User to import, label encode and split a given dataset for training purposes.

import pandas as pd                                 # Pandas
from sklearn import preprocessing                   # Using label encoder to get strings (categories) into numeric values for Bayesian Network
from sklearn import cross_validation                # K-Fold and cross validation

"""
THIS IS TEMP FOR OWN USAGE:

    data = dataset.dataimport("D:\Dropbox\St Andrews\IT\IS5189 MSc Thesis\\02 Data\InnoCentive_Challenge_9933493_training_data.csv")
    string_columns = ['C2','C4','C5','C6','C9','C11','C13','C15','C16','C17','C19','C20','C21','C22','C28','C30','C53','C60']
    data = dataset.labelencode(dataset = data,columns=string_columns)
    data = dataset.split(data)

"""
class DataSet():

    def __init__(self):
        self.numberoftime = 1
        self.loaded = False
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
    """
    dprint() - Making a simple function to number the output for debugging purposes. Added incremental numbers for readability.
    Req: none
    """
    def dprint(self,message):
        print str(self.numberoftime) + ": " + message
        self.numberoftime = self.numberoftime + 1

    def exists(self):
        return True

    """
    Import the csv file using pandas.
    Req: pandas.
    """
    def dataimport(self, filelocation):
        filelocation = str(filelocation)
        self.dataset = pd.read_csv(filelocation)
        self.loaded = True
        return self.dataset

    def get_dataset(self):
        if self.loaded:
            return self.dataset
        else:
            return 0
    # columns = ['C2','C4','C5','C6','C9','C11','C13','C15','C16','C17','C19','C20','C21','C22','C28','C30','C53','C60']

    """
    Label Encode the given columns (containing strings/chars) in the imported data set.
    Req: preprocessing from sklearn.
    """
    def labelencode(self,columns):
        le = preprocessing.LabelEncoder()
        for col in columns:
            le.fit(self.dataset[col])
            encoded_column = le.transform(self.dataset[col])
            # del trainset[column] # We maintain the column/variable structure by simply replacing the values rather than deleting them.
            self.dataset[col] = encoded_column
        return self.dataset

    """
    Split the data set into a training set and validation set. It also splits it into x (input) and y (target/output).
    Req: cross_validation from sklearn.
    """
    def split(self,target_column_name = 'target_purchase', test_set_size = 0.4, random_state_is = True):
        target_value = self.dataset[target_column_name]
        if random_state_is:
            self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.dataset, target_value, test_size=test_set_size)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.dataset,
                                                                                                     target_value,
                                                                                                     test_size=test_set_size,
                                                                                                     random_state=random_state_is)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def save_file(self, path, filename):
        filepath = str(path) + '\\' + str(filename)
        pd.DataFrame.to_csv(self.dataset, path_or_buf=filepath)