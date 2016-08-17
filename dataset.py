# -*- coding: utf-8 -*-
# Module "dataset.py"
# Made by Såvi Stubseid
# User to import, label encode and split a given dataset for training purposes.

import pandas as pd                                 # Pandas
from sklearn import preprocessing                   # Using label encoder to get strings (categories) into numeric values for Bayesian Network
from sklearn import cross_validation                # K-Fold and cross validation
from scipy.stats import pearsonr
from sklearn.cross_validation import StratifiedShuffleSplit

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
        self.X_train = None
        #self.X_test = pd.DataFrame()
        #self.y_train = pd.DataFrame()
        #self.y_test = pd.DataFrame()
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
            return 1
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
    Scale the numerical columns using RobustScaler() from sklearn.preprocessing, which is more robust
    to outliers.
    Req: preprocessing from sklearn.
    """
    def scale(self,columns,categorical_cols,apply_list,target_column):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()

        if apply_list:
            numerical_cols = columns
        else:
            numerical_cols = []
            for col in self.dataset.columns.values:
                if col not in categorical_cols:
                    numerical_cols.append(col)
                else:
                    pass
        # We don't want to scale the target variable, as it is already binary.
        # The target column uses the same value as target_value from Split Data section
        # in the settings popup.
        numerical_cols.remove(target_column)
        # Scale, fit and transform all the numerical columns
        scaled_data = scaler.fit_transform(self.dataset[numerical_cols])
        self.dataset[numerical_cols] = scaled_data
        return self.dataset

    """
    Split the data set into a training set and validation set. It also splits it into x (input) and y (target/output).
    Req: cross_validation from sklearn.
    Update 0.5: StratifiedShuffleSplit has been used in order to ensure the same label distribution
    """
    def split(self,target_column_name = 'target_purchase', test_set_size = 0.4, seed=16, random_state_is = True, quick = False, quick_test_size = 0.9):
        target_value = self.dataset[target_column_name]
        if quick:
            test_set_size = quick_test_size
        if random_state_is:
            #self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.dataset, target_value, test_size=test_set_size)
            y = self.dataset[target_column_name]
            x = self.dataset
            x = x.drop('record_ID', 1)
            x = x.drop('target_purchase', 1)
            sss = StratifiedShuffleSplit(y, 2, test_size=test_set_size)
            for train_index, test_index in sss:
                #print("TRAIN:", train_index, "TEST:", test_index)
                self.X_train, self.X_test = x.iloc[train_index], x.iloc[test_index]
                self.y_train, self.y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = cross_validation.train_test_split(self.dataset,
                                                                                                     target_value,
                                                                                                     test_size=test_set_size,
                                                                                                     random_state=seed)
        # Drop the record ID and target_purchase variables. This must be generalised!
        #self.X_train = self.X_train.drop('record_ID', 1)
        #self.X_train = self.X_train.drop('target_purchase', 1)
        #self.X_test = self.X_test.drop('record_ID', 1)
        #self.X_test = self.X_test.drop('target_purchase', 1)
        return self.X_train, self.X_test, self.y_train, self.y_test

    """
    Used to save time
    """
    def import_split(self,xtrain,xtest,ytrain,ytest):
        self.X_train = xtrain
        self.X_test = xtest
        self.y_train = ytrain
        self.y_test = ytest
        return self.X_train, self.X_test, self.y_train, self.y_test


    """
    Split the dataset based on
    """

    def save_file(self, path, filename):
        filepath = str(path) + '\\' + str(filename)
        pd.DataFrame.to_csv(self.dataset, path_or_buf=filepath)

    def information(self):
        if self.loaded:
            a = list(self.dataset.columns.values)
            #b = list(self.X_test.columns.values)
            #c = self.X_test.columns.values.tolist()
            return a
        else:
            return 0

    def correlation(self,check_recommendation=True,columns_to_check=None):
        # V1.1 added support for columns_to_check to check only certain columns (for FS).
        # added support for workdir - but in main.py (Root class)
        self.loaded = True
        #check_recommendation = False
        if self.loaded:
            # used_columns is used to 'solve' the handshake problem. The total number of
            # items in correlation_list should be : n(n-1)/2 where n = len(self.X_train.columns.values).
            used_columns = []
            target_corr = []
            correlation_list = pd.DataFrame(columns=['Var1', 'Var2','Correlation','P-Value','TargCorr'])
            descstats_list = pd.DataFrame(columns=['Var', 'Mean','SumOfValues','Minimum','Maximum','Count','Std'])
            i = 0
            ii = 0
            #
            # New feature to be used outside the GUI.. Now we can check correlations
            # between selected columns to increase Bayes performance.
            #
            if columns_to_check == None:
                columns_values = self.X_train.columns.values
            else:
                columns_values = columns_to_check

            for column_a in columns_values: # columns_values instead of self.X_train.columns.values.
                var = self.X_train[column_a]
                descstats_list.loc[ii] = [column_a, var.mean(), var.sum(), var.min(), var.max(), var.count(), var.std()]
                for column_b in columns_values:
                    if column_a == column_b or column_b in used_columns:
                        pass
                    else:
                        corr, p = pearsonr(self.X_train[column_a],self.X_train[column_b])
                        t_corr, pp = pearsonr(self.X_train[column_a],self.y_train)
                        correlation_list.loc[i] = [column_a, column_b, corr, p,t_corr]
                        i += 1
                used_columns.insert(i,column_a)
                ii += 1
            if check_recommendation:
                # df is short for DataFrame , to make it more readable when manipulating the Pandas DataFrame.
                # Might be easier (and is shorter) to read by developers as an in house var name.
                threshold = 0.1
                df = correlation_list[abs(correlation_list['TargCorr']) >= threshold]
                df = df.sort_values(by='TargCorr',ascending=False)


            ######
            ###### Exporting information to CSV is done in root class.
            ######
            #fileloc = "D:\Dropbox\St Andrews\IT\IS5189 MSc Thesis\\02 Data\\"
            #pd.DataFrame.to_csv(correlation_list,fileloc + "Correlations.csv")
            #pd.DataFrame.to_csv(self.X_train,fileloc + "X_Train.csv")
            return correlation_list, descstats_list, df

    def descstats(self,columns_to_exclude = ['record_ID','target_purchase'],write = False,workdir = 'C:\\'):
        if self.loaded:
            data = self.dataset
            descstats_list = pd.DataFrame(columns=['Var', 'Mean', 'SumOfValues', 'Minimum', 'Maximum', 'Count', 'Std'])
            for i, column_a in enumerate(data.columns.values):
                if column_a in columns_to_exclude:
                    pass
                else:
                    var = data[column_a]
                    descstats_list.loc[i] = [column_a, var.mean(), var.sum(), var.min(), var.max(), var.count(), var.std()]
            if write:
                filename = "DescStats.csv"
                fileloc = workdir + filename
                pd.DataFrame.to_csv(descstats_list, fileloc)
            return descstats_list
    def savedata(self,nrows = 1000, workdir = 'C:\\',target_value = 'target_purchase'):
        if self.loaded:
            train, test = cross_validation.train_test_split(self.dataset,test_size=0.1)
            pd.DataFrame.to_csv(test, workdir + "SmallDataSet.csv")
        else:
            return 0