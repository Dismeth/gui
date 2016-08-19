from kivy.config import Config
Config.set('graphics', 'width', '1366')
Config.set('graphics', 'height', '768')
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import NumericProperty
from kivy.properties import DictProperty
from kivy.properties import ListProperty
from kivy.properties import BooleanProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.switch import Switch
from dataset import DataSet
from greedyff import greedyFF
#some stuff
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
import numpy as np
#import graphs
#some stuff end
import randomForest
import nbn
import xgboost_model
import datetime
import time
import timeit
import pandas as pd
import os


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SplitData(FloatLayout):
    split = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Settings(FloatLayout):
    settings_save = ObjectProperty(None)
    cancel = ObjectProperty(None)
    #initialisesettings = ObjectProperty(None)
    newconfigGeneral = DictProperty({})
    newconfigCV = DictProperty({})
    newconfigLE = StringProperty(None)
    newconfigScale = StringProperty(None)
    newconfigFI = StringProperty(None)
    newconfigFIUse = BooleanProperty(None)

    """
    Functions to save all the settings from the settings window.
    """
    def saveGeneral(self, desc_stats_on_load, workdir,apply_scale):
        self.newconfigGeneral['desc_stats_on_load'] = bool(desc_stats_on_load)
        self.newconfigGeneral['workdir'] = str(workdir)
        self.newconfigGeneral['apply_scale'] = bool(apply_scale)
        return self.newconfigGeneral

    def saveLE(self,le_columns):
        self.newconfigLE = le_columns
        return self.newconfigLE

    def saveScale(self, scale_columns):
        self.newconfigScale = scale_columns
        return self.newconfigScale

    def saveCV(self, target_value, test_set_size, seed, random_state):
        self.newconfigCV['target_value'] = str(target_value)
        self.newconfigCV['test_set_size'] = float(test_set_size)
        self.newconfigCV['seed'] = int(seed)
        self.newconfigCV['random_state_is'] = bool(random_state)
        return self.newconfigCV

    def saveFI(self, feature_importance):
        self.newconfigFI = str(feature_importance)
        return self.newconfigFI

    def saveFIUse(self,apply_FI_list):
        self.newconfigFIUse = bool(apply_FI_list)
        return self.newconfigFIUse

class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    #splitfile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    # Define variables to be updated
    output_console = StringProperty(None)
    output_text = StringProperty(None)
    output_label = StringProperty(None)
    # Overview variables that need to be updated:
    data_loaded = StringProperty(None)
    fname = StringProperty(None)
    trainrows = StringProperty(None)
    testrows = StringProperty(None)
    ncols = StringProperty(None)
    best_score = StringProperty(None)

    def initialiseSettings(self):
        self.configGeneral = {'desc_stats_on_load': True,'workdir': 'D:\workdir\\','apply_scale': False}
        self.configCV = {'target_value': 'target_purchase', 'test_set_size': 0.4, 'seed': 16, 'random_state_is': True}
        self.configLE = ['C2', 'C4', 'C5', 'C6', 'C9', 'C11', 'C13', 'C15', 'C16', 'C17', 'C19', 'C20', 'C21', 'C22', 'C28', 'C30', 'C53', 'C60']
        self.configScale = ['C72','C73']
        self.configFI = ['C2'] #example columns to exclude
        self.configFIUse = False # This has been updated.. Need to figure out how to control this one. (because record_ID and target_purchase is included.)
        self.dataOverview = {'data_loaded': False, 'fname': 'N/A', 'trainrows': 0, 'testrows': 0,'ncols': 0, 'best_score': 0}
        self.feedback("Settings has been reset.")
        Config.set('graphics', 'width', '1366')
        Config.set('graphics', 'height', '768')

    def __init__(self,**kwargs):
        super(Root, self).__init__(**kwargs)
        """ Global variables to be edited in the settings """
        self.loaded = False
        self.split = False
        self.initialiseSettings()
        self._update_overviewGUI()
        """ Set up logging """
        self.output_text = ""
        self.output_log = ""
        self.output_last = "In depth function output is displayed here."
        self.log_toggled = False
        self.menu_toggle_main_output()
        """ Welcome message etc """
        version = 1.1
        welcome = "Welcome user. This is version " + str(version) + ". Click on Load to start."
        self.feedback(welcome)
        self.output_str("Recommended settings loaded. Open 'Settings' using top right button. Application ready to load data.")
        """
        Open Settings first time, needed a delay in order to wait for everything to be properly set.
        """
        #Clock.schedule_once(self.show_startup_settings, 1)

    """
    Function to dismiss the popups (Settings, Load and Save)
    """

    def dismiss_popup(self):
        self._popup.dismiss()

    """
    show_load(): The function called in the menu. Shows the popup.
    """

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    """
    show_save(): The function called in the menu. Shows the popup.
    """

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    """
    load() shows a file browser to find an appropriate data set (csv format).
    """
    def load(self, path, filename):
        global last_path
        global last_filename
        global data
        last_path = path
        last_filename  = filename[0]
        try:
            data = DataSet()
            data.dataimport(filename[0])
            self.loaded = True
        except (RuntimeError, TypeError, NameError):
            data.dprint("Error: most likely not a csv file.")
        self.output_str("Successfully loaded the data set.")
        self.feedback("Fileimport completed")
        if self.configGeneral['desc_stats_on_load']:
            data.descstats(self.configLE)
            self.output_str("Descriptive statistics performed.")
        ncols = len(data.information())
        # Get the filename and cut it to fit the GUI..
        # Filename only used to remind the user of which dataset has been loaded.
        head, tail = os.path.split(filename[0])
        fname = tail[:5]+ "." + tail[-4:]
        self.update_overview(fname=fname,ncols=ncols)
        self.dismiss_popup()

    """
    save() need to fix this one. Ultimately dataset.save_file(dataset,path)
    """
    def save(self, path, filename):
        if not self.loaded:
            self.feedback("You have nothing to save.")
        else:
            filepath = str(path) + '\\' + str(filename)
            data.dprint(str(filepath))
            data.save_file(str(path),str(filename))
        self.dismiss_popup()

    """

    Updates the variables on the dashboard (data overview).

    """

    def update_overview(self, fname = 0, trainrows = 0, testrows=0, ncols = 0, best_score = False):
        self.dataOverview['data_loaded'] = self.loaded
        if not fname == 0:
            self.dataOverview['fname'] = fname
        if not trainrows == 0:
            self.dataOverview['trainrows'] = trainrows
        if not testrows == 0:
            self.dataOverview['testrows'] = testrows
        if not ncols == 0:
            self.dataOverview['ncols'] = ncols
        if not best_score == False:
            if best_score > self.dataOverview['best_score']:
                self.dataOverview['best_score'] = best_score
        self._update_overviewGUI()

    """

    Pushes the updates so the GUI gets the latest information.

    """

    def _update_overviewGUI(self):
        self.data_loaded = str(self.dataOverview['data_loaded'])
        self.fname = str(self.dataOverview['fname'])
        self.trainrows = str(self.dataOverview['trainrows'])
        self.testrows = str(self.dataOverview['testrows'])
        self.ncols = str(self.dataOverview['ncols'])
        self.best_score = str(self.dataOverview['best_score'])

    """
    Transforms a list used as a configuration to a string, in order to be easilier edited using semicolon as a splitter.
    Semicolon used as a separator.
    """
    def transform_config(self,config,back=False):
        if back:
            newconfig = config.split(";") # Semicolon used as separator. Can be changed.
        else:
            firstline = True
            newconfig = ""
            for conf in config:
                if firstline:
                    newconfig += conf.upper() # .upper() should be removed in future..
                    firstline = False
                else:
                    newconfig += ";" + conf.upper() # Semicolon used as separator. Can be changed.
        return newconfig

    """
    Dummy function to enable the settings_popup to show at start.
    """
    def show_startup_settings(self, dt):
        self.show_settings()

    """
    Function to call the settings_popup.
    """
    def show_settings(self):
        content = Settings(settings_save=self.settings_save, cancel=self.dismiss_popup,
                           newconfigGeneral=self.configGeneral,
                           newconfigCV=self.configCV, newconfigScale=self.transform_config(self.configScale), newconfigLE=self.transform_config(self.configLE),
                           newconfigFI=self.transform_config(self.configFI), newconfigFIUse=self.configFIUse)
        self._popup = Popup(title="Data Analysis Settings", content=content, size_hint=(0.9, 0.9))
        self._popup.open()


    """
    Save Settings. Called when closing the settings_popup.
    """
    def settings_save(self, updated_configGeneral, updated_configle, updated_configScale, updated_configcv, updated_configfi, updated_use_fi):
        # Updates the config
        self.configGeneral = updated_configGeneral
        self.configCV = updated_configcv
        self.configLE = self.transform_config(updated_configle,back=True)
        self.configScale = self.transform_config(updated_configScale,back=True)
        self.configFI = self.transform_config(updated_configfi,back=True)
        self.configFIUse = updated_use_fi
        # Gives a feedback to the "console".
        self.feedback("Settings Saved")
        # Prints out the updated configs.. Uncomment for debugging.
        """
        self.output_str(updated_configcv)
        self.output_str(self.configLE)
        self.output_str(self.configFI)
        self.output_str(updated_use_fi)
        """
        # Dismiss popup..
        self.dismiss_popup()

    """
    Write output from functions to workdir.
        -   Expects DataFrame.
        -   Returns: file location.
    """

    def write_output(self,result_to_write,filename):
        # filename = "SomeFileName.csv"
        fileloc = self.configGeneral['workdir'] + filename
        pd.DataFrame.to_csv(result_to_write, fileloc)
        return fileloc

    """
    Function to export 10 % of the dataset randomly, in order to analyse by hand.
    """

    def menu_export_small_dataset(self):
        if self.loaded:
            data.savedata(workdir=self.configGeneral['workdir'])
            self.feedback("Randomly selected rows have been exported to workdir.")
        else:
            self.feedback("Please load a dataset.")

    """

    Now settings have been initialised and the popup-overhead is done.
    This is vere the data pre-processing, analysing and model generation starts.
    -----------
    functions starting with menu_ are found on the left menu.
    """

    """
    Label Encodes (categorical variables are converted to numerical values) the data set.
    """

    def menu_label_encode(self):
        if self.loaded:
            #Clock.schedule_once(lambda dt: self.feedback("Start label encoded the data."), -1)
            self.output_str("Start label encoded the data.")
            data.labelencode(columns=self.configLE)
            self.output_str("End label encoded the data.")
            self.feedback("Succesfully label encoded the data.")
        else:
            self.feedback("Please load a dataset.")

    def menu_scale(self):
        if self.loaded:
            # Clock.schedule_once(lambda dt: self.feedback("Start label encoded the data."), -1)
            self.output_str("Start scaling numerical values.")
            data.scale(columns=self.configScale,categorical_cols=self.configLE,apply_list=self.configGeneral['apply_scale'],target_column=self.configCV['target_value'])
            self.output_str("End scaling numerical values.")
            self.feedback("Succesfully label encoded the data.")
        else:
            self.feedback("Please load a dataset.")



    """
    function split_data(): splits the data into a training set and validation set with user's options.
    """
    def menu_split_data(self):
        if self.loaded:
            self.output_str("Start splitting the data.")
            data.split(target_column_name = self.configCV['target_value'], test_set_size = self.configCV['test_set_size'], seed = self.configCV['seed'], random_state_is = self.configCV['random_state_is'])
            self.output_str("Finished splitting the data.")
            self.feedback("Succesfully split the data.")
            self.update_overview(trainrows=len(data.X_train),testrows=len(data.X_test),ncols=len(data.X_train.columns.values))
            self.split = True
        else:
            self.feedback("Please load a dataset.")







    """
    Coarse Statistics
    """

    def menu_coarse_statistics(self):
        if self.loaded:
            desc = data.descstats(write=False)
            self.output_str("Coarse statistics finished.")
            self._output_last(desc)
            location = self.write_output(desc, filename='Coarse-Statistics.csv')
            self.feedback("Descriptive statistics calculated. Results written to file: " + str(location))
        else:
            self.feedback('Please load a dataset.')

    """
    Variable Correlations
    """
    def menu_var_corr(self):
        if self.loaded & self.split:
            correlation, descstats, df = data.correlation()
            location = self.write_output(df,"Correlations.csv")
            self._output_last(df)
            column_a_b = df['Var1']
            column_a_b = column_a_b.append(df['Var2'])
            top_vars = column_a_b.value_counts()

            string = """
======  ======  =======  ======
Var1    Var2     Corr     p
            """
            for index, row in df.iterrows():
                string += """
------  ------  -------  ------
%s       %s    %0.3f     %0.1f
                """ % (row['Var1'],row['Var2'],row['Correlation'],row['P-Value'])
            string += """
======  ======  =======  ======
            """
            self._output_last(string)
            self.feedback("Variable correlations calculated. Results written to file: " + location)
        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    """

    Find the best features leading to a better AUC with NBN

    """

    def menu_greedyff(self):
        if self.loaded & self.split:
            greedysearch = greedyFF(data.X_train,data.y_train, verbose=1)
            greedysearch.transform()
            self.output_str(greedysearch.get_features())
            greedysearch.plot()
            self.feedback("Succesfully made a Naive Bayesian Network.")
            # Creating the output from the function

        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    """
    Build a shallow RandomForest classifier and return feature importances.
    """
    def menu_feature_importance(self):
        if self.loaded & self.split:
            # Please adjust the estimators and maximum_depth for faster/instable or slower/stable
            # feature importance!
            info = randomForest.fi_RandomForest_improved(data, self.configFI, self.configFIUse, estimators=10, maximum_depth=7)
            #new_info = ""
            #for word in info.to_string():
            #    new_info += word
            #    new_info += "\r \n"
            #self._output_last(new_info)
            self.output_str("Random Forest Feature Importance finished (v2). Swap view for more information, or see workdir.")
            location = self.write_output(info, filename='Feature-Importance.csv')
            self.feedback("Feature importance calculated. Results written to file: " + str(location))
        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    """

    Create a Naive Bayesian Network Model

    """

    def menu_nbn(self):
        if self.loaded & self.split:
            alpha = 4
            binerize = 1.0
            output,auc,pred_proba = nbn.naivebayesian(data,self.configFIUse,self.configFI,alpha,binerize)
            #self.output_str(ypred)
            self.feedback("Succesfully made a Naive Bayesian Network.")
            #self.output_str(pred_proba)
            self.performance_report("Naive Bayesian Network",pred_proba,data.y_test)
            self.output_str(output)

            self.update_overview(best_score=float(format(auc, '.3f')))
            # Creating the output from the function

            # Showing ROC plot
            #self._roc_plot(y_proba=pred_proba, y=data.y_test,auc=auc, model='Naive Bayesian Network')

        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    """
    Build a RandomForest classifier.
    """

    def menu_randomforest(self):
        if self.loaded & self.split:
            auc, pred_proba, mislabeled = randomForest.buildRandomForest(data, self.configFI, self.configFIUse)
            self.output_str("AUC: " + str(auc))
            self.output_str(" " + str(mislabeled))
            self.performance_report("Random Forest", pred_proba, data.y_test)
            self.update_overview(best_score=float(format(auc, '.3f')))

        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    """
    Build an XGBoost classifier.
    """

    def menu_xgboost(self):
        if self.loaded & self.split:
            auc, pred_proba, mislabeled = xgboost_model.buildXGBoost(data, self.configFI, self.configFIUse)
            self.output_str("AUC: " + str(auc))
            self.output_str(" " + str(mislabeled))
            self.performance_report("XGBoost", pred_proba, data.y_test)
            self.update_overview(best_score=float(format(auc, '.3f')))

        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    """
    Internal function to produce the evaluation metrics and format the output.
    Is dependant on the prediction probabilities based on covariates X and the real y-values.
    Should be used with the test-set, and not training-set to get the generalisation error.
    """
    def performance_report(self,name,pred_proba,y):
        from sklearn.metrics import roc_auc_score, roc_curve, log_loss, classification_report, accuracy_score
        self.output_str(str(name) + ":")
        self.output_str(classification_report(y, np.argmax(pred_proba, axis=1), target_names=["0", "1"]))
        self.output_str("AUC      : %.4f" % roc_auc_score(y, pred_proba[:, 1]))
        self.output_str("Accuracy : %.4f" % accuracy_score(y, np.argmax(pred_proba, axis=1)))
        self.output_str("Log Loss : %.4f" % log_loss(y, pred_proba[:, 1]))

    def console(self):
        pass

    """
    output_str() is used for module/model output such as training error etc.
    """
    def output_str(self,output):
        # %Y-%m-%d
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        self.output_log = str(self.output_log) + str(st) + ": " + str(output)
        self.output_log += "\r \n"
        # Logging is done here.
        enable_logging = True
        if enable_logging:
            workfile = str(self.configGeneral['workdir']) + "output.log"
            with open(workfile, 'a') as f:
                to_log_file = str(st) + ": " + str(output) + "\r \n"
                f.write(to_log_file)
                f.close()
        # Need to push latest update to Kivy StringObject output_text.
        if self.log_toggled:
            self.output_text = self.output_log
        else:
            self.output_text = self.output_last

    """

    Adding newline to longer output for use in _output_last().

    """

    def _convert_newline(self,output):
        new_output = ""

        return new_output

    """
    _output_last() is used for more extensive information rather than "label encoding finished successfully" etc.
    """
    def _output_last(self, output):
        # Get the output from a function (usually much information)
        self.output_last = str(output)

        # Need to push latest update to Kivy StringObject output_text.
        if self.log_toggled:
            self.output_text = self.output_log
        else:
            self.output_text = self.output_last
    """
    _output_last_format(self, type, log_proba, score)
    """
    def _output_last_format(self, type,score,log_proba,time,classified):

        output = """
%s
========================
Runtime: X.X
Number of %s
- Log_Proba: %0.2f
                    """ % (type,score, log_proba)
        return output
    """
    feedback() is used for simple feedback to the user such as finished loading data set etc.
    """
    def feedback(self,feedback):
        self.output_console =  "Console: " + feedback

    def Clock_feedback(self, dt, feedback):
        self.output_console = "Console: " + str(feedback)


    """
    Function exp_quick_load() quickly loads the data and splits it into a small training set.
    """

    def exp_quick_load(self):
        self.output_str("Import.")
        global data
        data = DataSet()
        data.dataimport("D:\Dropbox\St Andrews\IT\IS5189 MSc Thesis\\02 Data\InnoCentive_Challenge_9933493_training_data.csv")
        self.loaded = True
        self.output_str("Label Encode.")
        data.labelencode(columns=self.configLE)
        self.output_str("Split (quick = True).")
        data.split(target_column_name=self.configCV['target_value'], test_set_size=self.configCV['test_set_size'],
                   seed=self.configCV['seed'], random_state_is=self.configCV['random_state_is'],quick=True)
        self.update_overview(trainrows=len(data.X_train), testrows=len(data.X_test),
                             ncols=len(data.X_train.columns.values))
        self.output_str("Function 'exp_quick_load()' finished running.")
        data.descstats(self.configLE,write=True,workdir=self.configGeneral['workdir'])



    """
    Experimental Function to test new features.
    """

    def exp_nbn_best(self):
        if self.loaded & self.split:
            result,string = nbn.imp_topten(data,self.configFIUse,self.configFI)
            self._output_last(string)
        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")


    """
    Experimental Function to test new features.
    """

    def exp_(self):
        #"""
        data = DataSet()
        self.quick = DataSet()
        data.dataimport("D:\Dropbox\St Andrews\IT\IS5189 MSc Thesis\\02 Data\InnoCentive_Challenge_9933493_training_data.csv")
        data.labelencode(columns=self.configLE)
        xtest, xtrain, ytest, ytrain = data.split(quick=True)
        self.quick.import_split(xtest, xtrain, ytest, ytrain)
        self.output_str("10 percent of original dataset loaded (into train. Testset is 90 percent).")
        rows_train = len(xtrain)
        self.feedback("Challenge data loaded. self.quick init with " + str(rows_train) + " rows.")
        correlation_list, descstats = self.quick.correlation()
        self._output_last(correlation_list)
        #print(test)
        #a = test.sort_values(by='Correlation', ascending=True).head(20)
        #b = test.sort_values(by='Correlation',ascending=False).head(20)
        #print(a)
        #print(b)
        #print(descstats)
        #self.quick.descstats()
        #"""
        #Clock.schedule_once(lambda dt: self.feedback("this is good"), -1)
        #descstats = data.descstats(self.configLE)
        ############################################################
        # df is short for DataFrame , to make it more readable when manipulating the Pandas DataFrame.
        # Might be easier (and is shorter) to read by developers as an in house var name.
        threshold = 0.7
        df = correlation_list[correlation_list['Correlation'] > threshold]
        df = df.sort_values(by='Correlation',ascending=False)
        column_a_b = df['Var1']
        column_a_b = column_a_b.append(df['Var2'])
        print(df[df['Var1'] == 'C31'])
        print(column_a_b.value_counts())
        #print(df.head(10))
        print(pd.crosstab(df['Var1'], df['Var2']))


    """

    FUTURE EXTENSIONS:

    """
    def menu_future_ext1(self):
        if self.loaded & self.split:
            self.output_str("Future Extension #1")
            est = 1
            depth = 2
            self.output_str("Demonstration 19.8 - Novel Feature Importance based on Random Forest")
            info, total_columns, dummy_y_percent = randomForest.fi_RandomForest_improved2(data,self.configFIUse,self.configFI, estimators=est, maximum_depth=depth)
        else:
            if self.loaded is False:
                self.feedback("Please load a dataset.")
            elif self.split is False:
                self.feedback("Please split the data set into training and validation sets.")

    def menu_future_ext2(self):
        self.output_str("Future Extension #2")
        self.feedback("Future Extension #2")


    """

     TESTING NEW CODE

    """

    def _roc_plot(self,y_proba,y,model='Unknown',title='Receiver Operating Characteristics',x_label="False Positive Rate",y_label='True Positive Rate',auc=None):
        #from sklearn.metrics import mean_squared_error
        #from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_curve,roc_auc_score

        if self.loaded:
            if auc is None:
                auc = roc_auc_score(y, y_proba[:, 1])

            #list_mse_testing.append(mean_squared_error(y,pred_proba[:, 1]))
            #list_acc.append(accuracy_score(data.y_test, np.argmax(pred_proba, axis = 1)))

            import seaborn as sns
            sns.set_style("white")
            plt.figure()
            plt.title(str(title))
            plt.xlabel(str(x_label))
            plt.ylabel(str(y_label))
            #train_scores_mean = np.mean(train_scores, axis=1)
            #train_scores_std = np.std(train_scores, axis=1)
            plt.grid()
            plt.plot(roc_curve(y, y_proba[:, 1])[:2], 'o-', c=sns.color_palette()[0], label=str(model) + " (AUC: %.2f)" % auc)
            plt.plot((0., 1.), (0., 1.), "--k", alpha=.7)
            #plt.plot(list_trees, list_mse_testing, '-', c=sns.color_palette()[1], label="Validation score")
            plt.legend(loc="best")
            # plt.savefig('D:\workdir\plot.png')
            plt.show()


    def menu_plot(self):
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import accuracy_score
        if self.loaded:
            max_trees = range(50,850,100)
            #max_depth = range(3,5,1)
            depth = 3
            list_auc = []
            list_acc = []
            list_trees = []
            list_mse_training = []
            list_mse_testing = []
            scores = []
            list_proba = []

            for trees in max_trees:
                auc, pred_proba, mislabeled,y_t_pred_proba = xgboost_model.buildXGBoost(data, self.configFI, self.configFIUse,max_depth=depth,n_est=trees)
                list_mse_testing.append(mean_squared_error(data.y_test,pred_proba[:, 1]))
                list_mse_training.append(mean_squared_error(data.y_train,y_t_pred_proba[:,1]))
                list_auc.append(auc)
                list_trees.append(trees)
                list_acc.append(accuracy_score(data.y_test, np.argmax(pred_proba, axis = 1)))

            import seaborn as sns
            sns.set_style("white")
            plt.figure()
            plt.title("XGBOOST")
            plt.xlabel("Number of Trees")
            plt.ylabel("Score")
            #train_scores_mean = np.mean(train_scores, axis=1)
            #train_scores_std = np.std(train_scores, axis=1)
            plt.grid()
            plt.plot(list_trees,list_mse_training, 'o-', c=sns.color_palette()[0], label="Training score")
            plt.plot(list_trees, list_mse_testing, '-', c=sns.color_palette()[1], label="Validation score")
            plt.legend(loc="best")
            # plt.savefig('D:\workdir\plot.png')
            plt.show()


    def menu_plot2(self):
        title = "Learning Curves (Naive Bayes)"
        cv = cross_validation.ShuffleSplit(n=data.X_train.shape[0], n_iter=10,test_size=0.2, random_state=0)

        estimator = BernoulliNB()

        #
        ylim = None
        n_jobs = 4
        train_sizes = np.linspace(.1, 1.0, 5)

        ds = data
        X = ds.X_train
        y = ds.y_train
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
        #plt.savefig('D:\workdir\plot.png')
        plt.show()



    """
    END TESTING
    """
    """
    Toggle main output and log
    """

    def menu_toggle_main_output(self):
        if self.log_toggled:
            self.output_label = "Output Log"
            self.output_text = self.output_last
            self.log_toggled = False
            self.feedback("Last output shown.")
        else:
            self.output_label = "Last Output"
            self.output_text = self.output_log
            self.log_toggled = True
            self.feedback("Output log shown.")

    def menu_output_clear(self):
        self.output_str("--- Output Window Cleared ---")
        self.output_log = ""
        self.feedback("Output log and window cleared.")

class PyPredictor(App):
    pass

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)
Factory.register('Settings', cls=Settings)

if __name__ == '__main__':
    PyPredictor().run()