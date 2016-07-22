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
import randomForest
import nbn
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
    newconfigFI = StringProperty(None)
    newconfigFIUse = BooleanProperty(None)

    """
    Functions to save all the settings from the settings window.
    """
    def saveGeneral(self, desc_stats_on_load, workdir):
        self.newconfigGeneral['desc_stats_on_load'] = bool(desc_stats_on_load)
        self.newconfigGeneral['workdir'] = str(workdir)
        return self.newconfigGeneral

    def saveLE(self,le_columns):
        self.newconfigLE = le_columns
        return self.newconfigLE

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
    splitfile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    # Define variables to be updated
    output_console = StringProperty(None)
    output_text = StringProperty(None)
    # Overview variables that need to be updated:
    data_loaded = StringProperty(None)
    fname = StringProperty(None)
    trainrows = StringProperty(None)
    testrows = StringProperty(None)
    ncols = StringProperty(None)
    best_score = StringProperty(None)

    def initialiseSettings(self):
        self.configGeneral = {'desc_stats_on_load': True,'workdir': 'D:\workdir\\'}
        self.configCV = {'target_value': 'target_purchase', 'test_set_size': 0.4, 'seed': 16, 'random_state_is': True}
        self.configLE = ['C2', 'C4', 'C5', 'C6', 'C9', 'C11', 'C13', 'C15', 'C16', 'C17', 'C19', 'C20', 'C21', 'C22', 'C28', 'C30', 'C53', 'C60']
        self.configFI = ['C2'] #example columns to exclude
        self.configFIUse = False # This has been updated.. Need to figure out how to control this one. (because record_ID and target_purchase is included.)
        self.dataOverview = {'data_loaded': False, 'fname': 'N/A', 'trainrows': 0, 'testrows': 0,'ncols': 0, 'best_score': 0}
        self.feedback("Settings has been reset.")

    def __init__(self,**kwargs):
        super(Root, self).__init__(**kwargs)
        """ Global variables to be edited in the settings """
        self.loaded = False
        self.initialiseSettings()
        self._update_overviewGUI()
        """ Welcome message etc """
        version = 0.3
        welcome = "Welcome user. This is version " + str(version) + ". Click on Load to start, or Help for more information."
        self.feedback(welcome)
        """ Set up logging """
        self.output_text = ""
        """
        Open Settings first time, needed a delay in order to wait for everything to be properly set.
        """
        Clock.schedule_once(self.show_startup_settings, 1)


    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def label_encode(self):
        if self.loaded:
            #Clock.schedule_once(lambda dt: self.feedback("Start label encoded the data."), -1)
            self.output_str("Start label encoded the data.")
            data.labelencode(columns=self.configLE)
            self.output_str("End label encoded the data.")
            self.feedback("Succesfully label encoded the data.")
        else:
            self.feedback("Please load a dataset.")

    """ Obsolete, all settings are made in settings-popup. """

    def show_split_data(self):
        content = SplitData(split=self.split_data, cancel=self.dismiss_popup)
        self._popup = Popup(title="Split data", content=content, size_hint=(0.9,0.9))
        self._popup.open()

    """

    Updates the variables on the top.

    """

    def update_overview(self, data_loaded = 0, fname = 0, trainrows = 0, testrows=0, ncols = 0, best_score = False):
        if not data_loaded == 0:
            self.dataOverview['data_loaded'] = data_loaded
        if not fname == 0:
            self.dataOverview['fname'] = fname
        if not trainrows == 0:
            self.dataOverview['trainrows'] = trainrows
        if not testrows == 0:
            self.dataOverview['testrows'] = testrows
        if not ncols == 0:
            self.dataOverview['ncols'] = ncols
        if not best_score == False:
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
                           newconfigCV=self.configCV, newconfigLE=self.transform_config(self.configLE),
                           newconfigFI=self.transform_config(self.configFI), newconfigFIUse=self.configFIUse)
        self._popup = Popup(title="Data Analysis Settings", content=content, size_hint=(0.9, 0.9))
        self._popup.open()


    """
    Save Settings. Called when closing the settings_popup.
    """
    def settings_save(self, updated_configGeneral, updated_configle, updated_configcv, updated_configfi, updated_use_fi):
        # Updates the config
        self.configGeneral = updated_configGeneral
        self.configCV = updated_configcv
        self.configLE = self.transform_config(updated_configle,back=True)
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
    Experimental Function to test new features.
    """

    def exp_quick_load(self):
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
    Build a RandomForest classifier and return feature importances.
    """
    def show_feature_importance(self):
        if self.loaded:
            randomForest.feature_importance_RandomForest(data, self.configFI, self.configFIUse)
        elif self.quick.exists():
            randomForest.feature_importance_RandomForest(self.quick, self.configFI, self.configFIUse)
        else:
            self.feedback("Please load a dataset.")

    """
    Build a RandomForest classifier.
    """

    def show_randomforest(self):
        if self.loaded:
            scores, total_points, mislabeled = randomForest.buildRandomForest(data, self.configFI, self.configFIUse)
            loaded = True
        elif self.quick.exists():
            scores, total_points, mislabeled = randomForest.buildRandomForest(self.quick, self.configFI, self.configFIUse)
            loaded = True
        else:
            self.feedback("Please load a dataset.")
        if loaded:
            self.output_str("Mean Score: " + str(scores))
            self.output_str("Total points: " + str(total_points))
            self.output_str(" " + str(mislabeled))


    """
    function split_data(): splits the data into a training set and validation set with user's options.
    """
    def split_data(self):
        if self.loaded:
            self.output_str("Start splitting the data.")
            data.split(target_column_name = self.configCV['target_value'], test_set_size = self.configCV['test_set_size'], seed = self.configCV['seed'], random_state_is = self.configCV['random_state_is'])
            self.output_str("Finished splitting the data.")
            self.feedback("Succesfully split the data.")
            self.update_overview(trainrows=len(data.X_train),testrows=len(data.X_test))
        else:
            self.feedback("Please load a dataset.")

    """

    Create a Naive Bayesian Network Model

    """

    def show_nbn(self):
        if self.loaded:
            output,log_proba = nbn.naivebayesian(data,self.configFIUse,self.configFI)
            #self.output_str(ypred)
            self.output_str(output)
            self.output_str(log_proba)
            self.feedback("Succesfully made a NBN.")
        else:
            self.feedback("Please load a dataset.")



    def console(self):
        pass

    """
    output_str() is used for module/model output such as training error etc.

    """
    def output_str(self,output):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        self.output_text = str(self.output_text) + str(st) + ": " + str(output)
        self.output_text += "\r \n"

        # Logging is done here.
        enable_logging = True
        if enable_logging:
            workfile = str(self.configGeneral['workdir']) + "output.log"
            with open(workfile, 'a') as f:
                to_log_file = str(st) + ": " + str(output) + "\r \n"
                f.write(to_log_file)
                f.close()

    """
    feedback() is used for simple feedback to the user such as finished loading data set etc.
    """
    def feedback(self,feedback):
        self.output_console =  "Console: " + feedback

    def Clock_feedback(self, dt, feedback):
        self.output_console = "Console: " + str(feedback)

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
        self.update_overview(data_loaded=self.loaded,fname=fname,ncols=ncols)
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

class Editor(App):
    pass

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)
Factory.register('Settings', cls=Settings)

if __name__ == '__main__':
    Editor().run()