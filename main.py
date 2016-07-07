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
from kivy.uix.switch import Switch
from dataset import DataSet
import nbn
import datetime
import time
import timeit


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
    newconfigCV = DictProperty({})
    newconfigLE = StringProperty(None)
    newconfigFI = StringProperty(None)
    newconfigFIUse = BooleanProperty(None)


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
    output_console = StringProperty(None)
    output_text = StringProperty(None)

    def initialiseSettings(self):
        self.configCV = {'target_value': 'target_purchase', 'test_set_size': 0.4, 'seed': 16, 'random_state_is': True}
        self.configLE = ['C2', 'C4', 'C5', 'C6', 'C9', 'C11', 'C13', 'C15', 'C16', 'C17', 'C19', 'C20', 'C21', 'C22', 'C28', 'C30', 'C53', 'C60']
        self.configFI = ['record_ID', 'target_purchase'] #example columns to exclude
        self.configFIUse = True # because record_ID and target_purchase is included.

    def __init__(self,**kwargs):
        super(Root, self).__init__(**kwargs)
        version = 0.2
        welcome = "Welcome user. This is version " + str(version) + ". Click on Load to start, or Help for more information."
        self.loaded = False
        self.feedback(welcome)
        self.output_text = ""
        """ Global variables to be edited in the settings """
        self.initialiseSettings()


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
                    newconfig += conf
                    firstline = False
                else:
                    newconfig += ";" + conf # Semicolon used as separator. Can be changed.
        return newconfig

    """ Not obsolete, all settings are made in settings-popup. """
    def show_settings(self):
        content = Settings(settings_save=self.settings_save, cancel=self.dismiss_popup,
                           newconfigCV=self.configCV, newconfigLE=self.transform_config(self.configLE),
                           newconfigFI=self.transform_config(self.configFI), newconfigFIUse=self.configFIUse)
        self._popup = Popup(title="Data Analysis Settings", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    """ Save Settings """
    def settings_save(self, updated_configle, updated_configcv, updated_configfi, updated_use_fi):
        self.feedback("Settings Saved")
        self.output_str(updated_configcv)
        self.output_str(updated_configle)
        self.output_str(updated_configfi)
        self.output_str(updated_use_fi)
        self.configCV = updated_configcv
        self.configLE = self.transform_config(updated_configle,back=True)
        self.configFI = self.transform_config(updated_configfi,back=True)
        self.configFIUse = updated_use_fi
        self.output_str(updated_configcv)
        self.output_str(self.configLE)
        self.output_str(self.configFI)
        self.output_str(updated_use_fi)
        self.dismiss_popup()

    def testfunction(self):
        data = DataSet()
        data.dataimport("D:\Dropbox\St Andrews\IT\IS5189 MSc Thesis\\02 Data\InnoCentive_Challenge_9933493_training_data.csv")
        data.labelencode(columns=self.configLE)
        data.split()
        best = nbn.imp_topten(data, self.configFI, self.configFIUse, target_column_name='target_purchase')
        print(best)
        #best = best.sort_values(by='Mislabeled',ascending=True)
        #print(best.head(n=10))



    def split_data(self):
        if self.loaded:
            self.output_str("Start splitting the data.")
            data.split(target_column_name = self.configCV['target_value'], test_set_size = self.configCV['test_set_size'], seed = self.configCV['seed'], random_state_is = self.configCV['random_state_is'])
            self.output_str("Finished splitting the data in UNKNOWN seconds")
            self.feedback("Succesfully split the data.")
        else:
            self.feedback("Please load a dataset.")

    def show_nbn(self):
        if self.loaded:
            ypred,output,log_proba = nbn.naivebayesian(data)
            self.output_str(ypred)
            self.output_str(output)
            self.output_str(log_proba)
            self.feedback("Succesfully made a lol the data.")
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

    """
    feedback() is used for simple feedback to the user such as finished loading data set etc.
    """
    def feedback(self,feedback):
        self.output_console =  "Console: " + feedback

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