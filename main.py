from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.properties import BoundedNumericProperty
from kivy.uix.popup import Popup
from dataset import DataSet


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
    #test_set_size = BoundedNumericProperty(0.4, min=0.0, max=1.0)
    #k_fold = BoundedNumericProperty(5, min=1, max=20)

class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    splitfile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    output_console = StringProperty(None)
    output_text = StringProperty(None)

    def __init__(self,**kwargs):
        super(Root, self).__init__(**kwargs)
        version = 0.1
        welcome = "Welcome user. This is version " + str(version)
        self.loaded = False
        self.feedback(welcome)

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
            string_columns = ['C2', 'C4', 'C5', 'C6', 'C9', 'C11', 'C13', 'C15', 'C16', 'C17', 'C19', 'C20', 'C21', 'C22', 'C28', 'C30', 'C53', 'C60']
            self.feedback("Start label encoded the data.")
            data.labelencode(columns=string_columns)
            self.feedback("Succesfully label encoded the data.")
        else:
            self.feedback("Please load a dataset.")

    def show_split_data(self):
        content = SplitData(split=self.split_data, cancel=self.dismiss_popup)
        self._popup = Popup(title="Split data", content=content, size_hint=(0.9,0.9))
        self._popup.open()

    def show_settings(self):
        content = Settings(settings_save=self.settings_save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Data Analysis Settings", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def settings_save(self):
        self.feedback("Settings Saved")
        #self.dismiss_popup()

    def split_data(self):
        if self.loaded:
            self.feedback("Start splitting the data.")
            data.split()
            self.feedback("Succesfully split the data.")
        else:
            self.feedback("Please load a dataset.")

    def show_nbn(self):
        self.feedback("To be implemented.")

    def console(self):
        pass

    """
    output_str() is used for module/model output such as training error etc.
    """
    def output_str(self,output):
        self.output_text = output

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

if __name__ == '__main__':
    Editor().run()