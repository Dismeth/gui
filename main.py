from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
import dataset


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)
    last_path = StringProperty(None)


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    output_console = StringProperty(None)
    output_text = StringProperty(None)

    def __init__(self,**kwargs):
        super(Root, self).__init__(**kwargs)
        version = 0.1
        welcome = "Welcome user. This is version " + str(version)
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

    def show_split(self):
        if "data" not in globals():
            self.feedback("Please load a dataset.")
        else:
            global data
            string_columns = ['C2', 'C4', 'C5', 'C6', 'C9', 'C11', 'C13', 'C15', 'C16', 'C17', 'C19', 'C20', 'C21', 'C22', 'C28', 'C30', 'C53', 'C60']
            self.feedback("Start label encoded the data.")
            data = dataset.labelencode(dataset=data, columns=string_columns)
            self.feedback("Succesfully label encoded the data.")
            self.feedback("Start splitting the data.")
            data = dataset.split(data)
            self.feedback("Succesfully split the data.")

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
            data = dataset.dataimport(filename[0])
        except (RuntimeError, TypeError, NameError):
            dataset.dprint("Error: most likely not a csv file.")
        self.feedback("Fileimport completed")
        self.dismiss_popup()

    """
    save() need to fix this one. Ultimately dataset.save_file(dataset,path)
    """
    def save(self, path, filename):
        dataset.dprint(str(path))
        dataset.dprint(str(filename[0]))
        self.dismiss_popup()

class Editor(App):
    pass

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == '__main__':
    Editor().run()