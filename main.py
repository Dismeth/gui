from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
import dataset

import os


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveDialog(FloatLayout):
    pass


class Root(FloatLayout):
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    output_console = StringProperty(None)

    def __init__(self,**kwargs):
        super(Root, self).__init__(**kwargs)
        self.feedback("Welcome")

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        pass
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

    def feedback(self,output):
        self.output_console =  output

    def progress_bar(self):
        pass

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


    def save(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text_input.text)

        self.dismiss_popup()


class Editor(App):
    pass

Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ == '__main__':
    Editor().run()