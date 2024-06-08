from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import *
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.properties import NumericProperty, StringProperty, ObjectProperty
from kivy.lang import Builder
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout

from kivy.config import Config
from kivy.core.window import Window

import PIL.Image as Pil
import numpy as np
import os.path as p
import csv
import os
import shutil

Builder.load_string("""
<LabelDialog>:
    title: 'InputDialog'
    size_hint: None, None
    size: 400, 120
    auto_dismiss: False
    text: input.text
    lb_error: er

    BoxLayout:
        orientation: 'vertical'
        pos: self.pos
        size: root.size

        BoxLayout:
            orientation: 'horizontal'
            Label:
                text: 'Enter labeled'

            TextInput:
                id: input
                multiline: False
                hint_text:'Label'
                input_filter: 'int'
                on_text: root.error = ''
                focus: True

        BoxLayout:
            orientation: 'horizontal'
            Button:
                text: 'Enter'
                background_color: 255,0,0,0.9
                on_release: root._enter()

        Label:
            id: er
            foreground_color: 1, 250, 100, 1
            color: 1, 0.67, 0, 1
            size_hint_y: None
            height: 0
            text: root.error
            
<LabelSquare>:
    
""")


class LabelSquare(Widget):
    """
    Square Widget.
    """
    def update_canvas(self, pos, size, c1=.5, c2=.5, c3=.5):
        self.canvas.clear()
        with self.canvas:
            Color(c1, c2, c3, .4)
            Rectangle(pos=pos, size=size)


class LabelDialog(Popup):
    """
    Label asker dialog
    """
    _label = NumericProperty()
    error = StringProperty()

    def __init__(self, parent, **kwargs):
        """
        :param parent: parent labeler
        :param kwargs: arguments of the popup
        """
        super(LabelDialog, self).__init__(**kwargs)
        # self.parent = parent
        self.par = parent
        self._label = -1
        self.bind(_label=parent.setter('labeled'))
        txt = self.ids.input
        txt.focus = True

    def _enter(self):
        """
        Function called when the enter button is clicked.
        """
        if self.text:
            self._label = int(self.text)
            self._label = -1
            self.ids.input.text = ""
            self.par.is_labeling = False
            self.dismiss()

    
class Labeler(FloatLayout):
    """
    Labeler layout.
    """
    label = NumericProperty()
    
    def __init__(self, unlabeled_dir, labeled_dir, path_to_csv, is_unique):
        """
        :param unlabeled_dir: directory of unlabeled images
        :param labeled_dir: directory of labeled images
        :param path_to_csv: path to csv
        :param is_unique: whether the amount of possible labels is 1 or more
        """
        super(Labeler, self).__init__(size_hint=(1, 1),
                                      pos_hint={'center_x': .5, 'center_y': .5})
        self.x0, self.x1, self.y0, self.y1 = 0, 0, 0, 0
        self.is_unique = is_unique
        self.label = -1
        self.last = None
        self.idx = 0
        self.labels = []
        self.labels.clear()

        self.unlabeled_dir, self.path_to_csv = unlabeled_dir, path_to_csv
        self.labeled_dir = labeled_dir
        self.images = [self.unlabeled_dir + file for file in os.listdir(self.unlabeled_dir)]

        x, y = Pil.open(self.images[0]).size
        Window.size = (720, y/x*720)

        self.img = Image(source=self.images[0],
                         size_hint=(1, 1),
                         pos_hint={'x': 0, 'y': 0},
                         keep_ratio=True,
                         allow_stretch=True)

        self.clear_btn = Button(text='Clear',
                                size_hint=(.1, .05),
                                pos_hint={'x': .9, 'y': .95})
        self.clear_btn.bind(on_release=self.clear_canvas)

        self.end_btn = Button(text='End',
                              size_hint=(.1, .05),
                              pos_hint={'x': .0, 'y': .95})
        self.end_btn.bind(on_release=self.add_to_dataset)

        self.square = LabelSquare()
        # noinspection PyTypeChecker
        self.dialog = LabelDialog(self,
                                  on_open=self.open_dialog,
                                  on_dismiss=self.dismiss_dialog)
        self.add_widget(self.img)
        self.add_widget(self.clear_btn)
        self.add_widget(self.end_btn)
        self.add_widget(self.square)
        self.dialog_is_open = False

    def dismiss_dialog(self, *args):
        """
        Sets the dialog as closed.
        :param args: ignored
        """
        del args
        self.dialog_is_open = False

    def open_dialog(self, *args):
        """
        Sets the dialog as opened.
        :param args: ignored
        """
        del args
        self.dialog_is_open = True

    def get_correct_measurements(self):
        """
        Gets the correct smallest coordinate point.
        :return: Top left coordinate.
        """
        if self.y0 > self.y1:
            if self.x0 > self.x1:
                return self.x1, self.y1
            else:
                return self.x0, self.y1
        else:
            if self.x0 > self.x1:
                return self.x1, self.y0
            else:
                return self.x0, self.y0

    def on_touch_move(self, touch):
        """
        Function called when mouse moves.
        :param touch: position
        """
        super().on_touch_move(touch)
        if not self.dialog_is_open:
            self.x1, self.y1 = touch.pos
            self.square.update_canvas(self.get_correct_measurements(),
                                      (np.abs(self.x1-self.x0), np.abs(self.y1-self.y0)))

    def on_touch_down(self, touch):
        """
        Function called when mouse clicks.
        :param touch: position
        """
        super().on_touch_down(touch)
        if not self.dialog_is_open:
            self.x0, self.y0 = touch.pos
            self.square.pos = touch.pos

    def on_touch_up(self, touch):
        """
        Function called when mouse un-clicks.
        :param touch: position
        """
        super().on_touch_down(touch)
        if not self.dialog_is_open:
            self.x1, self.y1 = touch.pos
            dist = np.sqrt((self.x1 - self.x0)**2 + (self.x1 - self.x0)**2)
            if dist > 200:
                if not self.is_unique:
                    self.dialog.open()
                else:
                    self.on_label()
                self.last = LabelSquare()

                seed = self.label if not self.is_unique else 2
                np.random.seed(seed)
                color = np.random.uniform(0, 1, 3)
                self.last.update_canvas(self.get_correct_measurements(),
                                        (np.abs(self.x1-self.x0), np.abs(self.y1-self.y0)),
                                        color[0], color[1], color[2])
                self.add_widget(self.last)
            self.square.update_canvas((0, 0), (0, 0))

    def on_label(self, *args):
        """
        Function called when the label changes
        :param args: ignored
        """
        del args
        if self.label >= 0 or self.is_unique and self.x0 != self.x1:
            self.labels.append({'labeled': self.label, 'x0': self.x0, 'x1': self.x1, 'y0': self.y0, 'y1': self.y1})
            label = self.labels[-1]['labeled']
            seed = label if not self.is_unique else 1
            np.random.seed(seed)
            color = np.random.uniform(0, 1, 3)
            self.square.update_canvas(self.get_correct_measurements(),
                                    (np.abs(self.x1-self.x0), np.abs(self.y1-self.y0)),
                                    color[0], color[1], color[2])
            self.label = -1

    def clear_canvas(self, obj):
        """
        Clears the canvas
        :param obj: ignored
        """
        del obj
        self.labels.clear()
        self.clear_widgets()
        self.add_widget(self.img)
        self.add_widget(self.clear_btn)
        self.add_widget(self.end_btn)
        self.add_widget(self.square)

    def add_to_dataset(self, *args):
        """
        Adds the labeled image to the dataset
        :param args: ignore.
        """
        del args
        labels_copy = self.labels.copy()

        x, y = Pil.open(self.images[self.idx]).size

        for elem in labels_copy:
            # (left, upper, right, lower)
            x0 = elem['x0']
            x1 = elem['x1']
            y0 = y/x*720-elem['y0']
            y1 = y/x*720-elem['y1']

            elem['x0'] = int(min(x0, x1)*x/720)
            elem['x1'] = int(max(x0, x1)*x/720)
            elem['y0'] = int(min(y0, y1)*x/720)
            elem['y1'] = int(max(y0, y1)*x/720)

        tags = [[obj['labeled'], obj['x0'], obj['y0'], obj['x1'], obj['y1']] for obj in labels_copy]

        _, file = p.split(p.abspath(self.images[self.idx]))

        shutil.move(self.images[self.idx], self.labeled_dir + file)

        row = [{'image': file, 'tags': tags}]

        with open(self.path_to_csv, 'a', newline='') as outfile:
            fieldnames = ['image', 'tags']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
            writer.writerows(row)
        self.idx += 1
        if self.idx >= len(self.images):
            Window.close()
        else:
            x, y = Pil.open(self.images[self.idx]).size
            Window.size = 720, y/x*720
            self.clear_widgets()
            self.img = Image(source=self.images[self.idx],
                             size_hint=(1, 1),
                             pos_hint={'x': 0, 'y': 0},
                             keep_ratio=True,
                             allow_stretch=True)
            self.add_widget(self.img)

            self.add_widget(self.clear_btn)
            self.add_widget(self.end_btn)

            self.add_widget(self.square)

        self.labels.clear()


class LabelerApp(App):
    """
    Labeler application.
    """
    def __init__(self, unlabeled_dir, labeled_dir, path_to_csv, is_unique):
        """
        :param unlabeled_dir: directory of unlabeled images
        :param labeled_dir: directory of labeled directory
        :param path_to_csv: path to csv
        :param is_unique: whether the number of classes is unique
        """
        super(LabelerApp, self).__init__()
        self.unlabeled_dir = unlabeled_dir
        self.labeled_dir = labeled_dir
        self.path_to_csv = path_to_csv
        self.is_unique = is_unique

    def build(self):
        """
        Builds the labeler
        :return: labeler
        """
        Config.set('graphics', 'resizable', False)
        parent = Labeler(self.unlabeled_dir, self.labeled_dir, self.path_to_csv, self.is_unique)
        return parent


def label(unlabeled_dir, labeled_dir, csv, n_classes=2):
    """
    Attemps to label all images in the given directory
    :param unlabeled_dir: directory of unlabeled images
    :param labeled_dir: directory of labeled images
    :param csv: path to csv file
    :param n_classes: number of classes (not enforced)
    """
    if len(os.listdir(unlabeled_dir)) == 0:
        print("Nothing to label")
    else:
        LabelerApp(unlabeled_dir, labeled_dir, csv, n_classes == 1).run()
