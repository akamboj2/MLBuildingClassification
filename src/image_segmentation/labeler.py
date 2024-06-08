from kivy.uix.button import Button
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.properties import NumericProperty, StringProperty
from kivy.lang import Builder
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config
import numpy as np
import os.path as p
import csv
import os
import shutil
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image as kiImage
from PIL import Image, ImageDraw
from io import BytesIO
import re
from kivy.core.window import Window
import ast


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
            
            
<BrushDialog>:
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
                text: 'Enter brush size'

            TextInput:
                id: input
                multiline: False
                hint_text:'ex: 10'
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


class LabelDialog(Popup):
    """
    Labeler dialog.
    Asks for a label.
    """
    """
    Current label
    """
    _label = NumericProperty()
    """
    Message
    """
    error = StringProperty()

    def __init__(self, parent, **kwargs):
        """
        Constructor

        :param parent: dialog parent labeler
        :param kwargs: kwargs of the popup
        """
        super(LabelDialog, self).__init__(**kwargs)
        self.par = parent
        self._label = 0
        self.bind(_label=parent.setter('label'))
        txt = self.ids.input
        txt.focus = True

    def _enter(self):
        """
        Function called when enter is pressed.
        """
        if self.text:
            self._label = int(self.text)
            self.ids.input.text = ""
            self.par.is_labeling = False
            self.dismiss()


class BrushDialog(Popup):
    """
    Brush stroke dialog.
    Asks for a brush stroke size.
    """
    """
    Current label
    """
    _brush = NumericProperty()
    """
    Message
    """
    error = StringProperty()

    def __init__(self, parent, **kwargs):
        """
        Constructor

        :param parent: dialog parent labeler
        :param kwargs: kwargs of the popup
        """
        super(BrushDialog, self).__init__(**kwargs)
        self.par = parent
        self._brush = 10
        self.bind(_brush=parent.setter('brush'))
        txt = self.ids.input
        txt.focus = True

    def _enter(self):
        """
        Function called when enter is pressed.
        """
        if self.text:
            self._brush = int(self.text)
            self.ids.input.text = ""
            self.par.is_labeling = False
            self.dismiss()


class Labeler(FloatLayout):
    """
    Labeler layout.
    """

    label = NumericProperty()
    brush = NumericProperty()
    """
    label
    """

    def __init__(self, unlabeled_dir, labeled_dir, path_to_csv, n_classes):
        """
        Constructor

        :param unlabeled_dir: directory of the unlabeled images (end with /)
        :param labeled_dir: directory of the labeled images (end with /)
        :param path_to_csv: path to csv file
        :param n_classes: number of classes (valid labels are not enforced)
        """
        super(Labeler, self).__init__(size_hint=(1, 1),
                                      pos_hint={'center_x': .5, 'center_y': .5})
        self.x, self.y = 0, 0
        self.classes = n_classes
        self.label = 0
        self.brush=10
        self.idx = 0
        self.picked_color = 0, 0, 0
        self.is_painting = False
        self.unlabeled_dir, self.path_to_csv = unlabeled_dir, path_to_csv
        self.labeled_dir = labeled_dir
        self.images = [self.unlabeled_dir + file for file in os.listdir(self.unlabeled_dir)]
        self.click_pos = 0, 0
        self.brush_selector = BrushDialog(self)
        img = Image.open(self.images[self.idx])
        resizedImage = img.resize(Window.size)
        rgb_im = resizedImage.convert('RGB')
        self.images[self.idx] = p.splitext(self.images[self.idx])[0] + '.jpg'
        rgb_im.save(self.images[self.idx])
        img.close()

        self.img = kiImage(source=self.images[0],
                           size_hint=(1, 1),
                           pos_hint={'x': 0, 'y': 0},
                           keep_ratio=True,
                           allow_stretch=True)

        np.random.seed(0)
        color = np.random.uniform(1, 255, 3)
        self.picked_color = int(color[0]), int(color[1]), int(color[2])
        self.canvas_img = Image.new('RGB', Window.size, color=self.picked_color)
        self.paint = kiImage(size_hint=(1, 1),
                             pos_hint={'x': 0, 'y': 0})
        self.paint.opacity = .5

        data = BytesIO()
        self.canvas_img.save(data, format='png')
        data.seek(0)  # yes you actually need this
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.paint.texture = im.texture

        # noinspection PyTypeChecker
        self.dialog = LabelDialog(self)

        self.clear_btn = Button(text='Clear',
                                size_hint=(.1, .05),
                                pos_hint={'x': .9, 'y': .95})
        self.clear_btn.bind(on_release=self.clear_canvas)

        self.color_btn = Button(text='Color',
                                size_hint=(.1, .05),
                                pos_hint={'x': .9, 'y': .9})
        self.color_btn.bind(on_press=lambda x: self.dialog.open())

        self.brush_btn = Button(text='Brush',
                                size_hint=(.1, .05),
                                pos_hint={'x': .9, 'y': .85})
        self.brush_btn.bind(on_press=lambda x: self.brush_selector.open())

        self.discard_btn = Button(text='Discard',
                                  size_hint=(.1, .05),
                                  pos_hint={'x': .0, 'y': .9})
        self.discard_btn.bind(on_press=self.discard)

        self.end_btn = Button(text='End',
                              size_hint=(.1, .05),
                              pos_hint={'x': .0, 'y': .95})
        self.end_btn.bind(on_release=self.add_to_dataset)

        self.add_widget(self.img)

        self.add_widget(self.paint)
        self.add_widget(self.clear_btn)
        self.add_widget(self.color_btn)
        self.add_widget(self.end_btn)
        self.add_widget(self.discard_btn)
        self.add_widget(self.brush_btn)

        self.dialog_is_open = False

    def discard(self, *args):
        """
        Discards the image.

        :param args: arguments
        """
        del args

        os.remove(self.images[self.idx])
        self.idx += 1

        if self.idx >= len(self.images):
            Window.close()
        else:
            img = Image.open(self.images[self.idx])
            resizedImage = img.resize(Window.size)
            rgb_im = resizedImage.convert('RGB')
            self.images[self.idx] = p.splitext(self.images[self.idx])[0] + '.jpg'
            rgb_im.save(self.images[self.idx])
            img.close()

            self.clear_widgets()

            self.img = kiImage(source=self.images[self.idx],
                               size_hint=(1, 1),
                               pos_hint={'x': 0, 'y': 0},
                               keep_ratio=True,
                               allow_stretch=True)
            self.clear_canvas(0)

    def paint_line(self, pos):
        """
        Paints a line from latest position known to the new position.

        :param pos: new position
        """
        draw = ImageDraw.Draw(self.canvas_img)
        steps = 20
        size = self.brush
        y1 = self.height / self.width * 720 - self.click_pos[1]
        y0 = (self.height / self.width * 720 - pos[1])
        x0 = pos[0]
        x1 = self.click_pos[0]
        for theta in range(0, steps, 1):
            pos = theta/steps*x0 + (1 - theta/steps)*x1, \
                  theta/steps*y0 + (1 - theta/steps)*y1

            draw.ellipse((pos[0]-size, pos[1]-size, pos[0]+size, pos[1]+size),
                         fill=self.picked_color)
        del draw

    def dismiss_dialog(self, *args):
        """
        Sets the dialog as closed.

        :param args: ignored
        """
        del args
        self.dialog_is_open = False

    def open_dialog(self, *args):
        """
        Sets the dialog as open.

        :param args: ignored
        """
        del args
        self.dialog_is_open = True

    def on_touch_move(self, touch):
        """
        Function called when window is clicked.
        Paints the line.

        :param touch: mouse position
        """
        if not self.dialog_is_open and self.is_painting:
            self.paint_line(touch.pos)
            self.click_pos = touch.pos

            data = BytesIO()
            self.canvas_img.save(data, format='png')
            data.seek(0)  # yes you actually need this
            im = CoreImage(BytesIO(data.read()), ext='png')
            self.paint.texture = im.texture

    def on_touch_down(self, touch):
        """
        Sets the state as painting.

        :param touch: mouse position
        """
        self.click_pos = touch.pos
        super().on_touch_down(touch)
        if not self.dialog_is_open:
            self.is_painting = True

    def on_touch_up(self, touch):
        """
        Sets the state as not painting.

        :param touch: mouse position
        """
        if not self.dialog_is_open:
            self.is_painting = False
        super().on_touch_down(touch)

    def on_label(self, *args):
        """
        Function called when a different label is chosen.
        :param args: param ignored
        """
        del args
        if self.label >= 0:
            np.random.seed(self.label)
            color = np.random.uniform(1, 255, 3)
            self.picked_color = int(color[0]), int(color[1]), int(color[2])

    def clear_canvas(self, obj):
        """
        Clears the canvas.

        :param obj: ignored
        """
        del obj
        self.clear_widgets()

        np.random.seed(0)
        color = np.random.uniform(1, 255, 3)
        self.picked_color = int(color[0]), int(color[1]), int(color[2])
        self.canvas_img = Image.new('RGB', Window.size, color=self.picked_color)
        self.paint = kiImage(size_hint=(1, 1),
                             pos_hint={'x': 0, 'y': 0})
        self.paint.opacity = .5

        data = BytesIO()
        self.canvas_img.save(data, format='png')
        data.seek(0)  # yes you actually need this
        im = CoreImage(BytesIO(data.read()), ext='png')
        self.paint.texture = im.texture

        self.add_widget(self.img)

        self.add_widget(self.paint)
        self.add_widget(self.clear_btn)
        self.add_widget(self.color_btn)
        self.add_widget(self.end_btn)
        self.add_widget(self.brush_btn)
        self.add_widget(self.discard_btn)

    def add_to_dataset(self, *args):
        """
        Adds the new sample to the dataset.

        :param args: ignored
        """
        del args

        output = np.zeros((self.height, self.width), dtype=int)
        labeled  = np.array(self.canvas_img)
        for cls in range(self.classes):
            np.random.seed(cls)
            color = np.random.uniform(1, 255, 3)
            first = labeled[:, :, 0] == int(color[0])
            second = labeled[:, :, 1] == int(color[1])
            third = labeled[:, :, 2] == int(color[2])
            output += cls*np.logical_and(first, np.logical_and(second, third))

        image_name = os.path.split(self.images[self.idx])[1]
        _, file = p.split(p.abspath(self.images[self.idx]))

        image_without_extension = os.path.splitext(image_name)[0]
        labeled_path = self.labeled_dir + image_without_extension + '.npy'

        np.save(file=labeled_path, arr=output)

        shutil.move(self.images[self.idx], self.labeled_dir + file)

        row = {'image': p.splitext(p.split(self.images[self.idx])[1])[0]}

        with open(self.path_to_csv, 'a', newline='') as outfile:
            fieldnames = ['image']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
            writer.writerow(row)

        self.idx += 1
        if self.idx >= len(self.images):
            Window.close()
        else:
            img = Image.open(self.images[self.idx])
            resizedImage = img.resize(Window.size)
            rgb_im = resizedImage.convert('RGB')
            self.images[self.idx] = p.splitext(self.images[self.idx])[0] + '.jpg'
            rgb_im.save(self.images[self.idx])
            img.close()

            self.clear_widgets()

            self.img = kiImage(source=self.images[self.idx],
                               size_hint=(1, 1),
                               pos_hint={'x': 0, 'y': 0},
                               keep_ratio=True,
                               allow_stretch=True)
            self.clear_canvas(0)


class LabelerApp(App):
    """
    Labeler Application
    """
    def __init__(self, path_to_labeled, path_to_unlabeled, path_to_csv, n_classes):
        """
        Constructor

        :param path_to_labeled: path to labeled images
        :param path_to_unlabeled: path to unlabeled images
        :param path_to_csv: path to csv file
        :param n_classes: number of classes
        """
        super(LabelerApp, self).__init__()
        self.path_to_labeled = path_to_labeled
        self.path_to_unlabeled = path_to_unlabeled
        self.path_to_csv = path_to_csv
        self.n_classes = n_classes

    def build(self):
        """
        Builds the application.
        Returns the built app.
        """
        Config.set('graphics', 'resizable', False)
        Window.size = (720, 720)
        parent = Labeler(self.path_to_labeled, self.path_to_unlabeled, self.path_to_csv, self.n_classes)
        return parent


def label(unlabeled_dir, labeled_dir, csv, n_classes=3):
    """
    Attempts to label all the images in the unlabeled directory and moves them to the labeled directory.

    :param unlabeled_dir: unlabeled image directory
    :param labeled_dir: kabeled image directory
    :param csv: csv path
    :param n_classes: number of classes
    """
    LabelerApp(unlabeled_dir, labeled_dir, csv, n_classes=n_classes).run()


def crop(img, box):
    """
    Returns the cropped image.

    :param img: image
    :param box: box
    """
    img = img.convert("RGB")
    return img.crop(tuple(box))


def crop_from_labeled_objects(image_dir, csv_file, output_dir, temp_dir, output_csv, n_classes=3):
    """
    Reads the dataset of object detection and crops all the images found.

    :param image_dir: image directory
    :param csv_file: csv path
    :param output_dir: output directory
    :param temp_dir: temporary directory
    :param output_csv: output csv
    :param n_classes: number of classes
    """
    results = []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    for row in results:
        img = Image.open(image_dir + row['image'])

        s = re.sub('\[ +', '[', row['tags'].strip())
        s = re.sub('[,\s]+', ', ', s)
        tags = np.array(ast.literal_eval(s))

        if len(tags) >= 1:
            boxes = tags[:, 1:]
        else:
            boxes = []

        idx = 0
        for box in boxes:
            cropped = crop(img, box)
            crop_name = os.path.splitext(row['image'])[0] + str(idx) + '.txt'
            cropped.save(temp_dir + crop_name)
            idx += 1

    label(temp_dir, output_dir, output_csv, n_classes)


if __name__ == "__main__":
    crop_from_labeled_objects(image_dir="FILL WITH ABS PATH TO THE IMAGE DIRECTORY",
                              csv_file="FILL WITH ABS PATH TO OBJ DETECTION CSV",
                              output_dir="FILL_WITH_ABS_PATH",
                              temp_dir="./",
                              output_csv="""CREATE CSV AND FILL""")



