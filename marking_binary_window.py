import flika
from flika.window import Window
from flika.utils.misc import save_file_gui, open_file_gui
from flika import global_vars as g
from skimage import measure
from skimage.measure import label

from qtpy import QtWidgets, QtCore
import numpy as np
import json, codecs


class Classifier_Window(Window):
    RED = np.array([True, False, False])
    GREEN = np.array([False, True, False])
    WHITE = np.array([True, True, True])
    BLACK = np.array([False, False, False])

    def __init__(self, tif, name='flika', filename='', commands=[], metadata=dict()):
        tif = tif.astype(np.bool)
        super().__init__(tif, name, filename, commands, metadata)

        self.labeled_img = label(tif, connectivity=2)
        self.nROIs = np.max(self.labeled_img)
        self.roi_states = np.zeros(np.max(self.labeled_img), dtype=np.uint8)
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)
        self.imageview.setImage(self.colored_img)
        self.menu.addAction(QtWidgets.QAction("&Save Classifications", self, triggered=self.save_classifications))
        self.menu.addAction(QtWidgets.QAction("&Load Classifications", self, triggered=self.load_classifications))
        self.menu.addAction(QtWidgets.QAction("&Create Binary Window", self, triggered=self.create_binary_window))
        self.features_array = None
        self.props = None

    def mouseClickEvent(self, ev):
        if self.props is None:
            self.props = measure.regionprops(self.labeled_img)
        if ev.button() == 1:
            x = int(self.x)
            y = int(self.y)
            try:
                roi_num = self.labeled_img[x, y] - 1
            except IndexError:
                roi_num = -1
            if roi_num < 0:
                pass
            else:
                print('ROI #{}'.format(roi_num))
                old_state = self.roi_states[roi_num]
                new_state = (old_state + 1 ) % 3
                self.roi_states[roi_num] = new_state
                color = [Classifier_Window.WHITE, Classifier_Window.GREEN, Classifier_Window.RED][new_state]
                x, y = self.props[roi_num].coords.T
                self.colored_img[x, y] = color
                self.update_image(self.colored_img)
        super().mouseClickEvent(ev)

    def update_image(self, image):
        viewrange = self.imageview.getView().viewRange()
        xrange, yrange = viewrange
        self.imageview.setImage(image)
        self.imageview.getView().setXRange(xrange[0], xrange[1], 0, False)
        self.imageview.getView().setYRange(yrange[0], yrange[1], 0)

    def get_training_data(self):
        # important features include:
        # convexity: ratio of convex_image area to image area
        # area: number of pixels total
        # eccentricity: 0 is a circle, 1 is a line
        if self.features_array is None:
            if self.props is None:
                self.props = measure.regionprops(self.labeled_img)
            area = np.array([p.filled_area for p in self.props])
            eccentricity = np.array([p.eccentricity for p in self.props])
            convexity = np.array([p.filled_area / p.convex_area for p in self.props])
            self.features_array = np.array([area, eccentricity, convexity]).T
        states = np.array([np.asscalar(a)for a in self.roi_states])
        X = self.features_array[states > 0, :]
        y = states[states > 0]
        y[y == 2] = 0
        return X, y

    def save_classifications(self):
        filename = save_file_gui("Save classifications", filetypes='*.json')
        if filename is None:
            return None
        states = [np.asscalar(a)for a in self.roi_states]
        data = {'states': states}
        json.dump(data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)  ### this saves the array in .json format

    def create_binary_window(self):
        true_rois = self.roi_states == 1
        bin_im = np.zeros_like(self.image, dtype=np.uint8)
        for i in np.nonzero(true_rois)[0]:
            x, y = self.props[i].coords.T
            bin_im[x, y] = 1
        Window(bin_im, 'Binary')

    def load_classifications(self):
        filename = open_file_gui("Open classifications", filetypes='*.json')
        if filename is None:
            return None
        obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
        data = json.loads(obj_text)
        roi_states = np.array(data['states'])
        if len(roi_states) != self.nROIs:
            g.alert('The number of ROIs in this file does not match the number of ROIs in the image. Cannot import classifications')
        else:
            self.set_roi_states(roi_states)

    def set_roi_states(self, roi_states):
        if self.props is None:
            self.props = measure.regionprops(self.labeled_img)
        self.roi_states = roi_states
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)

        for i in np.nonzero(self.roi_states == 1)[0]:
            x, y = self.props[i].coords.T
            self.colored_img[x, y] = Classifier_Window.GREEN
        for i in np.nonzero(self.roi_states == 2)[0]:
            x, y = self.props[i].coords.T
            self.colored_img[x, y] = Classifier_Window.RED
        self.update_image(self.colored_img)



#self = Classifier_Window(g.currentWindow.image)





