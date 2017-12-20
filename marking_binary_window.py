import flika
from flika.window import Window
from flika.utils.misc import save_file_gui, open_file_gui
from flika import global_vars as g
from skimage import measure
from skimage import morphology
from skimage.measure import label, find_contours
from qtpy import QtWidgets, QtCore
import numpy as np
import json, codecs

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

class Classifier_Window(Window):
    WHITE = np.array([True, True, True])
    BLACK = np.array([False, False, False])
    RED = np.array([True, False, False])
    GREEN = np.array([False, True, False])
    BLUE = np.array([False, False, True])
    PURPLE = np.array([True, False, True])
    YELLOW = np.array([True, True, False])

    def __init__(self, tif, name='flika', filename='', commands=[], metadata=dict()):
        tif = tif.astype(np.bool)
        super().__init__(tif, name, filename, commands, metadata)

        self.labeled_img = label(tif, connectivity=2)
        self.dapi_labeled_img = label(tif, connectivity=2)
        self.nROIs = np.max(self.labeled_img)
        self.eroded_labeled_img = label(tif, connectivity=2)
        self.roi_states = np.zeros(np.max(self.labeled_img), dtype=np.uint8)
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)
        self.imageview.setImage(self.colored_img)

        self.menu.addAction(QtWidgets.QAction("&Save Training Data", self, triggered=self.save_training_data))
        self.menu.addAction(QtWidgets.QAction("&Save Classifications", self, triggered=self.save_classifications))
        self.menu.addAction(QtWidgets.QAction("&Load Classifications", self, triggered=self.load_classifications_act))
        self.menu.addAction(QtWidgets.QAction("&Create Binary Window", self, triggered=self.create_binary_window))
        self.features_array = None
        # self.features_array_extended includes all features in self.features_array as well as features only calculated
        # for exporting.
        self.features_array_extended = None
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
                prop = self.props[roi_num]
                scaleFactor= g.myoquant.algorithm_gui.microns_per_pixel_SpinBox.value()
                print('ROI #{}. area={}. eccentricity={}. convexity={}. perimeter={}. minor_axis_length={}. '
                      .format(roi_num, prop.area, prop.eccentricity, prop.filled_area / prop.convex_area,
                              prop.perimeter, prop.minor_axis_length))
                old_state = self.roi_states[roi_num]
                # added a yellow color for testing: roi_state=3
                new_state = (old_state + 1 ) % 4
                self.roi_states[roi_num] = new_state
                color = [Classifier_Window.WHITE, Classifier_Window.GREEN, Classifier_Window.RED, Classifier_Window.YELLOW][new_state]
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

    def get_features_array(self):
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
            perimeter = np.array([p.perimeter for p in self.props])
            circularity = np.empty_like(perimeter)
            for i in np.arange(len(circularity)):
                if perimeter[i] == 0:
                    circularity[i] = 0
                else:
                    circularity[i] = (4 * np.pi * area[i]) / perimeter[i]**2
            self.features_array = np.array([area, eccentricity, convexity, circularity]).T
        return self.features_array

    def get_training_data(self):
        if self.features_array is None:
            self.features_array = self.get_features_array()
        states = np.array([np.asscalar(a)for a in self.roi_states])
        X = self.features_array[states > 0, :]
        y = states[states > 0]
        y[y == 2] = 0
        return X, y

    def get_extended_features_array(self):
        if self.features_array is None:
            self.features_array = self.get_features_array()

        min_ferets = np.array([g.myoquant.calc_min_feret_diameters(g.win.props)]).T
        roi_num = np.arange(self.nROIs)
        area = self.features_array[:,0]

        X = np.concatenate((roi_num[:, np.newaxis], area[:, np.newaxis], min_ferets), 1)
        if g.myoquant.intensity_img is not None and g.myoquant.labeled_img is not None:
            Y = measure.regionprops(g.myoquant.labeled_img, g.myoquant.intensity_img)
            mfi = np.array([p.mean_intensity for p in Y])
            X = np.concatenate((X, mfi[:,np.newaxis]),1)
        return X


    # if g.myoquant.intensity_img is not None:
    def save_classifications(self):
        filename = save_file_gui("Save classifications", filetypes='*.json')
        if filename is None:
            return None
        states = [np.asscalar(a)for a in self.roi_states]
        data = {'states': states}
        json.dump(data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)  ### this saves the array in .json format

    def save_training_data(self):
        filename = save_file_gui("Save training_data", filetypes='*.json')
        if filename is None:
            return None
        X, y = self.get_training_data()
        y = y.tolist()
        X = X.tolist()
        data = {'features': X, 'states': y}
        json.dump(data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)  ### this saves the array in .json format

    def create_binary_window(self):
        true_rois = self.roi_states == 1
        bin_im = np.zeros_like(self.image, dtype=np.uint8)
        for i in np.nonzero(true_rois)[0]:
            x, y = self.props[i].coords.T
            bin_im[x, y] = 1
        Window(bin_im, 'Binary')

    def load_classifications_act(self):
        self.load_classifications()

    def load_classifications(self, filename=None):
        if filename is None:
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
            g.myoquant.roiProps = self.props
        self.roi_states = roi_states
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)

        for i in np.nonzero(roi_states == 1)[0]:
            x, y = self.props[i].coords.T
            self.colored_img[x, y] = Classifier_Window.GREEN
        for i in np.nonzero(roi_states == 2)[0]:
            x, y = self.props[i].coords.T
            self.colored_img[x, y] = Classifier_Window.RED
        self.update_image(self.colored_img)

    def run_erosion(self):
        # Reset potentially old values=
        self.eroded_roi_states = None
        g.myoquant.overlapped_rois = None

        #Set all values in eroded_labeled_img to 0
        #The appropriate coordinates will be marked as 1 later
        self.eroded_labeled_img[:len(self.eroded_labeled_img - 1)] = 0
        for i in np.nonzero(self.roi_states == 1)[0]:
            # Reset the ROIs to green
            x, y = self.props[i].coords.T
            self.colored_img[x, y] = Classifier_Window.GREEN

            individualProp = self.props[i]

            targetSize = (100 - g.myoquant.algorithm_gui.erosion_percentage_SpinBox.value()) * .01
            targetArea = individualProp.area * targetSize

            if (targetArea > 10):
                while individualProp.area > targetArea:
                    label_im_1 = label(morphology.binary_erosion(individualProp.image, selem=None), connectivity=2)
                    newProps = measure.regionprops(label_im_1)

                    if (len(newProps) > 0):
                        tempProp = newProps[0]
                    else:
                        break

                    if (tempProp.area > 5):
                        individualProp = tempProp
                    else:
                        break

                #Calculate the difference between the original bbox and the new bbox
                originalX, originalY = self.props[i].centroid
                newX, newY = individualProp.centroid
                differenceX = newX - originalX
                differenceY = newY - originalY

                x, y = individualProp.coords.T

                #Updated each coordinate with an adjusted value, accounting for the difference
                for i in range(len(x)):
                    x[i] = x[i] + (-1 * differenceX)
                for i in range(len(y)):
                    y[i] = y[i] + (-1 * differenceY)
                self.eroded_labeled_img[x, y] = 1

        eroded_label = label(self.eroded_labeled_img, connectivity=2)
        g.myoquant.eroded_roi_states = measure.regionprops(eroded_label)
        g.myoquant.eroded_labeled_img = self.eroded_labeled_img
        g.myoquant.paintDapiColoredImage()