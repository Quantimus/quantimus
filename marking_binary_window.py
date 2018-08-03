from flika.window import Window
from flika.utils.misc import save_file_gui, open_file_gui
from flika import global_vars as g
from skimage import measure
from skimage import morphology
from skimage.measure import label
from skimage.morphology import diamond
from qtpy import QtWidgets
import numpy as np
import json
import codecs


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


class ClassifierWindow(Window):
    WHITE = np.array([True, True, True])
    BLACK = np.array([False, False, False])
    RED = np.array([True, False, False])
    GREEN = np.array([False, True, False])
    BLUE = np.array([False, False, True])
    PURPLE = np.array([True, False, True])
    YELLOW = np.array([True, True, False])

    TRAINING = "TRAINING"
    DAPI = "DAPI"
    FLR = "FLOURESCENCE"

    def __init__(self, tif, name='flika', filename='', commands=None, metadata=None):
        if commands is None:
            commands = []
        if metadata is None:
            metadata = dict()

        tif = tif.astype(np.bool)
        super().__init__(tif, name, filename, commands, metadata)

        # Window images
        self.imageIdentifier = None
        self.labeled_img = label(tif, connectivity=2)
        self.eroded_labeled_img = label(tif, connectivity=2)
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)
        self.imageview.setImage(self.colored_img)

        # Window specific ROI and States
        self.window_props = None
        self.window_states = None
        self.temp_props = None
        self.temp_states = None

        # GUI Actions
        self.menu.addAction(QtWidgets.QAction("&Save Training Data", self, triggered=self.save_training_data))
        self.menu.addAction(QtWidgets.QAction("&Save Classifications", self, triggered=self.save_classifications))
        self.menu.addAction(QtWidgets.QAction("&Load Classifications", self, triggered=self.load_classifications_act))
        self.menu.addAction(QtWidgets.QAction("&Create Binary Window", self, triggered=self.create_binary_window))
        self.features_array = None
        self.features_array_extended = None

    def mouseClickEvent(self, ev):
        if self.window_props is None:
            self.window_props = measure.regionprops(self.labeled_img)
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
                prop = self.window_props[roi_num]

                mfi = 'Unknown'
                if g.quantimus.flourescenceIntensities is not None:
                    mfi = g.quantimus.flourescenceIntensities[roi_num]

                print('ROI #{}. area={}. eccentricity={}. convexity={}. circularity={}. perimeter={}. minor_axis_length={}. MFI={}. '
                      .format(roi_num,
                              prop.area,
                              prop.eccentricity,
                              prop.area / prop.convex_area,
                              (4 * np.pi * prop.area) / (prop.perimeter * prop.perimeter),
                              prop.perimeter,
                              prop.minor_axis_length,
                              mfi))

                # Different windows have different MouseClickEvent logic
                if self.imageIdentifier == ClassifierWindow.TRAINING:
                    x, y = self.window_props[roi_num].coords.T
                    color, state = self.training_mouse_click_event(roi_num)
                    self.window_states[roi_num] = state
                    self.colored_img[x, y] = color
                    self.update_image(self.colored_img)
                elif self.imageIdentifier == ClassifierWindow.FLR:
                    if self.temp_states is None:
                        self.temp_states = np.copy(self.window_states)
                    x, y = self.window_props[roi_num].coords.T
                    color, state = self.flr_mouse_click_event(roi_num)
                    self.temp_states[roi_num] = state
                    self.colored_img[x, y] = color
                    self.update_image(self.colored_img)
                    self.update_parent_image(roi_num, x, y, self.temp_states)
                elif self.imageIdentifier == ClassifierWindow.DAPI:
                    x, y = self.window_props[roi_num].coords.T
                    color, state = self.dapi_mouse_click_event(roi_num)
                    self.window_states[roi_num] = state
                    self.colored_img[x, y] = color
                    self.update_image(self.colored_img)
                    self.update_parent_image(roi_num, x, y, self.window_states)

        super().mouseClickEvent(ev)

    def training_mouse_click_event(self, roi_num):
        old_state = self.window_states[roi_num]
        new_state = (old_state + 1) % 3
        color = [ClassifierWindow.WHITE, ClassifierWindow.GREEN, ClassifierWindow.RED][new_state]
        return color, new_state

    def flr_mouse_click_event(self, roi_num):
        old_state = self.temp_states[roi_num]
        new_state = (old_state + 1) % 4
        # Skip White. There is no need to have White in Flourescence
        if new_state == 0:
            new_state = 1
        color = [ClassifierWindow.WHITE, ClassifierWindow.GREEN, ClassifierWindow.RED, ClassifierWindow.YELLOW][new_state]
        return color, new_state

    def dapi_mouse_click_event(self, roi_num):
        old_state = self.window_states[roi_num]
        new_state = (old_state + 1) % 4
        # Skip White. There is no need to have White in DAPI
        if new_state == 0:
            new_state = 1
        color = [ClassifierWindow.WHITE, ClassifierWindow.GREEN, ClassifierWindow.RED, ClassifierWindow.PURPLE][new_state]
        return color, new_state

    def update_image(self, image):
        viewrange = self.imageview.getView().viewRange()
        xrange, yrange = viewrange
        self.imageview.setImage(image)
        self.imageview.getView().setXRange(xrange[0], xrange[1], 0, False)
        self.imageview.getView().setYRange(yrange[0], yrange[1], 0)

    def update_parent_image(self, roi_num, x, y, states):
        # Update the Parent window's colors
        if g.quantimus.filtered_trained_img is not None:
            # Update the Filtered Trained Image if available
            try:
                trained_color, trained_state = self.filtered_mouse_click_event(roi_num, states)
                g.quantimus.filtered_trained_img.window_states[roi_num] = trained_state
                g.quantimus.filtered_trained_img.colored_img[x, y] = trained_color
                g.quantimus.filtered_trained_img.update_image(g.quantimus.filtered_trained_img.colored_img)
                # Update the Parent window's States
                g.quantimus.roiStates[roi_num] = trained_state
            except AttributeError:
                print("No Parent Filtered Trained Image to Update")
        elif g.quantimus.trained_img is not None:
            # Update the Trained Image if available
            try:
                trained_color, trained_state = self.filtered_mouse_click_event(roi_num, states)
                g.quantimus.trained_img.window_states[roi_num] = trained_state
                g.quantimus.trained_img.colored_img[x, y] = trained_color
                g.quantimus.trained_img.update_image(g.quantimus.trained_img.colored_img)
                # Update the Parent window's States
                g.quantimus.roiStates[roi_num] = trained_state
            except AttributeError:
                print("No Parent Trained Image to Update")
        else:
            print("There are no Parent Images open and available for updating")

    def filtered_mouse_click_event(self, roi_num, states):
        new_state = states[roi_num]
        # Skip White and Purple
        if new_state == 3:
            new_state = 1
        color = [ClassifierWindow.WHITE, ClassifierWindow.GREEN, ClassifierWindow.RED, ClassifierWindow.PURPLE][new_state]
        return color, new_state

    def get_features_array(self):
        # important features include:
        # convexity: ratio of convex_image area to image area
        # area: number of pixels total
        # eccentricity: 0 is a circle, 1 is a line
        if self.features_array is None:
            if self.window_props is None:
                self.window_props = measure.regionprops(self.labeled_img)
            area = np.array([p.filled_area for p in self.window_props])
            eccentricity = np.array([p.eccentricity for p in self.window_props])
            convexity = np.array([p.filled_area / p.convex_area for p in self.window_props])
            perimeter = np.array([p.perimeter for p in self.window_props])
            circularity = np.empty_like(perimeter)

            for i in np.arange(len(circularity)):
                if perimeter[i] == 0:
                    circularity[i] = 0
                else:
                    circularity[i] = (4 * np.pi * area[i]) / perimeter[i]**2
            self.features_array = np.array([area, eccentricity, convexity, circularity]).T
        return self.features_array

    def calculate_window_props(self):
        if self.window_props is None:
            self.window_props = measure.regionprops(self.labeled_img)

    def get_training_data(self):
        if self.features_array is None:
            self.features_array = self.get_features_array()
        states = np.array([np.asscalar(a)for a in self.window_states])
        x = self.features_array[states > 0, :]
        y = states[states > 0]
        y[y == 2] = 0
        return x, y

    def get_extended_features_array(self):
        if self.features_array is None:
            self.features_array = self.get_features_array()

        min_ferets = np.array([g.quantimus.calc_min_feret_diameters(g.win.props)]).T
        roi_num = np.arange(self.window_states)
        area = self.features_array[:, 0]

        x = np.concatenate((roi_num[:, np.newaxis], area[:, np.newaxis], min_ferets), 1)
        if g.quantimus.intensity_img is not None and g.quantimus.flourescence_img is not None:
            y = measure.regionprops(g.quantimus.flourescence_img, g.quantimus.intensity_img)
            mfi = np.array([p.mean_intensity for p in y])
            x = np.concatenate((x, mfi[:, np.newaxis]), 1)
        return x

    def save_classifications(self):
        filename = save_file_gui("Save classifications", filetypes='*.json')
        if filename is None:
            return None
        states = [np.asscalar(a)for a in self.window_states]
        data = {'states': states}
        # this saves the array in .json format
        json.dump(data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    def save_training_data(self):
        progress = g.quantimus.create_progress_bar('Please wait training data is being saved...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        filename = save_file_gui("Save training_data", filetypes='*.json')
        if filename is None:
            return None
        x, y = self.get_training_data()
        y = y.tolist()
        x = x.tolist()
        data = {'features': x, 'states': y}
        # this saves the array in .json format
        json.dump(data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    def create_binary_window(self):
        true_rois = self.window_states == 1
        bin_im = np.zeros_like(self.image, dtype=np.uint8)
        for i in np.nonzero(true_rois)[0]:
            x, y = self.window_props[i].coords.T
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

        if len(roi_states) != len(self.window_states):
            g.alert('The number of ROIs in this file does not match the number of ROIs in the image. Cannot import classifications')
        else:
            g.quantimus.roiStates = np.copy(roi_states)
            self.window_states = np.copy(roi_states)
            self.set_roi_states()

    def set_roi_states(self):
        if self.window_props is None:
            self.window_props = measure.regionprops(self.labeled_img)
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)
        for i in np.nonzero(self.window_states == 1)[0]:
            x, y = self.window_props[i].coords.T
            self.colored_img[x, y] = ClassifierWindow.GREEN
        for i in np.nonzero(self.window_states == 2)[0]:
            x, y = self.window_props[i].coords.T
            self.colored_img[x, y] = ClassifierWindow.RED
        for i in np.nonzero(self.window_states == 3)[0]:
            x, y = self.window_props[i].coords.T
            if self.imageIdentifier == ClassifierWindow.DAPI:
                self.colored_img[x, y] = ClassifierWindow.PURPLE
            elif self.imageIdentifier == ClassifierWindow.FLR:
                self.colored_img[x, y] = ClassifierWindow.BLUE
            else:
                self.colored_img[x, y] = ClassifierWindow.GREEN
        self.update_image(self.colored_img)

    def run_erosion(self):
        progress = g.quantimus.create_progress_bar('Please wait while fibers are being eroded...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        for i in np.nonzero(self.window_states == 3)[0]:
            self.window_states[i] = 1
        g.quantimus.saved_dapi_states = None
        # Set all values in eroded_labeled_img to 0
        # The appropriate coordinates will be marked as 1 later
        self.eroded_labeled_img[:len(self.eroded_labeled_img - 1)] = 0

        for state in np.nonzero(self.window_states == 1)[0]:
            QtWidgets.QApplication.processEvents()

            # Reset the ROIs to green
            x, y = self.window_props[state].coords.T
            self.colored_img[x, y] = ClassifierWindow.GREEN

            individualprop = self.window_props[state]
            targetsize = (100 - g.quantimus.algorithm_gui.erosion_percentage_SpinBox.value()) * .01
            targetarea = individualprop.area * targetsize

            if targetarea > 10:
                image = individualprop.image
                while True:
                    # Get the size of the array
                    arrayrows = len(image)
                    arraycols = len(image[0])
                    arraysize = arrayrows*arraycols
                    # Count the number of 'false' in the array
                    arrayfalsecount = (image == False).sum()
                    # Subtract the two to get the positive area
                    erodedarea = arraysize - arrayfalsecount

                    if erodedarea > targetarea:
                        image = morphology.binary_erosion(image, selem=diamond(1))
                    else:
                        break

                # Get the coordinate for the pixel at the upper left of the bbox for the original ROI Prop image
                originalx = individualprop.bbox[0]
                originaly = individualprop.bbox[1]
                erodedx = []
                erodedy = []

                for i in range(len(image)):
                    for j in range(len(image[i])):
                        if image[i][j]:
                            erodedx.append(i + originalx)
                            erodedy.append(j + originaly)

                self.eroded_labeled_img[erodedx, erodedy] = 1

        eroded_label = label(self.eroded_labeled_img, connectivity=2)
        g.quantimus.eroded_roi_states = measure.regionprops(eroded_label)
        g.quantimus.eroded_labeled_img = self.eroded_labeled_img
        g.quantimus.paint_dapi_colored_image()
