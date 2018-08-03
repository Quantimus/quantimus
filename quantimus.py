import os
import scipy
from distutils.version import StrictVersion
import xlsxwriter
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d
from qtpy import uic, QtGui
from skimage.morphology import binary_dilation
from itertools import chain
from sklearn import svm
import pyqtgraph as pg
import math
import flika

from .marking_binary_window import *

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox


def show_label_img(binary_img):
    newlabel = label(binary_img, connectivity=2)
    image = np.zeros((np.max(newlabel), newlabel.shape[0], newlabel.shape[1]), dtype=np.bool)
    for i in np.arange(1, np.max(newlabel)):
        image[i - 1] = newlabel == i
    return Window(image)


def get_important_features(binary_image):
    features = {}
    label_img = label(binary_image, connectivity=2)
    props = measure.regionprops(label_img)
    features['convexity'] = np.array([p.filled_area / p.convex_area for p in props])
    features['eccentricity'] = np.array([p.eccentricity for p in props])
    features['area'] = np.array([p.filled_area for p in props]) / 4000
    features['circularity'] = np.array([p.filled_area * (4 * np.pi) / p.perimeter ** 2 for p in props])
    return features


def remove_borders(binary_image):
    label_img = label(binary_image, connectivity=2)
    mx, my = binary_image.shape
    border_labels = set(label_img[:, 0]) | set(label_img[0, :]) | set(label_img[mx - 1, :]) | set(label_img[:, my - 1])
    for i in border_labels:
        binary_image[label_img == i] = 0
    return binary_image


def remove_false_positives(binary_window, features):
    label_img = binary_window.labeled_img
    elements = np.max(label_img)
    for i in np.arange(elements):
        if features['area'][i] < .05:
            binary_window.roi_states[i] = 2
        elif features['convexity'][i] < .7:
            binary_window.roi_states[i] = 2
        elif features['eccentricity'][i] > .96 and features['convexity'][i] < .85:
            binary_window.roi_states[i] = 2
        elif features['area'][i] > 3:
            binary_window.roi_states[i] = 2
        elif features['circularity'][i] < 0.4:
            binary_window.roi_states[i] = 2
        else:
            binary_window.roi_states[i] = 1
    binary_window.colored_img = np.repeat(binary_window.image[:, :, np.newaxis], 3, 2)
    binary_window.colored_img[binary_window.image == 1] = ClassifierWindow.GREEN
    for roi_num, new_state in enumerate(binary_window.roi_states):
        if new_state != 1:
            color = [ClassifierWindow.WHITE, ClassifierWindow.GREEN, ClassifierWindow.RED][new_state]
            binary_window.colored_img[binary_window.labeled_img == roi_num + 1] = color
    binary_window.update_image(binary_window.colored_img)


def generate_kernel(theta=0):
    frequency = .1
    sigma_x = 1  # left right axis. Bigger this number, smaller the width
    sigma_y = 2  # right left axis. Bigger this number, smaller the height
    kernel = np.real(gabor_kernel(frequency, theta, sigma_x, sigma_y))
    kernel -= np.mean(kernel)
    return kernel


def get_kernels():
    # prepare filter bank kernels
    kernels = []
    for theta in np.linspace(0, np.pi, 40):
        kernel = generate_kernel(theta)
        kernels.append(kernel)
    return kernels


kernels = get_kernels()


def convolve_with_kernels_fft(i, kernels):
    results = []
    for k, kernel in enumerate(kernels):
        print(k)
        filtered = scipy.signal.fftconvolve(i, kernel, 'same')
        results.append(filtered)
    results = np.array(results)
    return results


def plot_regression_results(xparam1, xparam2, y):
    p = pg.plot()
    x1 = xparam1[y == 1]
    x2 = xparam2[y == 1]
    s1 = pg.ScatterPlotItem(x1, x2, size=10, pen=None, brush=pg.mkBrush(0, 255, 0, 255))
    p.addItem(s1)
    x1 = xparam1[y == 0]
    x2 = xparam2[y == 0]
    s1.addPoints(x1, x2, size=10, pen=None, brush=pg.mkBrush(255, 0, 0, 255))


def get_border_between_two_props(prop1, prop2):
    image2 = prop2.image
    image1 = np.zeros_like(image2)
    bbox = np.array(prop2.bbox)
    top_left = bbox[:2]
    a = prop1.coords - top_left
    image1[a[:, 0], a[:, 1]] = 1
    image1_expanded1 = binary_dilation(image1)
    image1_expanded2 = binary_dilation(binary_dilation(image1_expanded1))
    image1_expanded2[image1_expanded1] = 0
    border = image1_expanded2 * image2
    return np.argwhere(border) + top_left


def get_new_image(image, thresh1=.20, thresh2=.30):
    resizefactor = g.quantimus.algorithm_gui.resize_factor_SpinBox.value()
    label_im_1 = label(image < thresh1, connectivity=2)
    label_im_2 = label(image < thresh2, connectivity=2)
    props_1 = measure.regionprops(label_im_1)
    props_2 = measure.regionprops(label_im_2)
    borders = np.zeros_like(image)

    #  The maximum of the labeled image is the number of contiguous regions, or ROIs.
    rois = np.max(label_im_1)
    for roi_num in np.arange(rois):
        QtWidgets.QApplication.processEvents()
        prop1 = props_1[roi_num]
        x, y = prop1.coords[0, :]
        prop2 = props_2[label_im_2[x, y] - 1]

        if prop1.area > 65 * resizefactor ** 2:
            area_ratio = prop2.area / prop1.area
            if area_ratio > 1.2:
                border_idx = get_border_between_two_props(prop1, prop2)
                borders[border_idx[:, 0], border_idx[:, 1]] = 1
    image_new = np.copy(image)
    image_new[np.where(borders)] = 2
    return image_new


class Quantimus:
    """
    Muscle Cell Analysis Software
    """

    MARKERS = "MARKERS"
    BINARY = "BINARY"

    def __init__(self):
        # Windows
        self.markers_win = None
        self.filled_boundaries_win = None
        self.binary_img = None
        self.classifier_window = None
        self.trained_img = None
        self.filtered_trained_img = None
        self.dapi_img = None
        self.dapi_binarized_img = None
        self.eroded_labeled_img = None
        self.flourescence_img = None
        self.intensity_img = None

        # ROIs and States
        self.roiStates = None
        self.eroded_roi_states = None
        self.dapi_rois = None
        self.roiProps = None
        self.flourescenceIntensities = None
        self.positiveFiberRois = None
        self.positiveFiberStates = None

        # Printing Data
        self.saved_flourescence_rois = None
        self.saved_flourescence_states = None
        self.saved_dapi_rois = None
        self.saved_dapi_states = None
        self.saved_positive_rois = None
        self.saved_positive_states = None

        # Misc
        self.isMarkersFirstSelection = True
        self.isBinaryFirstSelection = True
        self.isIntensityCalculated = False

        # GUI
        self.algorithm_gui = None
        self.original_window_selector = None
        self.threshold1_slider = None
        self.threshold2_slider = None
        self.binary_img_selector = None
        self.intensity_img_selector = None
        self.flourescence_img_selector = None
        self.dapi_img_selector = None
        self.binarized_dapi_img_selector = None

        pass

    def gui(self):

        # GUI Setup
        gui = uic.loadUi(os.path.join(os.path.dirname(__file__), 'quantimus.ui'))
        self.algorithm_gui = gui
        gui.show()
        self.original_window_selector = WindowSelector()
        self.original_window_selector.valueChanged.connect(self.create_markers_win)
        gui.gridLayout_18.addWidget(self.original_window_selector)
        self.threshold1_slider = SliderLabel(3)
        self.threshold1_slider.setRange(0, 1)
        self.threshold1_slider.setValue(.2)
        self.threshold1_slider.valueChanged.connect(self.threshold_slider_changed)
        self.threshold2_slider = SliderLabel(2)
        self.threshold2_slider.setRange(0, 1)
        self.threshold2_slider.setValue(.4)
        self.threshold2_slider.valueChanged.connect(self.threshold_slider_changed)
        gui.gridLayout_threshold_one.addWidget(self.threshold1_slider)
        gui.gridLayout_threshold_two.addWidget(self.threshold2_slider)
        gui.fill_boundaries_button.pressed.connect(self.fill_boundaries_button)
        gui.SVM_button.pressed.connect(self.run_svm_classification_on_image)
        gui.SVM_saved_button.pressed.connect(self.run_svm_classification_on_saved_training_data)
        gui.load_classification_button.pressed.connect(self.load_classification_to_trained_image)
        gui.manual_filter_button.pressed.connect(self.filter_update)

        self.binary_img_selector = WindowSelector()
        self.binary_img_selector.valueChanged.connect(self.select_binary_image)
        gui.gridLayout_import_binary_image.addWidget(self.binary_img_selector)

        self.intensity_img_selector = WindowSelector()
        self.intensity_img_selector.valueChanged.connect(self.select_intensity_image)
        gui.gridLayout_intensity_image.addWidget(self.intensity_img_selector)

        self.flourescence_img_selector = WindowSelector()
        self.flourescence_img_selector.valueChanged.connect(self.select_flourescence_image)
        gui.gridLayout_flourescence_image.addWidget(self.flourescence_img_selector)

        self.dapi_img_selector = WindowSelector()
        self.dapi_img_selector.valueChanged.connect(self.select_dapi_image)
        gui.gridLayout_import_DAPI.addWidget(self.dapi_img_selector)

        self.binarized_dapi_img_selector = WindowSelector()
        self.binarized_dapi_img_selector.valueChanged.connect(self.select_dapi_binarized_image)
        gui.gridLayout_contains_DAPI.addWidget(self.binarized_dapi_img_selector)

        gui.run_DAPI_button.pressed.connect(self.calculate_dapi)
        gui.save_DAPI_button.pressed.connect(self.save_dapi)
        gui.run_Flr_button.pressed.connect(self.calculate_flourescence)
        gui.save_flourescence_button.pressed.connect(self.save_flourescence)
        gui.print_button.pressed.connect(self.print_data)

        gui.determine_positives_button.pressed.connect(self.determine_positives)
        gui.measure_positives_button.pressed.connect(self.measure_positives)
        gui.clear_positives_button.pressed.connect(self.clear_positives)
        gui.save_positives_button.pressed.connect(self.save_positives)

        gui.closeEvent = self.close_event

    def create_markers_win(self):
        if self.original_window_selector.window is None:
            g.alert('You must select a Window before creating the markers window.')
        else:
            if self.reset_data(Quantimus.MARKERS):
                win = self.original_window_selector.window
                needalert = False
                if np.max(win.image) > 1:
                    needalert = True
                    image = win.image.astype(np.float)
                    image -= np.min(image)
                    image /= np.max(image)
                    win.image = image
                    win.dtype = image.dtype
                    win.imageview.setImage(win.image)
                    win._init_dimensions(win.image)
                    win.imageview.ui.graphicsView.addItem(win.top_left_label)
                original = win.image
                self.markers_win = Window(np.zeros_like(original, dtype=np.uint8), 'Binary Markers')
                self.markers_win.imageview.setLevels(0, 2)
                self.markers_win.imageview.ui.histogram.gradient.addTick(0, QtGui.QColor(0, 0, 255), True)
                self.markers_win.imageview.ui.histogram.gradient.setTickValue(1, .50)
                self.threshold1_slider.setRange(np.min(original), np.max(original))
                self.threshold2_slider.setRange(np.min(original), np.max(original))
                self.threshold_slider_changed()
                if needalert:
                    g.alert("The window you select must have values between 0 and 1. Scaling the window now.")
                self.isMarkersFirstSelection = False

    def threshold_slider_changed(self):
        if self.original_window_selector.window is None:
            g.alert('You must select a Window before adjusting the levels.')
        else:
            thresh1 = self.threshold1_slider.value()
            thresh2 = self.threshold2_slider.value()
            image = self.original_window_selector.window.image
            markers = (image > thresh1).astype(dtype=np.uint8)
            markers[image > thresh2] = 2
            self.markers_win.imageview.setImage(markers, autoRange=False, autoLevels=False)

    def fill_boundaries_button(self):
        # Reset any data currently saved in the system
        lower_bound = self.threshold1_slider.value()
        upper_bound = self.threshold2_slider.value()
        # Original linspace = 8
        thresholds = np.linspace(lower_bound, upper_bound, 8)
        image = self.original_window_selector.window.image
        image_new = image

        label_im_1 = label(image < lower_bound, connectivity=2)
        label_im_2 = label(image < upper_bound, connectivity=2)
        props_2 = measure.regionprops(label_im_2)

        progress = self.create_progress_bar('Please wait while image is processed...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        for i in np.arange(len(thresholds) - 1):
            QtWidgets.QApplication.processEvents()
            print(thresholds[i])
            image_new = get_new_image(image_new, thresholds[i], thresholds[i + 1])

        # Remove ROIs that
        rois_2 = np.max(label_im_2)
        for roi_num in np.arange(rois_2):
            prop2 = props_2[roi_num]
            x, y = prop2.coords.T
            if np.max(label_im_1[x, y]) == 0:
                image_new[x, y] = 2

        self.filled_boundaries_win = Window(image_new, 'Filled Boundaries')
        classifier_image = remove_borders(image_new < upper_bound)
        self.binary_img = ClassifierWindow(classifier_image, 'Binary Window')

    def get_norm_coeffs(self, x):
        mean = np.mean(x, 0)
        std = np.std(x, 0)
        return mean, std

    def normalize_data(self, x, mean, std):
        x = x - mean
        x = x / (2 * std)
        return x

    def close_event(self, event):
        print('Closing quantimus gui')
        if self.classifier_window is not None:
            self.classifier_window.close()
        event.accept()  # let the window close

    def create_progress_bar(self, msg):
        progress = QtWidgets.QProgressDialog()
        progress.parent = self
        progress.setLabelText(msg)
        progress.setRange(0, 0)
        progress.setMinimumWidth(375)
        progress.setMinimumHeight(100)
        progress.setCancelButton(None)
        progress.setModal(True)
        return progress

    def select_binary_image(self):
        # Reset any data currently saved in the system
        if self.reset_data(Quantimus.BINARY):
            print('Binary image selected.')
            self.classifier_window = ClassifierWindow(self.binary_img_selector.window.image, 'Training Image')
            self.classifier_window.imageIdentifier = ClassifierWindow.TRAINING
            self.roiStates = np.zeros(np.max(self.classifier_window.labeled_img), dtype=np.uint8)
            self.classifier_window.window_states = np.copy(self.roiStates)
            self.isBinaryFirstSelection = False

    def run_svm_classification_on_image(self):
        if self.classifier_window is None:
            g.alert("Please select a Binary Image")
        else:
            # Start threading and Progress Bar
            progress = g.quantimus.create_progress_bar('Please wait while fibers are being classified...')
            progress.show()
            QtWidgets.QApplication.processEvents()

            x_train, y_train = self.classifier_window.get_training_data()
            mu, sigma = self.get_norm_coeffs(self.classifier_window.features_array)
            self.run_svm_classification_general(x_train, y_train, mu, sigma)

    def run_svm_classification_on_saved_training_data(self):
        if self.classifier_window is None:
            g.alert("Please select a Binary Image")
        else:
            filename = open_file_gui("Open training_data", filetypes='*.json')
            if filename is None:
                return None
            obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
            data = json.loads(obj_text)

            # Start threading and Progress Bar
            progress = self.create_progress_bar('Please wait while fibers are being classified...')
            progress.show()
            QtWidgets.QApplication.processEvents()

            x_train = np.array(data['features'])
            y_train = np.array(data['states'])
            mu, sigma = self.get_norm_coeffs(x_train)
            self.run_svm_classification_general(x_train, y_train, mu, sigma)

    def run_svm_classification_general(self, x_train, y_train, mu, sigma):
        print('Running SVM classification')
        try:
            x_train = self.normalize_data(x_train, mu, sigma)
            clf = svm.SVC()
            clf.fit(x_train, y_train)
            x_test = self.normalize_data(self.classifier_window.get_features_array(), mu, sigma)
            y = clf.predict(x_test)
            self.roiStates = np.zeros_like(y)
            self.roiStates[y == 1] = 1
            self.roiStates[y == 0] = 2
            self.trained_img = ClassifierWindow(self.classifier_window.image, 'Trained Image')
            self.trained_img.imageIdentifier = ClassifierWindow.TRAINING
            self.trained_img.window_states = np.copy(self.roiStates)

            # Add hand-designed rules here if you want.
            # For instance, you could remove all ROIs smaller than 15 pixels like this:

            # X = self.classifier_window.features_array
            # roi_states[X[:, 0] < 15] = 2 # Area must be smaller than 15 pixels
            # roi_states[X[:, 3] < 0.6] = 2 # Convexity must be smaller than 0.6

            self.trained_img.set_roi_states()
            self.roiStates = np.copy(self.trained_img.window_states)
        except ValueError:
            g.alert('Please train a minimum of 1 positive and 1 negative sample')

    def load_classification_to_trained_image(self):
        print('Loading Classification to Trained Image')

        progress = self.create_progress_bar('Please wait while fibers are being classified...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        self.trained_img = ClassifierWindow(self.classifier_window.image, 'Trained Image')
        self.trained_img.imageIdentifier = ClassifierWindow.TRAINING
        self.trained_img.window_states = np.copy(self.roiStates)
        self.trained_img.load_classifications_act()
        self.trained_img.set_roi_states()

    def filter_update(self):
        print('Manually filtering...')

        progress = self.create_progress_bar('Please wait while image is filtered...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            min_circularity = g.quantimus.algorithm_gui.min_circularity_SpinBox.value()
            max_circularity = g.quantimus.algorithm_gui.max_circularity_SpinBox.value()
            circularitycheckbox = g.quantimus.algorithm_gui.circularity_CheckBox
            min_area = g.quantimus.algorithm_gui.min_area_SpinBox.value()
            max_area = g.quantimus.algorithm_gui.max_area_SpinBox.value()
            areacheckbox = g.quantimus.algorithm_gui.area_CheckBox
            min_convexity = g.quantimus.algorithm_gui.min_convexity_SpinBox.value()
            max_convexity = g.quantimus.algorithm_gui.max_convexity_SpinBox.value()
            convexitycheckbox = g.quantimus.algorithm_gui.convexity_CheckBox
            min_eccentricity = g.quantimus.algorithm_gui.min_eccentricity_SpinBox.value()
            max_eccentricity = g.quantimus.algorithm_gui.max_eccentricity_SpinBox.value()
            eccentricitycheckbox = g.quantimus.algorithm_gui.eccentricity_CheckBox

            features = self.trained_img.get_features_array()
            states = np.copy(self.trained_img.window_states)
            count = 0

            for feature in features:
                # Update the progress bar so it shows movement
                QtWidgets.QApplication.processEvents()
                if self.trained_img.window_states[count] == 1:
                    # Area
                    if areacheckbox.isChecked():
                        if feature[0] >= min_area and feature[0] <= max_area:
                            states[count] = 1
                        else:
                            states[count] = 2
                    # Eccentricity
                    if eccentricitycheckbox.isChecked():
                        if states[count] == 1 and feature[1] >= min_eccentricity and feature[1] <= max_eccentricity:
                            states[count] = 1
                        else:
                            states[count] = 2
                    # Convexity
                    if convexitycheckbox.isChecked():
                        if states[count] == 1 and feature[2] >= min_convexity and feature[2] <= max_convexity:
                            states[count] = 1
                        else:
                            states[count] = 2
                    # Circularity
                    if circularitycheckbox.isChecked():
                        if states[count] == 1 and feature[3] >= min_circularity and feature[3] <= max_circularity:
                            states[count] = 1
                        else:
                            states[count] = 2
                else:
                    states[count] = 2
                count += 1

            self.filtered_trained_img = ClassifierWindow(self.trained_img.image, 'Filtered Trained Image')
            self.filtered_trained_img.imageIdentifier = ClassifierWindow.TRAINING
            self.filtered_trained_img.window_states = states
            self.filtered_trained_img.set_roi_states()
            self.roiStates = np.copy(self.filtered_trained_img.window_states)
        except AttributeError:
            g.alert('Please run the SVM Classification Training')

    def select_flourescence_image(self):
        print('Flourescence image selected.')
        # Reset potentially old data
        self.reset_flourescence_data()
        self.flourescence_img = None
        # Select the image
        self.flourescence_img = ClassifierWindow(self.flourescence_img_selector.window.image, 'Flourescence Image')
        self.flourescence_img.imageIdentifier = None
        self.flourescence_img.window_states = np.copy(self.flourescence_img_selector.window.window_states)
        self.paint_flr_colored_image()

    def select_intensity_image(self):
        print('Intensity image selected.')
        # Reset potentially old data
        self.reset_flourescence_data()
        # Select the image
        self.intensity_img = self.intensity_img_selector.window.image
        self.flourescence_img.set_bg_im()
        self.flourescence_img.bg_im_dialog.setWindowTitle("Select an image")
        if self.flourescence_img.bg_im_dialog.parent.bg_im is not None:
            self.flourescence_img.bg_im_dialog.parent.imageview.view.removeItem(
                self.flourescence_img.bg_im_dialog.parent.bg_im)
            self.flourescence_img.bg_im_dialog.bg_im = None
        # Remove the 'Select Window' button from the popup
        self.flourescence_img.bg_im_dialog.formlayout.removeRow(0)
        self.flourescence_img.bg_im_dialog.parent.bg_im = pg.ImageItem(self.intensity_img)
        self.flourescence_img.bg_im_dialog.parent.bg_im.setOpacity(
            self.flourescence_img.bg_im_dialog.alpha_slider.value())
        self.flourescence_img.bg_im_dialog.parent.imageview.view.addItem(
            self.flourescence_img.bg_im_dialog.parent.bg_im)

    def calculate_flourescence(self):
        print('Calculating Flourescence Intensity')

        progress = self.create_progress_bar('Please wait while fluorescence intensity is being calculated...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        if self.flourescence_img is None:
            g.alert('Make sure a Flourescence image is selected')
        elif self.intensity_img is None:
            g.alert('Make sure an Intensity image is selected')
        else:
            intensityprops = measure.regionprops(self.flourescence_img.labeled_img, self.intensity_img)
            self.flourescenceIntensities = np.array([p.mean_intensity for p in intensityprops])
            self.isIntensityCalculated = True

    def save_flourescence(self):
        print("Saving Flourescence Data")

        progress = self.create_progress_bar('Please wait while fluorescence intensity is being saved...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        if not self.isIntensityCalculated:
            g.alert("Make sure the Flourescence Intensity has been calculated")
        else:
            self.saved_flourescence_rois = self.flourescence_img.window_props
            self.saved_flourescence_states = np.copy(self.flourescence_img.window_states)

    def determine_positives(self):
        print("Determining Positive Fibers")
        self.flourescence_img.imageIdentifier = ClassifierWindow.FLR

    def measure_positives(self):
        print("Measuring Positive Fibers")
        if not self.isIntensityCalculated:
            g.alert("Make sure the Flourescence Intensity has been calculated")
        else:
            # Get the user-selected Positive Fiber's MFI values
            userselectedprops = []
            for i in range(len(self.flourescence_img.window_props)):
                if self.flourescence_img.temp_states is not None and self.flourescence_img.temp_states[i] == 3:
                    userselectedprops.append(g.quantimus.flourescenceIntensities[i])

            # Sort the MFI values
            userselectedprops.sort()

            if len(userselectedprops) > 0:
                # Get the lowest MFI from the list
                lowest_mfi_value = userselectedprops[0]

                # Loop through all ROIs and get any MFI that is higher than the lowest user selected
                self.positiveFiberRois = []
                self.positiveFiberStates = []
                for i in range(len(self.flourescence_img.window_props)):
                    # build the positive states list - for printing
                    self.positiveFiberStates.append(self.flourescence_img.temp_states[i])
                    if self.flourescence_img.temp_states[i] != 2 and g.quantimus.flourescenceIntensities[i] >= lowest_mfi_value:
                        self.positiveFiberRois.append(self.flourescence_img.window_props[i])
                        self.positiveFiberStates[i] = 3

                # Paint the image appropriately
                self.paint_positive_fibers(self.positiveFiberRois)
            else:
                g.alert("Please select at least one Positive Fiber")

    def clear_positives(self):
        print("Clearing Positive Fibers")
        self.flourescence_img.imageIdentifier = None
        self.flourescence_img.temp_states = None
        self.saved_positive_rois = None
        self.saved_positive_states = None
        self.flourescence_img.window_states = np.copy(self.roiStates)
        self.flourescence_img.set_roi_states()

    def save_positives(self):
        print("Saving Positive Fibers")
        self.saved_positive_rois = self.positiveFiberRois
        self.saved_positive_states = self.positiveFiberStates

    def paint_flr_colored_image(self):
        if self.flourescence_img is not None:
            self.flourescence_img.set_roi_states()

    def reset_flourescence_data(self):
        self.flourescenceIntensities = None
        self.isIntensityCalculated = False
        self.positiveFiberRois = None
        self.positiveFiberStates = None
        self.saved_flourescence_rois = None
        self.saved_flourescence_states = None
        self.saved_positive_rois = None
        self.saved_positive_states = None

    def paint_positive_fibers(self, props):
        if props is not None:
            for prop in props:
                x, y = prop.coords.T
                self.flourescence_img.colored_img[x, y] = ClassifierWindow.BLUE
            self.flourescence_img.update_image(self.flourescence_img.colored_img)

    def select_dapi_image(self):
        print('DAPI image selected.')
        # Reset potentially old data
        self.reset_dapi_data()
        # Select the image
        self.dapi_img = ClassifierWindow(self.dapi_img_selector.window.image, 'CNF Image')
        self.dapi_img.imageIdentifier = ClassifierWindow.DAPI
        self.dapi_img.window_states = np.copy(self.dapi_img_selector.window.window_states)
        self.algorithm_gui.run_erosion_button.pressed.connect(self.dapi_img.run_erosion)
        self.paint_dapi_colored_image()

    def select_dapi_binarized_image(self):
        print('DAPI image selected.')
        # Reset potentially old data
        self.reset_dapi_data()
        # Select the image
        self.dapi_binarized_img = self.binarized_dapi_img_selector.window.image
        self.dapi_rois = measure.regionprops(self.dapi_binarized_img)
        # Overlay the DAPI onto the image
        self.dapi_img.set_bg_im()

        self.dapi_img.bg_im_dialog.setWindowTitle("Select an image")

        if self.dapi_img.bg_im_dialog.parent.bg_im is not None:
            self.dapi_img.bg_im_dialog.parent.imageview.view.removeItem(self.dapi_img.bg_im_dialog.parent.bg_im)
            self.dapi_img.bg_im_dialog.bg_im = None
        self.dapi_img.bg_im_dialog.formlayout.removeRow(0)
        self.dapi_img.bg_im_dialog.parent.bg_im = pg.ImageItem(
            self.binarized_dapi_img_selector.window.imageview.imageItem.image)
        self.dapi_img.bg_im_dialog.parent.bg_im.setOpacity(self.dapi_img.bg_im_dialog.alpha_slider.value())
        self.dapi_img.bg_im_dialog.parent.imageview.view.addItem(self.dapi_img.bg_im_dialog.parent.bg_im)
        self.paint_dapi_colored_image()

    def calculate_dapi(self):
        print('Calculating DAPI')

        progress = self.create_progress_bar('Please wait while CNF is being calculated...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        if self.dapi_img is None:
            g.alert('Make sure a DAPI image is selected')
        elif self.dapi_binarized_img is None:
            g.alert('Make sure a classified, DAPI image is selected')
        elif self.eroded_roi_states is None:
            g.alert('Make sure to run the Fiber Erosion before calculating DAPI Overlap')
        else:
            # Turn each image into lists
            erodedlist = list(chain.from_iterable(zip(*self.eroded_labeled_img)))
            dapilist = list(chain.from_iterable(zip(*self.dapi_binarized_img)))

            overlappedcoords = []
            imagewidth = len(list(self.dapi_binarized_img))

            count = 0
            # loop to check if there is overlap between DAPI and the eroded rois
            while count < len(erodedlist):
                # add an item to the overlapped coordinates list
                if erodedlist[count] > 0 and dapilist[count] > 0:
                    overlapx = math.floor(count / imagewidth) - 1
                    overlapy = count % imagewidth
                    newlist = [overlapx, overlapy]
                    overlappedcoords.append(newlist)
                count += 1

            previouscentroid = 0

            for coord in overlappedcoords:
                roi_num = self.dapi_img.labeled_img[coord[1], coord[0]] - 1
                if self.dapi_img.window_states[roi_num] == 1:
                    prop = self.dapi_img.window_props[roi_num]
                    # Check that the last processed ROI's centroid is not the exact same as the current ROI's centroid
                    # This is a method of checking uniqueness that doesn't require the use of nested loops
                    centroid = prop.centroid
                    if centroid != previouscentroid:
                        previouscentroid = centroid
                        self.dapi_img.window_states[roi_num] = 3
            self.paint_dapi_colored_image()

    def save_dapi(self):
        print("Saving DAPI Data")

        progress = self.create_progress_bar('Please wait while CNF data is being saved...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        self.saved_dapi_rois = self.dapi_img.window_props
        self.saved_dapi_states = np.copy(self.dapi_img.window_states)

    def paint_dapi_colored_image(self):
        if self.dapi_img is not None:
            # Green, Red, and Purple
            self.dapi_img.set_roi_states()
            # Yellow eroded ROIS
            if self.eroded_roi_states is not None:
                for prop in self.eroded_roi_states:
                    x, y = prop.coords.T
                    self.dapi_img.colored_img[x, y] = ClassifierWindow.YELLOW
            self.dapi_img.update_image(self.dapi_img.colored_img)

    def reset_dapi_data(self):
        self.dapi_rois = None
        self.eroded_roi_states = None
        self.saved_dapi_rois = None
        self.saved_dapi_states = None
        if self.dapi_img is not None:
            for i in np.nonzero(self.dapi_img.window_states == 3)[0]:
                self.dapi_img.window_states[i] = 1

    def print_data(self):

        if self.classifier_window is not None:
            self.classifier_window.calculate_window_props()
            props = self.classifier_window.window_props
        elif self.trained_img is not None:
            self.trained_img.calculate_window_props()
            props = self.trained_img.window_props
        elif self.filtered_trained_img is not None:
            self.filtered_trained_img.calculate_window_props()
            props = self.filtered_trained_img.window_props
        elif self.intensity_img is not None:
            self.intensity_img.calculate_window_props()
            props = self.intensity_img.window_props
        else:
            self.dapi_img.calculate_window_props()
            props = self.dapi_img.window_props

        progress = self.create_progress_bar('Please wait while data is printed...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        scalefactor = self.algorithm_gui.microns_per_pixel_SpinBox.value()
        resizefactor = g.quantimus.algorithm_gui.resize_factor_SpinBox.value()
        minferetprops = self.calc_min_feret_diameters(props)

        # Set up the multi-dimensional array to store all of the data
        dataarray = [['ROI #'], ['Area'], ['Minferet'], ['CNF'], ['MFI'], ['Positive']]

        for i in range(len(props)):
            QtWidgets.QApplication.processEvents()

            # Green States
            if self.roiStates[i] == 1 or self.roiStates[i] == 3:
                # ROI Number
                dataarray[0].append(str(i))

                # Area
                area = props[i].area
                area /= (scalefactor ** 2 * resizefactor ** 2)
                dataarray[1].append(area)

                # MinFeret
                minferet = minferetprops[i] / (scalefactor * resizefactor)
                dataarray[2].append(minferet)

                # CNF - Purple States
                if self.saved_dapi_states is not None:
                    if self.saved_dapi_states[i] == 3:
                        dataarray[3].append("1")
                    else:
                        dataarray[3].append("0")

                # MFI
                if self.isIntensityCalculated:
                    subtractionvalue = g.quantimus.algorithm_gui.flourescence_subtraction_SpinBox.value()
                    measuredintensity = self.flourescenceIntensities[i]
                    intensity = 0

                    if measuredintensity > subtractionvalue:
                        intensity = measuredintensity - subtractionvalue

                    dataarray[4].append(intensity)

                # Positive Fibers
                if self.saved_positive_rois is not None:
                    if self.saved_positive_states[i] == 3:
                        dataarray[5].append("1")
                    else:
                        dataarray[5].append("0")

        filesaveasname = save_file_gui('Save file as...', filetypes='*.xlsx')
        workbook = xlsxwriter.Workbook(filesaveasname)
        worksheet = workbook.add_worksheet()
        worksheet.write_column('A1', dataarray[0])
        worksheet.write_column('B1', dataarray[1])
        worksheet.write_column('C1', dataarray[2])
        worksheet.write_column('D1', dataarray[3])
        worksheet.write_column('E1', dataarray[4])
        worksheet.write_column('F1', dataarray[5])

        worksheet.write('G1', 'Scale Factor (microns/pixel)')
        worksheet.write('G2', scalefactor)
        worksheet.write('H1', 'Resize Factor')
        worksheet.write('H2', resizefactor)

        workbook.close()

    def calc_min_feret_diameters(self, props):
        # Calculates all the minimum feret diameters for regions in props
        min_feret_diameters = []
        thetas = np.arange(0, np.pi / 2, .01)
        rs = [rotation_matrix(theta) for theta in thetas]
        for prop in props:
            # Update the progress bar so it shows movement
            QtWidgets.QApplication.processEvents()

            # Determine if all items in the array are True
            alltrue = True
            for row in prop.convex_image:
                if not all(row):
                    alltrue = False
                    break

            if alltrue:
                min_feret_diameters.append(len(prop.convex_image.shape))
            else:
                identity_convex_hull = prop.convex_image
                coordinates = np.vstack(measure.find_contours(identity_convex_hull, 0.5, fully_connected='high'))
                coordinates -= np.mean(coordinates, 0)
                diams = []
                for r in rs:
                    newcoords = np.dot(coordinates, r.T)
                    w, h = np.max(newcoords, 0) - np.min(newcoords, 0)
                    diams.extend([w, h])
                min_feret_diameters.append(np.min(diams))
        min_feret_diameters = np.array(min_feret_diameters)
        return min_feret_diameters

    def reset_data(self, originating_window):
        reset = False
        if originating_window == Quantimus.MARKERS:
            print("Markers")
            if self.isMarkersFirstSelection:
                self.reset_all_data()
                if self.markers_win is not None:
                    self.markers_win.close()
                    self.markers_win = None
                if self.filled_boundaries_win is not None:
                    self.filled_boundaries_win.close()
                    self.filled_boundaries_win = None
                if self.classifier_window is not None:
                    self.classifier_window.close()
                    self.classifier_window = None
                if self.binary_img is not None:
                    self.binary_img.close()
                    self.binary_img = None
                self.isBinaryFirstSelection = True
                reset = True
            elif not self.isMarkersFirstSelection:
                if self.reset_question() == QtWidgets.QMessageBox.Yes:
                    self.reset_all_data()
                    if self.markers_win is not None:
                        self.markers_win.close()
                        self.markers_win = None
                    if self.filled_boundaries_win is not None:
                        self.filled_boundaries_win.close()
                        self.filled_boundaries_win = None
                    if self.classifier_window is not None:
                        self.classifier_window.close()
                        self.classifier_window = None
                    if self.binary_img is not None:
                        self.binary_img.close()
                        self.binary_img = None
                    self.isBinaryFirstSelection = True
                    reset = True
        elif originating_window == Quantimus.BINARY:
            print("Binary")
            if self.isBinaryFirstSelection:
                self.reset_all_data()
                reset = True
            elif not self.isBinaryFirstSelection:
                if self.reset_question() == QtWidgets.QMessageBox.Yes:
                    self.reset_all_data()
                    if self.classifier_window is not None:
                        self.classifier_window.close()
                        self.classifier_window = None
                    reset = True
        else:
            self.reset_all_data()
            reset = True
        return reset

    def reset_all_data(self):
        if self.trained_img is not None:
            self.trained_img.close()
            self.trained_img = None
        if self.filtered_trained_img is not None:
            self.filtered_trained_img.close()
            self.filtered_trained_img = None
        if self.dapi_img is not None:
            self.dapi_img.close()
            self.dapi_img = None
        if self.flourescence_img is not None:
            self.flourescence_img.close()
            self.flourescence_img = None
        if self.intensity_img is not None:
            self.intensity_img = None
        if self.dapi_binarized_img is not None:
            self.dapi_binarized_img = None
        if self.eroded_labeled_img is not None:
            self.eroded_labeled_img = None

        # ROIs and States
        self.roiStates = None
        self.eroded_roi_states = None
        self.dapi_rois = None
        self.roiProps = None
        self.flourescenceIntensities = None
        self.positiveFiberRois = None
        self.positiveFiberStates = None
        # Printing Data
        self.saved_flourescence_rois = None
        self.saved_flourescence_states = None
        self.saved_dapi_rois = None
        self.saved_dapi_states = None
        self.saved_positive_rois = None
        self.saved_positive_states = None

        # Misc
        self.isIntensityCalculated = False

    def reset_question(self):
        return QtWidgets.QMessageBox.question(
            self.algorithm_gui,
            "Message",
            "This will clear all image data, do you want to continue?",
            buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            defaultButton=QtWidgets.QMessageBox.No)


quantimus = Quantimus()
g.quantimus = quantimus
