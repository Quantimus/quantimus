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
import threading
import time

from .marking_binary_window import *

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox

def show_label_img(binary_img):
    A = label(binary_img, connectivity=2)
    I = np.zeros((np.max(A), A.shape[0], A.shape[1]), dtype=np.bool)
    for i in np.arange(1, np.max(A)):
        I[i-1] = A == i
    return Window(I)

def get_important_features(binary_image):
    features = {}
    label_img = label(binary_image, connectivity=2)
    props = measure.regionprops(label_img)
    features['convexity'] = np.array([p.filled_area / p.convex_area for p in props])
    features['eccentricity'] = np.array([p.eccentricity for p in props])
    features['area'] = np.array([p.filled_area for p in props]) / 4000
    features['circularity'] = np.array([p.filled_area*(4*np.pi)/p.perimeter**2 for p in props])
    return features
    p = pg.plot(features['convexity'], pen=pg.mkPen('r'))
    p.plot(features['eccentricity'], pen=pg.mkPen('g'))
    p.plot(features['area'], pen=pg.mkPen('y'))

def remove_borders(binary_image):
    label_img = label(binary_image, connectivity=2)
    mx, my = binary_image.shape
    border_labels = set(label_img[:, 0]) | set(label_img[0, :]) | set(label_img[mx-1, :]) | set(label_img[:, my-1])
    for i in border_labels:
        binary_image[label_img==i] = 0
    return binary_image

def remove_false_positives(binary_window, features):
    label_img = binary_window.labeled_img
    nElements = np.max(label_img)
    for i in np.arange(nElements):
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
    binary_window.colored_img[binary_window.image == 1] = Classifier_Window.GREEN
    for roi_num, new_state in enumerate(binary_window.roi_states):
        if new_state != 1:
            color = [Classifier_Window.WHITE, Classifier_Window.GREEN, Classifier_Window.RED][new_state]
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

def convolve_with_kernels_fft(I, kernels):
    results = []
    for k, kernel in enumerate(kernels):
        print(k)
        filtered = scipy.signal.fftconvolve(I, kernel, 'same')
        results.append(filtered)
    results = np.array(results)
    return results

def plot_regression_results(X1, X2, y):
    p = pg.plot()
    x1 = X1[y==1]
    x2 = X2[y==1]
    s1 = pg.ScatterPlotItem(x1, x2, size=10, pen=None, brush=pg.mkBrush(0, 255, 0, 255))
    p.addItem(s1)
    x1 = X1[y==0]
    x2 = X2[y==0]
    s1.addPoints(x1, x2, size=10, pen=None, brush=pg.mkBrush(255, 0, 0, 255))

def get_border_between_two_props(prop1, prop2):
    I2 = prop2.image
    I1 = np.zeros_like(I2)
    bbox = np.array(prop2.bbox)
    top_left = bbox[:2]
    a = prop1.coords - top_left
    I1[a[:, 0], a[:, 1]] = 1
    I1_expanded1 = binary_dilation(I1)
    I1_expanded2 = binary_dilation(binary_dilation(I1_expanded1))
    I1_expanded2[I1_expanded1] = 0
    border = I1_expanded2 * I2
    return np.argwhere(border) + top_left

def get_new_I(I, thresh1=.20, thresh2=.30):
    resizeFactor = g.myoquant.algorithm_gui.resize_factor_SpinBox.value()
    label_im_1 = label(I < thresh1, connectivity=2)
    label_im_2 = label(I < thresh2, connectivity=2)
    props_1 = measure.regionprops(label_im_1)
    props_2 = measure.regionprops(label_im_2)
    borders = np.zeros_like(I)

    #  The maximum of the labeled image is the number of contiguous regions, or ROIs.
    nROIs = np.max(label_im_1)
    for roi_num in np.arange(nROIs):
        QtWidgets.QApplication.processEvents()
        prop1 = props_1[roi_num]
        x, y = prop1.coords[0,:]
        prop2 = props_2[label_im_2[x, y] - 1]

        if prop1.area > 65 * resizeFactor :
            area_ratio = prop2.area/prop1.area
            if area_ratio > 1.2:
                border_idx = get_border_between_two_props(prop1, prop2)
                borders[border_idx[:,0], border_idx[:, 1]] = 1
    I_new = np.copy(I)
    I_new[np.where(borders)] = 2
    return I_new

class Myoquant():
    """myoquant()
    Muscle Cell Analysis Software
    """

    MARKERS = "MARKERS"
    BINARY = "BINARY"

    def __init__(self):
        pass

    def gui(self):
        #Windows
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

        #ROIs and States
        self.roiStates = None
        self.eroded_roi_states = None
        self.dapi_rois = None
        self.roiProps = None
        self.flourescenceIntensities = None

        #Printing Data
        self.saved_flourescence_rois = None
        self.saved_flourescence_states = None
        self.saved_dapi_rois = None
        self.saved_dapi_states = None

        # Misc
        self.isMarkersFirstSelection = True
        self.isBinaryFirstSelection = True
        self.isIntensityCalculated = False

        #GUI Setup
        gui = uic.loadUi(os.path.join(os.path.dirname(__file__), 'myoquant.ui'))
        self.algorithm_gui = gui
        gui.show()
        self.original_window_selector = WindowSelector()
        self.original_window_selector.valueChanged.connect(self.create_markers_win)
        gui.gridLayout_18.addWidget(self.original_window_selector)
        self.threshold1_slider = SliderLabel(3)
        self.threshold1_slider.setRange(0, 1)
        self.threshold1_slider.setValue(.22)
        self.threshold1_slider.valueChanged.connect(self.threshold_slider_changed)
        self.threshold2_slider = SliderLabel(2)
        self.threshold2_slider.setRange(0, 1)
        self.threshold2_slider.setValue(.1)
        self.threshold2_slider.valueChanged.connect(self.threshold_slider_changed)
        gui.gridLayout_threshold_one.addWidget(self.threshold1_slider)
        gui.gridLayout_threshold_two.addWidget(self.threshold2_slider)
        gui.fill_boundaries_button.pressed.connect(self.fill_boundaries_button)
        gui.SVM_button.pressed.connect(self.run_SVM_classification_on_image)
        gui.SVM_saved_button.pressed.connect(self.run_SVM_classification_on_saved_training_data)
        gui.load_classification_button.pressed.connect(self.load_classification_to_trained_image)
        gui.manual_filter_button.pressed.connect(self.filter_update)

        self.validation_manual_selector = WindowSelector()
        self.validation_manual_selector.valueChanged.connect(self.validate)
        gui.gridLayout_11.addWidget(self.validation_manual_selector)
        self.validation_automatic_selector = WindowSelector()
        self.validation_automatic_selector.valueChanged.connect(self.validate)
        gui.gridLayout_12.addWidget(self.validation_automatic_selector)

        self.binary_img_selector = WindowSelector()
        self.binary_img_selector.valueChanged.connect(self.select_binary_image)
        gui.gridLayout_import_binary_image.addWidget(self.binary_img_selector)

        self.intensity_img_selector= WindowSelector()
        self.intensity_img_selector.valueChanged.connect(self.select_intensity_image)
        gui.gridLayout_intensity_image.addWidget(self.intensity_img_selector)

        self.flourescence_img_selector= WindowSelector()
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

        gui.closeEvent = self.closeEvent

    def validate(self):
        print('validating...')
        if self.validation_manual_selector.window is None:
            return None
        if self.validation_automatic_selector.window is None:
            return None
        man = self.validation_manual_selector.window
        auto = self.validation_automatic_selector.window
        man_states = man.roi_states
        auto_states = auto.roi_states
        assert len(man_states) == len(auto_states)
        true_positives = np.count_nonzero(np.logical_and(man_states == 1, auto_states == 1))
        true_negatives = np.count_nonzero(np.logical_and(man_states == 2, auto_states == 2))
        false_positives = np.count_nonzero(np.logical_and(man_states == 2, auto_states == 1))
        false_negatives = np.count_nonzero(np.logical_and(man_states == 1, auto_states == 2))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        self.algorithm_gui.true_pos_label.setText(str(true_positives))
        self.algorithm_gui.true_neg_label.setText(str(true_negatives))
        self.algorithm_gui.false_pos_label.setText(str(false_positives))
        self.algorithm_gui.false_neg_label.setText(str(false_negatives))
        self.algorithm_gui.precision_label.setText(str(precision))
        self.algorithm_gui.recall_label.setText(str(recall))
        self.algorithm_gui.f1_score_label.setText(str(f1_score))

    def create_markers_win(self):
        if self.original_window_selector.window is None:
            g.alert('You must select a Window before creating the markers window.')
        else:
            if self.resetData(Myoquant.MARKERS):
                win = self.original_window_selector.window
                needAlert = False
                if np.max(win.image) > 1:
                    needAlert = True
                    I = win.image.astype(np.float)
                    I -= np.min(I)
                    I /= np.max(I)
                    win.image = I
                    win.dtype = I.dtype
                    win.imageview.setImage(win.image)
                    win._init_dimensions(win.image)
                    win.imageview.ui.graphicsView.addItem(win.top_left_label)
                original = win.image
                self.markers_win = Window(np.zeros_like(original, dtype=np.uint8), 'Binary Markers')
                self.markers_win.imageview.setLevels(0, 2)
                self.markers_win.imageview.ui.histogram.gradient.addTick(0, QtGui.QColor(0,0,255), True)
                self.markers_win.imageview.ui.histogram.gradient.setTickValue(1, .50)
                self.threshold1_slider.setRange(np.min(original), np.max(original))
                self.threshold2_slider.setRange(np.min(original), np.max(original))
                self.threshold_slider_changed()
                if needAlert:
                    g.alert("The window you select must have values between 0 and 1. Scaling the window now.")
                self.isMarkersFirstSelection = False

    def threshold_slider_changed(self):
        if self.original_window_selector.window is None:
            g.alert('You must select a Window before adjusting the levels.')
        else:
            thresh1 = self.threshold1_slider.value()
            thresh2 = self.threshold2_slider.value()
            I = self.original_window_selector.window.image
            markers = (I > thresh1).astype(dtype=np.uint8)
            markers[I > thresh2] = 2
            self.markers_win.imageview.setImage(markers, autoRange=False, autoLevels=False)

    def fill_boundaries_button(self):
        #Reset any data currently saved in the system
        lower_bound = self.threshold1_slider.value()
        upper_bound = self.threshold2_slider.value()
        #Original linspace = 8
        thresholds = np.linspace(lower_bound, upper_bound, 8)
        I = self.original_window_selector.window.image
        I_new = I

        progress = self.createProgressBar('Please wait while image is processed...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        for i in np.arange(len(thresholds) - 1):
            QtWidgets.QApplication.processEvents()
            print(thresholds[i])
            I_new = get_new_I(I_new, thresholds[i], thresholds[i + 1])
        self.filled_boundaries_win = Window(I_new, 'Filled Boundaries')
        classifier_image = remove_borders(I_new < upper_bound)
        self.binary_img = Classifier_Window(classifier_image, 'Binary Window')

    def get_norm_coeffs(self, X):
        mean = np.mean(X, 0)
        std = np.std(X, 0)
        return mean, std

    def normalize_data(self, X, mean, std):
        X = X - mean
        X = X / (2 * std)
        return X

    def closeEvent(self, event):
        print('Closing myoquant gui')
        if self.classifier_window is not None:
            self.classifier_window.close()
        event.accept() # let the window close

    def createProgressBar(self, msg):
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
        #Reset any data currently saved in the system
        if self.resetData(Myoquant.BINARY):
            print('Binary image selected.')
            self.classifier_window = Classifier_Window(self.binary_img_selector.window.image, 'Training Image')
            self.classifier_window.imageIdentifier = Classifier_Window.TRAINING
            self.roiStates = np.zeros(np.max(self.classifier_window.labeled_img), dtype=np.uint8)
            self.classifier_window.window_states = np.copy(self.roiStates)
            self.isBinaryFirstSelection = False

    def run_SVM_classification_on_image(self):
        if self.classifier_window is None:
            g.alert("Please select a Binary Image")
        else:
            #Start threading and Progress Bar
            progress = g.myoquant.createProgressBar('Please wait while fibers are being classified...')
            progress.show()
            QtWidgets.QApplication.processEvents()

            X_train, y_train = self.classifier_window.get_training_data()
            mu, sigma = self.get_norm_coeffs(self.classifier_window.features_array)
            self.run_SVM_classification_general(X_train, y_train, mu, sigma)

    def run_SVM_classification_on_saved_training_data(self):
        if self.classifier_window is None:
            g.alert("Please select a Binary Image")
        else:
            filename = open_file_gui("Open training_data", filetypes='*.json')
            if filename is None:
                return None
            obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
            data = json.loads(obj_text)

            # Start threading and Progress Bar
            progress = self.createProgressBar('Please wait while fibers are being classified...')
            progress.show()
            QtWidgets.QApplication.processEvents()

            X_train = np.array(data['features'])
            y_train = np.array(data['states'])
            mu, sigma = self.get_norm_coeffs(X_train)
            self.run_SVM_classification_general(X_train, y_train, mu, sigma)

    def run_SVM_classification_general(self, X_train, y_train, mu, sigma):
        print('Running SVM classification')
        try:
            X_train = self.normalize_data(X_train, mu, sigma)
            clf = svm.SVC()
            clf.fit(X_train, y_train)
            X_test = self.normalize_data(self.classifier_window.get_features_array(), mu, sigma)
            y = clf.predict(X_test)
            self.roiStates = np.zeros_like(y)
            self.roiStates[y == 1] = 1
            self.roiStates[y == 0] = 2
            self.trained_img = Classifier_Window(self.classifier_window.image, 'Trained Image')
            self.trained_img.imageIdentifier = Classifier_Window.TRAINING
            self.trained_img.window_states = np.copy(self.roiStates)
            ######################################################################################
            ##############   Add hand-designed rules here if you want  ###########################
            ######################################################################################
            # For instance, you could remove all ROIs smaller than 15 pixels like this:
            #X = self.classifier_window.features_array
            #roi_states[X[:, 0] < 15] = 2 # Area must be smaller than 15 pixels
            #roi_states[X[:, 3] < 0.6] = 2 # Convexity must be smaller than 0.6

            self.trained_img.set_roi_states()
            self.roiStates = np.copy(self.trained_img.window_states)
        except ValueError:
            g.alert('Please train a minimum of 1 positive and 1 negative sample')

    def load_classification_to_trained_image(self):
        print('Loading Classification to Trained Image')

        progress = self.createProgressBar('Please wait while fibers are being classified...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        self.trained_img = Classifier_Window(self.classifier_window.image, 'Trained Image')
        self.trained_img.imageIdentifier = Classifier_Window.TRAINING
        self.trained_img.window_states = np.copy(self.roiStates)
        self.trained_img.load_classifications_act()
        self.trained_img.set_roi_states()

    def filter_update(self):
        print('Manually filtering...')

        progress = self.createProgressBar('Please wait while image is filtered...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            min_circularity = g.myoquant.algorithm_gui.min_circularity_SpinBox.value()
            max_circularity = g.myoquant.algorithm_gui.max_circularity_SpinBox.value()
            circularityCheckbox = g.myoquant.algorithm_gui.circularity_CheckBox
            min_area = g.myoquant.algorithm_gui.min_area_SpinBox.value()
            max_area = g.myoquant.algorithm_gui.max_area_SpinBox.value()
            areaCheckbox = g.myoquant.algorithm_gui.area_CheckBox
            min_convexity = g.myoquant.algorithm_gui.min_convexity_SpinBox.value()
            max_convexity = g.myoquant.algorithm_gui.max_convexity_SpinBox.value()
            convexityCheckbox = g.myoquant.algorithm_gui.convexity_CheckBox
            min_eccentricity = g.myoquant.algorithm_gui.min_eccentricity_SpinBox.value()
            max_eccentricity = g.myoquant.algorithm_gui.max_eccentricity_SpinBox.value()
            eccentricityCheckbox = g.myoquant.algorithm_gui.eccentricity_CheckBox

            features = self.trained_img.get_features_array()
            states = np.copy(self.trained_img.window_states)
            count = 0


            for feature in features:
                #Update the progress bar so it shows movement
                QtWidgets.QApplication.processEvents()
                if self.trained_img.window_states[count] == 1:
                    #Area
                    if areaCheckbox.isChecked():
                        if feature[0] >= min_area and feature[0] <= max_area:
                            states[count] = 1
                        else:
                            states[count] = 2
                    #Eccentricity
                    if eccentricityCheckbox.isChecked():
                        if states[count] == 1 and feature[1] >= min_eccentricity and feature[1] <= max_eccentricity:
                            states[count] = 1
                        else:
                            states[count] = 2
                    #Convexity
                    if convexityCheckbox.isChecked():
                        if states[count] == 1 and feature[2] >= min_convexity and feature[2] <= max_convexity:
                            states[count] = 1
                        else:
                            states[count] = 2
                    #Circularity
                    if circularityCheckbox.isChecked():
                        if states[count] == 1 and feature[3] >= min_circularity and feature[3] <= max_circularity:
                            states[count] = 1
                        else:
                            states[count] = 2
                else:
                    states[count] = 2
                count = count + 1

            self.filtered_trained_img = Classifier_Window(self.trained_img.image, 'Filtered Trained Image')
            self.filtered_trained_img.imageIdentifier = Classifier_Window.TRAINING
            self.filtered_trained_img.window_states = states
            self.filtered_trained_img.set_roi_states()
            self.roiStates = self.filtered_trained_img.window_states
        except AttributeError:
            g.alert('Please run the SVM Classification Training')

    def select_flourescence_image(self):
        print('Flourescence image selected.')
        # Reset potentially old data
        self.resetFlourescenceData()
        self.flourescence_img = None
        # Select the image
        self.flourescence_img = Classifier_Window(self.flourescence_img_selector.window.image, 'Flourescence Image')
        self.flourescence_img.imageIdentifier = Classifier_Window.FLR
        self.flourescence_img.window_states = np.copy(self.flourescence_img_selector.window.window_states)
        self.paintFlrColoredImage()

    def select_intensity_image(self):
        print('Intensity image selected.')
        # Reset potentially old data
        self.resetFlourescenceData()
        # Select the image
        self.intensity_img = self.intensity_img_selector.window.image
        self.flourescence_img.set_bg_im()
        self.flourescence_img.bg_im_dialog.setWindowTitle("Select an image")
        if self.flourescence_img.bg_im_dialog.parent.bg_im is not None:
            self.flourescence_img.bg_im_dialog.parent.imageview.view.removeItem(self.flourescence_img.bg_im_dialog.parent.bg_im)
            self.flourescence_img.bg_im_dialog.bg_im = None
        #Remove the 'Select Window' button from the popup
        self.flourescence_img.bg_im_dialog.formlayout.removeRow(0)
        self.flourescence_img.bg_im_dialog.parent.bg_im = pg.ImageItem(self.intensity_img)
        self.flourescence_img.bg_im_dialog.parent.bg_im.setOpacity(self.flourescence_img.bg_im_dialog.alpha_slider.value())
        self.flourescence_img.bg_im_dialog.parent.imageview.view.addItem(self.flourescence_img.bg_im_dialog.parent.bg_im)

    def calculate_flourescence(self):
        print('Calculating Flourescence Intensity')

        progress = self.createProgressBar('Please wait while fluorescence intensity is being calculated...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        if self.flourescence_img is None:
            g.alert('Make sure a Flourescence image is selected')
        elif self.intensity_img is None:
            g.alert('Make sure an Intensity image is selected')
        else:
            intensityProps = measure.regionprops(self.flourescence_img.labeled_img, self.intensity_img)
            rois = np.max(self.flourescence_img.labeled_img)
            roi_num = np.arange(rois)
            self.flourescenceIntensities = np.array([p.mean_intensity for p in intensityProps])
            self.isIntensityCalculated = True

    def save_flourescence(self):
        print("Saving Flourescence Data")

        progress = self.createProgressBar('Please wait while fluorescence intensity is being saved...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        if self.isIntensityCalculated == False:
            g.alert("Make sure the Flourescence Intensity has been calculated")
        else:
            self.saved_flourescence_rois = self.flourescence_img.window_props
            self.saved_flourescence_states = np.copy(self.flourescence_img.window_states)

    def paintFlrColoredImage(self):
        if self.flourescence_img is not None:
            self.flourescence_img.set_roi_states()

    def resetFlourescenceData(self):
        self.flourescenceIntensities = None
        self.isIntensityCalculated = False
        self.saved_flourescence_rois = None
        self.saved_flourescence_states = None

    def select_dapi_image(self):
        print('DAPI image selected.')
        #Reset potentially old data
        self.resetDAPIData
        #Select the image
        self.dapi_img = Classifier_Window(self.dapi_img_selector.window.image, 'CNF Image')
        self.dapi_img.imageIdentifier = Classifier_Window.DAPI
        self.dapi_img.window_states = np.copy(self.dapi_img_selector.window.window_states)
        self.algorithm_gui.run_erosion_button.pressed.connect(self.dapi_img.run_erosion)
        self.paintDapiColoredImage()

    def select_dapi_binarized_image(self):
        print('DAPI image selected.')
        # Reset potentially old data
        self.resetDAPIData()
        #Select the image
        self.dapi_binarized_img = self.binarized_dapi_img_selector.window.image
        self.dapi_rois = measure.regionprops(self.dapi_binarized_img)
        #Overlay the DAPI onto the image
        self.dapi_img.set_bg_im()

        self.dapi_img.bg_im_dialog.setWindowTitle("Select an image")

        if self.dapi_img.bg_im_dialog.parent.bg_im is not None:
            self.dapi_img.bg_im_dialog.parent.imageview.view.removeItem(self.dapi_img.bg_im_dialog.parent.bg_im)
            self.dapi_img.bg_im_dialog.bg_im = None
        self.dapi_img.bg_im_dialog.formlayout.removeRow(0)
        self.dapi_img.bg_im_dialog.parent.bg_im = pg.ImageItem(self.binarized_dapi_img_selector.window.imageview.imageItem.image)
        self.dapi_img.bg_im_dialog.parent.bg_im.setOpacity(self.dapi_img.bg_im_dialog.alpha_slider.value())
        self.dapi_img.bg_im_dialog.parent.imageview.view.addItem(self.dapi_img.bg_im_dialog.parent.bg_im)
        self.paintDapiColoredImage()

    def calculate_dapi(self):
        print('Calculating DAPI')

        progress = self.createProgressBar('Please wait while CNF is being calculated...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        if self.dapi_img is None:
            g.alert('Make sure a DAPI image is selected')
        elif self.dapi_binarized_img is None:
            g.alert('Make sure a classified, DAPI image is selected')
        elif self.eroded_roi_states is None:
            g.alert('Make sure to run the Fiber Erosion before calculating DAPI Overlap')
        else:
            #Turn each image into lists
            erodedList = list(chain.from_iterable(zip(*self.eroded_labeled_img)))
            dapiList = list(chain.from_iterable(zip(*self.dapi_binarized_img)))

            overlappedCoords = []
            imageWidth = len(list(self.dapi_binarized_img))

            count = 0
            # loop to check if there is overlap between DAPI and the eroded rois
            while count < len(erodedList):
                #add an item to the overlapped coordinates list
                if(erodedList[count] > 0 and dapiList[count] > 0):
                    overlapX = math.floor(count / imageWidth) - 1
                    overlapY = count % imageWidth
                    newList = [overlapX, overlapY]
                    overlappedCoords.append(newList)
                count = count + 1

            previousCentroid = 0

            for coord in overlappedCoords:
                roi_num = self.dapi_img.labeled_img[coord[1], coord[0]] - 1
                if self.dapi_img.window_states[roi_num] == 1:
                    prop = self.dapi_img.window_props[roi_num]
                    #Check that the last processed ROI's centroid is not the exact same as the current ROI's centroid
                    #This is a method of checking uniqueness that doesn't require the use of nested loops
                    centroid = prop.centroid
                    if centroid != previousCentroid:
                        previousCentroid = centroid
                        self.dapi_img.window_states[roi_num] = 3
            self.paintDapiColoredImage()

    def save_dapi(self):
        print("Saving DAPI Data")

        progress = self.createProgressBar('Please wait while CNF data is being saved...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        self.saved_dapi_rois = self.dapi_img.window_props
        self.saved_dapi_states = np.copy(self.dapi_img.window_states)

    def paintDapiColoredImage(self):
        if self.dapi_img is not None:
            #Green, Red, and Purple
            self.dapi_img.set_roi_states()
            #Yellow eroded ROIS
            if self.eroded_roi_states is not None:
                for prop in self.eroded_roi_states:
                    x, y = prop.coords.T
                    self.dapi_img.colored_img[x, y] = Classifier_Window.YELLOW
            self.dapi_img.update_image(self.dapi_img.colored_img)

    def resetDAPIData(self):
        self.dapi_rois = None
        self.eroded_roi_states = None
        self.saved_dapi_rois = None
        self.saved_dapi_states = None
        if self.dapi_img is not None:
            for i in np.nonzero(self.dapi_img.window_states == 3)[0]:
                self.dapi_img.window_states[i] = 1

    def print_data(self):

        props = None
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

        progress = self.createProgressBar('Please wait while data is printed...')
        progress.show()
        QtWidgets.QApplication.processEvents()

        scaleFactor = self.algorithm_gui.microns_per_pixel_SpinBox.value()
        resizeFactor = g.myoquant.algorithm_gui.resize_factor_SpinBox.value()
        minferetProps = self.calc_min_feret_diameters(props)

        # Set up the multi-dimensional array to store all of the data
        dataArray = [['ROI #'], ['Area'], ['Minferet'], ['MFI'], ['CNF']]

        count = 0
        for prop in props:
            QtWidgets.QApplication.processEvents()

            #Green States
            if self.roiStates[count] == 1 or self.roiStates[count] == 3:
                # ROI Number
                dataArray[0].append(count)

                #Area
                area = prop.area
                area /= (scaleFactor**2 * resizeFactor**2)
                dataArray[1].append(area)

                # MinFeret
                minferet = minferetProps[count] / (scaleFactor * resizeFactor)
                dataArray[2].append(minferet)

                #MFI
                if self.isIntensityCalculated:
                    subtractionValue = g.myoquant.algorithm_gui.flourescence_subtraction_SpinBox.value()
                    measuredIntensity = self.flourescenceIntensities[count]
                    intensity = 0

                    if measuredIntensity > subtractionValue:
                        intensity = measuredIntensity - subtractionValue

                    dataArray[3].append(intensity)

                #CNF - Purple States
                if self.saved_dapi_states is not None:
                    if self.saved_dapi_states[count] == 3:
                        dataArray[4].append("1")
                    else:
                        dataArray[4].append("0")

            count += 1

        fileSaveAsName = save_file_gui('Save file as...', filetypes='*.xlsx')
        workbook = xlsxwriter.Workbook(fileSaveAsName)
        worksheet = workbook.add_worksheet()
        worksheet.write_column('A1',dataArray[0])
        worksheet.write_column('B1',dataArray[1])
        worksheet.write_column('C1',dataArray[2])
        worksheet.write_column('D1',dataArray[3])
        worksheet.write_column('E1',dataArray[4])

        worksheet.write('F1', 'Scale Factor (microns/pixel)')
        worksheet.write('F2', scaleFactor)
        worksheet.write('G1', 'Resize Factor')
        worksheet.write('G2', resizeFactor)

        #worksheet.write_column('E1', dataArray[4])
        workbook.close()

    def calc_min_feret_diameters(self, props):
        '''  calculates all the minimum feret diameters for regions in props '''
        min_feret_diameters = []
        thetas = np.arange(0, np.pi / 2, .01)
        Rs = [rotation_matrix(theta) for theta in thetas]
        for roi in props:

            #Update the progress bar so it shows movement
            QtWidgets.QApplication.processEvents()

            #Determine if all items in the array are True
            allTrue = True
            for row in roi.convex_image:
                if not all(row):
                    allTrue = False
                    break

            if allTrue:
                min_feret_diameters.append(len(roi.convex_image.shape))
            else:
                identity_convex_hull = roi.convex_image
                coordinates = np.vstack(find_contours(identity_convex_hull, 0.5, fully_connected='high'))
                coordinates -= np.mean(coordinates, 0)
                diams = []
                for R in Rs:
                    newcoords = np.dot(coordinates, R.T)
                    w, h = np.max(newcoords, 0) - np.min(newcoords, 0)
                    diams.extend([w, h])
                min_feret_diameters.append(np.min(diams))
        min_feret_diameters = np.array(min_feret_diameters)
        return min_feret_diameters

    def resetData(self, originatingWindow):
        reset = False
        if originatingWindow == Myoquant.MARKERS:
            print("Markers")
            if self.isMarkersFirstSelection:
                self.resetAllData()
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
                if self.resetQuestion() == QtWidgets.QMessageBox.Yes:
                    self.resetAllData()
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
        elif originatingWindow == Myoquant.BINARY:
            print("Binary")
            if self.isBinaryFirstSelection:
                self.resetAllData()
                reset = True
            elif not self.isBinaryFirstSelection:
                if self.resetQuestion() == QtWidgets.QMessageBox.Yes:
                    self.resetAllData()
                    if self.classifier_window is not None:
                        self.classifier_window.close()
                        self.classifier_window = None
                    reset = True
        else:
            self.resetAllData()
            reset = True
        return reset

    def resetAllData(self):
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
        # Printing Data
        self.saved_flourescence_rois = None
        self.saved_flourescence_states = None
        self.saved_dapi_rois = None
        self.saved_dapi_states = None
        # Misc
        self.isIntensityCalculated = False

    def resetQuestion(self):
        return QtWidgets.QMessageBox.question(
            self.algorithm_gui,
            "Message",
            "This will clear all image data, do you want to continue?",
            buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            defaultButton=QtWidgets.QMessageBox.No)

myoquant = Myoquant()
g.myoquant = myoquant


# def testing():
#     from plugins.myoquant.marking_binary_window import Classifier_Window
#     original = open_file(r'C:\Users\kyle\Desktop\tmp.tif')
#     binary_tmp = open_file(r'C:\Users\kyle\Desktop\binary.tif')
#     binary = Classifier_Window(binary_tmp.image, 'Classifier Window')
#     close(binary_tmp)
#     binary.load_classifications(r'C:\Users\kyle\Desktop\classifications.json')
#
#
# if __name__ == '__main__':
#     original = open_file(r'C:\Users\kyle\Dropbox\Software\2017 Jennas cell counting\mdx_224_Laminin.tif')
#     split_channels()
#     crop
#     original = resize(2)
#     g.myoquant.gui()

