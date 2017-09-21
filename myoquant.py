import os
import scipy
import numpy as np
from distutils.version import StrictVersion
import skimage
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d
from skimage import measure
from skimage.measure import label
from sklearn.linear_model import LogisticRegression
from qtpy import uic

import flika
from flika.roi import makeROI
from flika import global_vars as g
from flika.process import difference_of_gaussians, threshold, zproject, remove_small_blobs
from flika.window import Window
from flika.process.file_ import open_file

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox




from .marking_binary_window import Classifier_Window

def show_label_img(binary_img):
    A = label(binary_img, connectivity=2)
    I = np.zeros((np.max(A), A.shape[0], A.shape[1]), dtype=np.bool)
    for i in np.arange(1, np.max(A)):
        I[i-1] = A == i
    return Window(I)

def get_important_features(binary_image):
    features = {}
    # important features include:
    # convexity: ratio of convex_image area to image area
    # area: number of pixels total
    # eccentricity: 0 is a circle, 1 is a line
    label_img = label(binary_image, connectivity=2)
    props = measure.regionprops(label_img)
    features['convexity'] = np.array([p.filled_area / p.convex_area for p in props])
    features['eccentricity'] = np.array([p.eccentricity for p in props])
    features['area'] = np.array([p.filled_area for p in props]) / 4000
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






class Myoquant():
    """myoquant()
    Muscle Cell Analysis Software
    """

    def __init__(self):
        pass
    def gui(self):
        self.classifier_window = None
        self.lines_win = None
        gui = uic.loadUi(os.path.join(os.path.dirname(__file__), 'myoquant.ui'))
        self.algorithm_gui = gui
        gui.show()
        self.original_window_selector = WindowSelector()
        gui.gridLayout_11.addWidget(self.original_window_selector)
        gui.apply_filter_button.pressed.connect(self.apply_filter)
        self.threshold1_slider = SliderLabel(3)
        self.threshold1_slider.setRange(0, .1)
        self.threshold1_slider.setValue(.02)
        self.threshold1_slider.valueChanged.connect(self.threshold_slider_changed)
        self.threshold2_slider = SliderLabel(2)
        self.threshold2_slider.setRange(0, 1)
        self.threshold2_slider.setValue(.6)
        self.threshold2_slider.valueChanged.connect(self.threshold_slider_changed)
        gui.gridLayout_9.addWidget(self.threshold1_slider)
        gui.gridLayout_16.addWidget(self.threshold2_slider)
        gui.run_watershed_button.pressed.connect(self.run_watershed)
        gui.logistic_regression_button.pressed.connect(self.run_logistic_regression)
        gui.save_fiber_button.pressed.connect(self.save_fiber_data)

    def apply_filter(self):
        if self.original_window_selector.window is None:
            g.alert('You must select a Window before applying the spatial filter.')
        else:
            sobel_im = sobel(self.original_window_selector.window.image)
            results = convolve_with_kernels_fft(sobel_im, kernels)
            self.lines_win = Window(np.max(results, 0), 'Spatially filtered')
            self.markers_win = Window(np.zeros_like(sobel_im, dtype=np.uint8), 'Binary Markers')
            self.threshold1_slider.setRange(np.min(self.lines_win.image), np.max(self.lines_win.image))
            self.threshold2_slider.setRange(np.min(self.original_window_selector.window.image), np.max(self.original_window_selector.window.image))
            self.algorithm_gui.analyze_tab_widget.setCurrentIndex(1)
            self.threshold_slider_changed()

    def threshold_slider_changed(self):
        thresh1 = self.threshold1_slider.value()
        thresh2 = self.threshold2_slider.value()
        print("thresh1 = {} thresh2 = {}".format(thresh1, thresh2))
        markers = 1 + (self.lines_win.image < thresh1).astype(dtype=np.uint8)
        markers[self.original_window_selector.window.image > thresh2] = 0
        #self.markers_win.image = markers
        self.markers_win.imageview.setImage(markers, autoLevels=False)
        self.markers_win.imageview.setLevels(-.1, 2.1)

    def run_watershed(self):
        if self.classifier_window is not None:
            self.algorithm_gui.gridLayout_17.removeWidget(self.classifier_window)
        M = self.markers_win.imageview.image
        markers = np.zeros_like(M, dtype=np.uint8)
        markers[M == 0] = 1
        markers[M == 2] = 2
        binary_image = watershed(markers, markers)
        binary_image -= 1
        self.classifier_window = Classifier_Window(remove_borders(binary_image))
        self.algorithm_gui.gridLayout_17.addWidget(self.classifier_window)
        self.algorithm_gui.analyze_tab_widget.setCurrentIndex(2)

    def run_logistic_regression(self):
        X, y = self.classifier_window.get_training_data()
        self.logreg = LogisticRegression(C=1e9)
        self.logreg.fit(X, y)
        #print('Accuracy = {}'.format(logreg.score(X,y)))
        X = self.classifier_window.features_array
        y = self.logreg.predict(X)
        result_win = Classifier_Window(self.classifier_window.image)
        roi_states = np.zeros_like(y)
        roi_states[y == 1] = 1
        roi_states[y == 0] = 2


        ######################################################################################
        ##############   Add hand-designed rules here if you want  ###########################
        ######################################################################################
        # For instance, you could remove all ROIs smaller than 20 pixels like this:
        roi_states[X[:, 0] < 20] = 2




        result_win.set_roi_states(roi_states)
        params = list(self.logreg.intercept_) + list(self.logreg.coef_[0])
        params = ', '.join(['Beta_' + str(i) + '=' + str(coef) for i, coef in enumerate(params) ])
        self.algorithm_gui.model_params_label.setText(params)

    def save_fiber_data(self):
        g.alert('save_fiber_data() Not yet implemented')

myoquant = Myoquant()
g.myoquant = myoquant




"""
from plugins.myoquant.marking_binary_window import Classifier_Window
open_file(r'C:/Users/kyle/Desktop/tmp2.tif')
C = Classifier_Window(g.win.image)
C.load_classifications()




self = g.myoquant
self.classifier_window.save_classifications()


self.classifier_window = Classifier_Window(self.classifier_window.image)
self.classifier_window.load_classifications()
X, y = self.classifier_window.get_training_data()

logreg = LogisticRegression(C=1e9)
logreg.fit(X, y, sample_weight=X[:,0])
print('Accuracy = {}'.format(logreg.score(X,y)))


y = logreg.predict(X)
plot_regression_results(X[:,0], X[:,1], y)


"""



if __name__ == '__main__':
    original = open_file(r'C:\Users\kyle\Dropbox\Software\2017 Jennas cell counting\mdx_224_Laminin.tif')
    split_channels()
    crop
    original = resize(2)
    sobelwin = Window(sobel(original.image), 'Sobel')
    #laplacian = Window(skimage.filters.laplace(sobel.image), 'Laplacian')
    #results = convolve_with_kernels_fft(laplacian.image, kernels)
    results = convolve_with_kernels_fft(sobelwin.image, kernels)
    #Window(np.sort(results,0))
    lines = Window(np.max(results, 0), 'lines')
    lines_threshold = Window(remove_borders(lines.image < .02), 'lines_threshold')
    markers = np.zeros_like(lines_threshold.image, dtype=np.uint8)
    markers[lines_threshold.image == 1] = 2
    markers[original.image > .6] = 1
    Window(markers)
    binary_image = watershed(markers, markers)
    binary_image -= 1
    binary_window = Classifier_Window(remove_borders(binary_image))
    features = get_important_features(binary_window.image)
    remove_false_positives(binary_window, features)
