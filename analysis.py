import scipy
import numpy as np
import skimage
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.morphology import disk
from skimage.filters import gabor_kernel
#from marking_binary_window import Classifier_Window


from scipy.signal import convolve2d
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters.rank import gradient
from skimage.measure import label

from flika.roi import makeROI
from flika import global_vars as g
from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox
from flika.process import difference_of_gaussians, threshold, zproject, remove_small_blobs
from flika.window import Window
from flika.process.file_ import open_file

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


if __name__ == '__main__':
    original = open_file(r'C:\Users\kyle\Dropbox\Software\2017 Jennas cell counting\mdx_224_Laminin.tif')
    split_channels()
    original = resize(2)
    Window(sobel(g.currentWindow.image))
    lap = Window(skimage.filters.laplace(g.currentWindow.image), 'laplacian')
    results = convolve_with_kernels_fft(lap.image, kernels)
    #Window(np.sort(results,0))
    lines = Window(np.max(results, 0), 'lines')
    lines_threshold = Window(remove_borders(g.currentWindow.image < .006), 'lines_threshold')
    markers = np.zeros_like(lines_threshold.image, dtype=np.uint8)
    markers[lines_threshold.image == 1] = 2
    markers[original.image > .6] = 1
    Window(markers)
    binary_image = watershed(markers, markers)
    binary_image -= 1
    binary_window = Classifier_Window(remove_borders(binary_image))
    features = get_important_features(binary_window.image)
    remove_false_positives(binary_window, features)
