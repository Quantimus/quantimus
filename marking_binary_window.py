from flika import Window
from flika.utils.misc import save_file_gui, open_file_gui
from skimage.measure import label
from qtpy import QtWidgets, QtCore
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
        self.roi_states = np.zeros(np.max(self.labeled_img), dtype=np.uint8)
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)
        self.imageview.setImage(self.colored_img)
        self.menu.addAction(QtWidgets.QAction("&Save Classifications", self, triggered=self.save_classifications))
        self.menu.addAction(QtWidgets.QAction("&Load Classifications", self, triggered=self.load_classifications))

    def mouseClickEvent(self, ev):
        if ev.button() == 1:
            roi_num = self.labeled_img[int(self.x), int(self.y)] - 1
            if roi_num < 0:
                pass
            else:
                print('ROI #{}'.format(roi_num))
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    self.colored_img[self.labeled_img == roi_num + 1] = Classifier_Window.BLACK
                    #self.image[self.labeled_img == roi_num + 1] = 0
                    self.labeled_img[self.labeled_img == roi_num + 1] = 0
                    self.roi_states[roi_num] = -1
                    self.update_image(self.colored_img)
                else:
                    old_state = self.roi_states[roi_num]
                    new_state = (old_state + 1 ) % 3
                    self.roi_states[roi_num] = new_state
                    color = [Classifier_Window.WHITE, Classifier_Window.GREEN, Classifier_Window.RED][new_state]
                    self.colored_img[self.labeled_img == roi_num + 1] = color
                    self.update_image(self.colored_img)
        super().mouseClickEvent(ev)

    def update_image(self, image):
        viewrange = self.imageview.getView().viewRange()
        xrange, yrange = viewrange
        self.imageview.setImage(image)
        self.imageview.getView().setXRange(xrange[0], xrange[1], 0, False)
        self.imageview.getView().setYRange(yrange[0], yrange[1], 0)

    def save_classifications(self):
        filename = save_file_gui("Save classifications", filetypes='*.json')
        if filename is None:
            return None
        states = [np.asscalar(a)for a in self.roi_states]
        data = {'states': states}
        json.dump(data, codecs.open(filename, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)  ### this saves the array in .json format

    def load_classifications(self):
        filename = open_file_gui("Open classifications", filetypes='*.json')
        if filename is None:
            return None
        obj_text = codecs.open(filename, 'r', encoding='utf-8').read()
        data = json.loads(obj_text)
        self.roi_states = np.array(data['states'])
        self.colored_img = np.repeat(self.image[:, :, np.newaxis], 3, 2)
        for roi_num, new_state in enumerate(self.roi_states):
            color = [Classifier_Window.WHITE, Classifier_Window.GREEN, Classifier_Window.RED][new_state]
            self.colored_img[self.labeled_img == roi_num + 1] = color
        self.update_image(self.colored_img)


#self = Classifier_Window(g.currentWindow.image)





