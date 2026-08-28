import sys
from collections import OrderedDict
from typing import Iterable

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QApplication, QBoxLayout
from dicomUtils.ui.pyDicomView import ImageShowWidget, ImageShow
from . import hue_compass_colormap
from ..config import GlobalConfig

from ..utils.resource_utils import get_resource_path
from .SupportDataViewerUI import Ui_SupportDataViewerDialog


def _rgba_to_css(rgba: Iterable) -> str:
    r, g, b, a = rgba
    return f"rgba({round(r * 255)}, {round(g * 255)}, {round(b * 255)}, {a})"


class ImageShowWithMasks(ImageShow):
    def __init__(self, data: np.ndarray, masks: dict, resolution: Iterable, **kwargs):
        ImageShow.__init__(self, **kwargs)
        self.data = data
        self.masks = masks
        self.loadNumpyArray(data)
        self.resolution = list(resolution)  # after loadNumpyArray, which resets it
        self.displayImage(int(len(self.imList)/2))
        self.enabled_masks = {
            mask_name: True for mask_name in masks.keys()
        }
        self.mask_colormap = hue_compass_colormap.generate_colormap('red', len(masks), 0, 'support_mask_colormap')
        self.maskImPlot = None
        self.redraw()

    def enable_mask(self, mask_name, enabled=True):
        self.enabled_masks[mask_name] = enabled
        self.redraw()

    def enable_all_masks(self, enabled=True):
        for mask_name in self.masks.keys():
            self.enabled_masks[mask_name] = enabled
        self.redraw()

    def disable_mask(self, mask_name):
        self.enable_mask(mask_name, False)

    def disable_all_masks(self, mask_name):
        self.enable_all_masks(False)

    def refreshCB(self):
        mask_layer = np.zeros_like(self.imList[int(self.curImage)], dtype=np.uint8)
        current_mask_value=0
        for mask_name, mask in self.masks.items():
            current_mask_value += 1
            if not self.enabled_masks[mask_name]:
                continue
            mask_layer += current_mask_value * mask[:,:,int(self.curImage)]

        if self.maskImPlot is None:
            self.maskImPlot = self.axes.imshow(mask_layer, cmap=self.mask_colormap,
                                               alpha=GlobalConfig['MASK_LAYER_ALPHA'],
                                               vmin=0, vmax=len(self.masks)+1, zorder=100)
        else:
            self.maskImPlot.set_data(mask_layer)

    def clear(self):
        super().clear()
        try:
            self.maskImPlot.remove()
        except:
            pass
        self.maskImPlot = None

    def reset_view(self):
        self.clear()
        resolution = self.resolution
        self.loadNumpyArray(self.data)
        self.resolution = resolution  # loadNumpyArray resets the resolution
        self.displayImage(int(len(self.imList) / 2))
        self.resetContrast()
        self.redraw()

    def transpose_all(self):
        self.data = self.data.transpose([1,2,0])
        self.resolution = [self.resolution[1], self.resolution[2], self.resolution[0]]
        for mask_name, mask in self.masks.items():
            self.masks[mask_name] = mask.transpose([1,2,0])
        self.reset_view()

    def mirror_x_all(self):
        self.data = np.flip(self.data, axis=1)
        for mask_name, mask in self.masks.items():
            self.masks[mask_name] = np.flip(mask, axis=1)
        self.reset_view()

    def mirror_y_all(self):
        self.data = np.flip(self.data, axis=0)
        for mask_name, mask in self.masks.items():
            self.masks[mask_name] = np.flip(mask, axis=0)
        self.reset_view()

    def rotate_90_all(self):
        self.data = np.flip(self.data.transpose([1,0,2]), axis=1)
        self.resolution = [self.resolution[1], self.resolution[0], self.resolution[2]]
        for mask_name, mask in self.masks.items():
            self.masks[mask_name] = np.flip(mask.transpose([1,0,2]), axis=1)
        self.reset_view()

class SupportDataViewerDialog(QDialog, Ui_SupportDataViewerDialog):

    mask_transfer_signal = pyqtSignal(np.ndarray, dict, list)

    def __init__(self, npz_file, parent=None):
        super(SupportDataViewerDialog, self).__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setupUi(self)
        self.setWindowTitle("Support Data")

        with get_resource_path('button_xyz.png') as button_path:
            self.transposeButton.setIcon(QIcon(button_path))
        with get_resource_path('button_mirror_x.png') as button_path:
            self.mirrorXButton.setIcon(QIcon(button_path))
        with get_resource_path('button_mirror_y.png') as button_path:
            self.mirrorYButton.setIcon(QIcon(button_path))
        with get_resource_path('button_rotate.png') as button_path:
            self.rotateButton.setIcon(QIcon(button_path))

        with open(npz_file, 'rb') as f:
            data = np.load(f)
            image = data['data']
            resolution = data['resolution']
            masks = OrderedDict({})
            for key in data:
                if key.startswith('mask_'):
                    masks[key[len('mask_'):]] = data[key]

        self.imshowWidget = ImageShowWidget(parent=self.imageViewerParent, viewer_class=ImageShowWithMasks, data=image, masks=masks, resolution=resolution, use_global_contrast_window=False)
        layout = QBoxLayout(QBoxLayout.LeftToRight, self.imageViewerParent)
        layout.addWidget(self.imshowWidget)

        self.transposeButton.clicked.connect(self.imshowWidget.transpose_all)
        self.mirrorXButton.clicked.connect(self.imshowWidget.mirror_x_all)
        self.mirrorYButton.clicked.connect(self.imshowWidget.mirror_y_all)
        self.rotateButton.clicked.connect(self.imshowWidget.rotate_90_all)

        self.transferButton.clicked.connect(self.emit_signal)

        roiGroup_layout = self.roiGroup.layout()
        colors = self.imshowWidget.mask_colormap.colors
        current_color = 1
        self.mask_checkboxes = []
        for mask_name in masks:
            mask_checkbox = QtWidgets.QCheckBox(self.roiGroup)
            mask_checkbox.setChecked(True)
            mask_checkbox.setText(mask_name)
            mask_checkbox.setStyleSheet(f"QCheckBox {{ background-color: {_rgba_to_css([*colors[current_color][:3], 0.2])}; }}")
            current_color += 1
            roiGroup_layout.insertWidget(len(roiGroup_layout)-2, mask_checkbox)
            def make_callback(mask_name, mask_checkbox):
                def clicked_callback():
                    all_checked = all(cb.isChecked() for cb in self.mask_checkboxes)
                    all_unchecked = all(not cb.isChecked() for cb in self.mask_checkboxes)
                    if all_checked:
                        self.allROITristateBox.setCheckState(Qt.CheckState.Checked)
                    elif all_unchecked:
                        self.allROITristateBox.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        self.allROITristateBox.setCheckState(Qt.CheckState.PartiallyChecked)
                    self.imshowWidget.enable_mask(mask_name, mask_checkbox.isChecked())
                return clicked_callback

            mask_checkbox.clicked.connect(make_callback(mask_name, mask_checkbox))
            self.mask_checkboxes.append(mask_checkbox)

        self.allROITristateBox.clicked.connect(self.tristate_clicked)
        self.allROITristateBox.setCheckState(Qt.CheckState.Checked)

    @pyqtSlot()
    def tristate_clicked(self):
        state = self.allROITristateBox.checkState()
        if state == Qt.CheckState.PartiallyChecked:
            self.allROITristateBox.setCheckState(Qt.CheckState.Checked)
            state = Qt.CheckState.Checked
        if state == Qt.CheckState.Unchecked:
            self.imshowWidget.enable_all_masks(False)
            for mask_checkbox in self.mask_checkboxes:
                mask_checkbox.setChecked(False)
        elif state == Qt.CheckState.Checked:
            self.imshowWidget.enable_all_masks(True)
            for mask_checkbox in self.mask_checkboxes:
                mask_checkbox.setChecked(True)


    def emit_signal(self):
        data = self.imshowWidget.data
        masks = {mask_name: self.imshowWidget.masks[mask_name] for mask_name in self.imshowWidget.masks.keys() if self.imshowWidget.enabled_masks[mask_name]}
        resolution = [self.imshowWidget.resolution[0], self.imshowWidget.resolution[1], self.imshowWidget.resolution[2]]
        self.mask_transfer_signal.emit(data, masks, resolution)


def main():
    app = QApplication(sys.argv)
    window = SupportDataViewerDialog('/mnt/data/dicom/chaos_npz/1.npz')
    window.show()
    app.exec_()