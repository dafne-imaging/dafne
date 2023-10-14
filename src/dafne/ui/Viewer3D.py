#  Copyright (c) 2023 Dafne-Imaging Team
# Part of this code are based on "wezel": https://github.com/QIB-Sheffield/wezel/

import os
import numpy as np
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QPushButton, QHBoxLayout, QLabel, QSlider
from ..config.config import GlobalConfig

os.environ["QT_API"] = "pyqt5"
import pyvista as pv
from pyvistaqt import QtInteractor

WIDTH = 400
HEIGHT = 400

OPACITY_ARRAY = np.array([0, 0.2, 0.3, 0.6, 0.2, 0])
COLORMAP = 'bone'

class Viewer3D(QWidget):

    hide_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.plotter = QtInteractor(self)
        self.plotter.background_color = 'black'
        self.plotter.add_camera_orientation_widget()
        self.spacing = (1.0, 1.0, 1.0)
        self.data = None
        self.anatomy = None
        self.actor_roi = None
        self.actor_anatomy = None
        self.anatomy_opacity = 0.2
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.plotter)

        fit_button = QPushButton("Fit to scene")
        layout.addWidget(fit_button)
        fit_button.clicked.connect(self.plotter.reset_camera)

        opacity_widget = QWidget()
        opacity_widget_layout = QHBoxLayout()
        opacity_widget.setLayout(opacity_widget_layout)
        opacity_widget_layout.addWidget(QLabel('Anatomy opacity:'))
        self.anat_opacity_slider = QSlider(Qt.Horizontal)
        self.anat_opacity_slider.setMinimum(0)
        self.anat_opacity_slider.setMaximum(100)
        self.anat_opacity_slider.setValue(20)
        self.anat_opacity_slider.valueChanged.connect(self.set_global_anat_opacity)
        opacity_widget_layout.addWidget(self.anat_opacity_slider)
        layout.addWidget(opacity_widget)


        self.setLayout(layout)
        self.setWindowTitle("3D Viewer")
        screen_width = QApplication.desktop().screenGeometry().width()
        self.setGeometry(screen_width - WIDTH, 0, WIDTH, HEIGHT)
        self.real_close_flag = False

    @pyqtSlot(list, np.ndarray)
    def set_spacing_and_anatomy(self, spacing, anatomy):
        self.spacing = spacing
        self.anatomy = anatomy.astype(np.uint16)
        self.visualize_anatomy()

    @pyqtSlot(int)
    def set_global_anat_opacity(self, value):
        self.anatomy_opacity = float(value)/100
        if self.actor_anatomy is None:
            return

        lut = pv.LookupTable(cmap=COLORMAP)
        lut.apply_opacity(OPACITY_ARRAY * self.anatomy_opacity)
        self.actor_anatomy.prop.apply_lookup_table(lut)
        self.plotter.render()

    def visualize_anatomy(self):
        self.plotter.remove_actor(self.actor_anatomy, render=False)
        if self.anatomy is None or self.spacing is None or self.anatomy_opacity == 0:
            self.plotter.render()
            return
        vol = pv.ImageData(dimensions=np.array(self.anatomy.shape)+1, spacing=self.spacing)
        vol.cell_data['values'] = self.anatomy.flatten(order='F')

        opacity = (OPACITY_ARRAY* self.anatomy_opacity)

        self.actor_anatomy = self.plotter.add_volume(vol,
                                                     scalars='values',
                                                     clim=[self.anatomy.min(), self.anatomy.max()],
                                                     opacity=opacity,
                                                     cmap=COLORMAP,
                                                     show_scalar_bar=False)
        self.plotter.show()

    @pyqtSlot(list, np.ndarray)
    def set_spacing_and_data(self, spacing, data):
        """
        Set the data and spacing.
        """
        self.spacing = spacing
        self.data = data
        self.update_data()

    @pyqtSlot(list)
    def set_spacing(self, spacing):
        """
        Set the affine transformation matrix.
        """
        self.spacing = spacing

    @pyqtSlot(np.ndarray)
    def set_affine(self, affine):
        """
        Set the affine transformation matrix.
        """
        column_spacing = np.linalg.norm(affine[:3, 0])
        row_spacing = np.linalg.norm(affine[:3, 1])
        slice_spacing = np.linalg.norm(affine[:3, 2])
        self.spacing = (column_spacing, row_spacing, slice_spacing)  # mm
        self.data = None

    def update_data(self):
        if not self.isVisible():
            return

        camera_position = self.plotter.camera_position
        #self.plotter.clear()
        self.plotter.remove_actor(self.actor_roi, reset_camera=False, render=False)
        if self.data is None or self.spacing is None or not np.any(self.data):
            print("No data to plot")
            self.plotter.render()
            return


        #grid = pv.UniformGrid(dimensions=self.data.shape, spacing=self.spacing)
        grid = pv.ImageData(dimensions=self.data.shape, spacing=self.spacing)
        surf = grid.contour([0.5], self.data.flatten(order="F"), method='marching_cubes')
        color = GlobalConfig['ROI_COLOR']
        color = [color[0], color[1], color[2]]
        self.actor_roi = self.plotter.add_mesh(surf,
                                               color=color,
                                               opacity=0.8,
                                               show_edges=False,
                                               smooth_shading=True,
                                               specular=0.5,
                                               show_scalar_bar=False,
                                               render=False
                                               )

        #restore camera position if it's not the default, which is too narrow
        if np.max(np.abs(camera_position[0])) > 1:
            self.plotter.camera_position = camera_position
        self.plotter.render()

    def real_close(self):
        self.real_close_flag = True
        self.close()

    def closeEvent(self, event):
        if self.real_close_flag: # if the window is closed by the user
            event.accept()
        self.hide_signal.emit()
        event.ignore()
        self.hide()

    @pyqtSlot(np.ndarray)
    def set_data(self, data):
        """
        Set the data to be plotted.
        """
        self.data = data
        self.update_data()

    @pyqtSlot(int, np.ndarray)
    def set_slice(self, slice_number, slice_data):
        """
        Set the slice to be plotted.
        """
        if self.data is None:
            return
        self.data[:, :, slice_number] = slice_data
        self.update_data()
