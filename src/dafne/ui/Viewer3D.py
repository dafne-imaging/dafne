#  Copyright (c) 2023 Dafne-Imaging Team
# Part of this code are based on "wezel": https://github.com/QIB-Sheffield/wezel/

import os
import numpy as np
from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QApplication, QPushButton, QHBoxLayout, QLabel, \
    QSlider, QCheckBox
from matplotlib.colors import ListedColormap

from dicomUtils.ui.pyDicomView import ImageShow, ImageShowWidget

from . import hue_compass_colormap
from ..config.config import GlobalConfig

# pyvista/VTK are only imported lazily (see _ensure_vtk_imported below), since VTK's OpenGL
# context creation can crash with low-level X11 errors on some systems (e.g. Wayland/XWayland
# sessions). The triplanar 2D views do not depend on them and work with GlobalConfig['DISABLE_3D_RENDER'].
pv = None
QtInteractor = None


def _ensure_vtk_imported():
    global pv, QtInteractor
    if pv is None:
        os.environ["QT_API"] = "pyqt5"
        import pyvista as _pv
        from pyvistaqt import QtInteractor as _QtInteractor
        pv = _pv
        QtInteractor = _QtInteractor

WIDTH = 900
HEIGHT = 800

OPACITY_ARRAY = np.array([0, 0.2, 0.3, 0.6, 0.2, 0])
COLORMAP = 'bone'

CROSSHAIR_COLOR = '#40c0ff'

# The main window displays slices along axis 2 of the volume
MAIN_PLANE_AXIS = 2


def _make_binary_colormap(color):
    return ListedColormap(np.array([
        [0, 0, 0, 0],
        [*color[:3], 1]]))


class _AxisSliceProxy:
    """ Lazily extracts 2D slices of a 3D volume along an arbitrary axis,
        exposing the list-like interface expected by ImageShow. """

    def __init__(self, volume, axis):
        self.volume = volume
        self.axis = axis

    def __getitem__(self, item):
        return np.take(self.volume, int(item), axis=self.axis).astype(np.float32)

    def __len__(self):
        return self.volume.shape[self.axis]


class TriplanarView(ImageShow):
    """ One plane of the triplanar viewer: shows the slices of the current volume along
        a fixed axis, overlays the active/other ROI masks with the same colormaps as the
        main window, and draws a crosshair marking the position of the other two planes.
        Clicking (or dragging) moves the crosshair; scrolling changes the displayed slice. """

    def __init__(self, controller=None, fixed_axis=MAIN_PLANE_AXIS, title='', **kwargs):
        ImageShow.__init__(self, use_global_contrast_window=True, **kwargs)
        self.controller = controller
        self.fixed_axis = fixed_axis
        # in-plane volume axes: rows of the displayed image, columns of the displayed image
        self.axis_y, self.axis_x = [axis for axis in (0, 1, 2) if axis != fixed_axis]
        self.instructions = title
        self.maskImPlot = None
        self.otherMaskImPlot = None
        self.crosshair_hline = None
        self.crosshair_vline = None

    def clear(self):
        super().clear()
        for artist in (self.maskImPlot, self.otherMaskImPlot, self.crosshair_hline, self.crosshair_vline):
            try:
                artist.remove()
            except:
                pass
        self.maskImPlot = None
        self.otherMaskImPlot = None
        self.crosshair_hline = None
        self.crosshair_vline = None

    def reload_volume(self):
        """ (Re)load the controller's anatomy volume and display the slice at the current position. """
        self.clear()
        volume = self.controller.anatomy
        if volume is None:
            try:
                self.fig.canvas.draw()
            except:
                pass
            return
        self.imList = _AxisSliceProxy(volume, self.fixed_axis)
        spacing = self.controller.spacing
        self.resolution = [spacing[self.axis_y], spacing[self.axis_x], spacing[self.fixed_axis]]
        self.contrastWindow = None  # recalculate the contrast for the new volume
        self.displayImage(int(self.controller.position[self.fixed_axis]), redraw=False)
        self.axes.set_xlim(-0.5, self.image.shape[1] - 0.5)
        self.axes.set_ylim(self.image.shape[0] - 0.5, -0.5)
        self.redraw()

    def show_slice(self, slice_number):
        if self.imList is None or len(self.imList) == 0:
            return
        self.displayImage(int(slice_number))

    def _mask_slice(self, mask_volume):
        if mask_volume is None or self.controller.anatomy is None:
            return None
        if mask_volume.shape != self.controller.anatomy.shape:
            return None
        return np.take(mask_volume, int(self.curImage), axis=self.fixed_axis)

    def _draw_masks(self):
        aspect = self.resolution[0] / self.resolution[1]
        alpha = GlobalConfig['MASK_LAYER_ALPHA']

        active_slice = self._mask_slice(self.controller.active_mask)
        if active_slice is not None:
            if self.maskImPlot is None:
                self.maskImPlot = self.axes.imshow(active_slice,
                                                   cmap=_make_binary_colormap(GlobalConfig['ROI_COLOR']),
                                                   alpha=alpha, vmin=0, vmax=1, zorder=100,
                                                   interpolation='none', aspect=aspect)
            else:
                self.maskImPlot.set_data(active_slice)
                self.maskImPlot.set_alpha(alpha)

        other_slice = self._mask_slice(self.controller.other_mask) \
            if self.controller.show_other_rois() else None
        if other_slice is not None and self.controller.other_cmap is not None:
            if self.otherMaskImPlot is None:
                self.otherMaskImPlot = self.axes.imshow(other_slice, zorder=101, interpolation='none',
                                                        aspect=aspect)
            self.otherMaskImPlot.set_data(other_slice)
            self.otherMaskImPlot.set_cmap(self.controller.other_cmap)
            self.otherMaskImPlot.set_clim(vmin=0, vmax=self.controller.other_cmap.N - 1)
            self.otherMaskImPlot.set_alpha(alpha)
        elif self.otherMaskImPlot is not None:
            self.otherMaskImPlot.set_data(np.zeros_like(self.image, dtype=np.uint8))

    def _draw_crosshair(self):
        x = self.controller.position[self.axis_x]
        y = self.controller.position[self.axis_y]
        if self.crosshair_vline is None:
            self.crosshair_vline = self.axes.axvline(x, color=CROSSHAIR_COLOR, linewidth=0.8,
                                                     alpha=0.7, zorder=200)
        else:
            self.crosshair_vline.set_xdata([x, x])
        if self.crosshair_hline is None:
            self.crosshair_hline = self.axes.axhline(y, color=CROSSHAIR_COLOR, linewidth=0.8,
                                                     alpha=0.7, zorder=200)
        else:
            self.crosshair_hline.set_ydata([y, y])

    def refreshCB(self):
        if self.imList is None or len(self.imList) == 0:
            return
        self._draw_masks()
        self._draw_crosshair()

    def _navigate(self, event):
        if event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            return
        self.controller.set_position({self.axis_x: int(round(event.xdata)),
                                      self.axis_y: int(round(event.ydata))},
                                     source_view=self)

    def leftPressCB(self, event):
        self._navigate(event)

    def leftMoveCB(self, event):
        self._navigate(event)

    def mouseScrollCB(self, event):
        if self.imList is None or len(self.imList) == 0:
            return
        super().mouseScrollCB(event)
        if self.curImage is not None:
            self.controller.set_position({self.fixed_axis: int(self.curImage)}, source_view=self)


class Viewer3D(QWidget):
    """ Triplanar + 3D viewer: three orthogonal planes of the currently displayed volume
        (with the ROI masks overlaid) plus a 3D rendering of the active ROI, the other
        ROIs and the anatomy. The plane corresponding to the main window follows (and
        drives) the slice displayed in the main window. """

    hide_signal = pyqtSignal()
    main_slice_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.spacing = (1.0, 1.0, 1.0)
        self.anatomy = None
        self.data = None  # active ROI mask volume
        self.other_mask = None  # labeled volume of the non-active ROIs (values from 2 up)
        self.other_cmap = None
        self.actor_roi = None
        self.actor_anatomy = None
        self.other_actors = []
        self.anatomy_opacity = 0.2
        self.position = [0, 0, 0]

        self.render_3d_enabled = not GlobalConfig['DISABLE_3D_RENDER']
        self.plotter = None
        if self.render_3d_enabled:
            _ensure_vtk_imported()
            self.plotter = QtInteractor(self)
            self.plotter.background_color = 'black'
            self.plotter.add_camera_orientation_widget()

        # Triplanar views: main plane on the top left, 3D view on the bottom right
        self.view_widgets = []
        self.views = []
        view_definitions = [(MAIN_PLANE_AXIS, 'Main plane', 0, 0),
                            (1, 'Orthogonal 1', 0, 1),
                            (0, 'Orthogonal 2', 1, 0)]

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        for fixed_axis, title, row, col in view_definitions:
            view_widget = ImageShowWidget(parent=self, viewer_class=TriplanarView,
                                          controller=self, fixed_axis=fixed_axis, title=title)
            self.view_widgets.append(view_widget)
            self.views.append(view_widget.viewer)
            layout.addWidget(view_widget, row, col)

        # 3D view cell with its controls
        self.anat_opacity_slider = None
        self.show_other_rois_checkbox = None
        if self.render_3d_enabled:
            plotter_widget = QWidget(self)
            plotter_layout = QVBoxLayout()
            plotter_layout.setContentsMargins(0, 0, 0, 0)
            plotter_layout.setSpacing(0)
            plotter_layout.addWidget(self.plotter)

            fit_button = QPushButton("Fit to scene")
            plotter_layout.addWidget(fit_button)
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

            self.show_other_rois_checkbox = QCheckBox('Show other ROIs')
            self.show_other_rois_checkbox.setChecked(True)
            self.show_other_rois_checkbox.stateChanged.connect(self.toggle_other_rois)
            opacity_widget_layout.addWidget(self.show_other_rois_checkbox)
            plotter_layout.addWidget(opacity_widget)

            plotter_widget.setLayout(plotter_layout)
            layout.addWidget(plotter_widget, 1, 1)

        for row in (0, 1):
            layout.setRowStretch(row, 1)
            layout.setColumnStretch(row, 1)

        self.setLayout(layout)
        self.setWindowTitle("Triplanar / 3D Viewer")
        screen_geometry = QApplication.desktop().screenGeometry()
        width = min(WIDTH, screen_geometry.width())
        height = min(HEIGHT, screen_geometry.height())
        self.setGeometry(screen_geometry.width() - width, 0, width, height)
        self.real_close_flag = False

    @property
    def active_mask(self):
        return self.data

    def show_other_rois(self):
        if self.show_other_rois_checkbox is None:
            return True
        return self.show_other_rois_checkbox.isChecked()

    @pyqtSlot(int)
    def toggle_other_rois(self, state):
        self.visualize_other_masks()
        self.refresh_triplanar()

    ### position handling

    def set_position(self, changes, source_view=None, notify_main=True):
        """ Set the crosshair position along one or more axes.
            changes: {axis: value}. Updates the affected views and, if the main plane
            position changed, notifies the main window (unless notify_main is False). """
        if self.anatomy is None:
            for axis, value in changes.items():
                self.position[axis] = int(value)
            return
        shape = self.anatomy.shape
        changed_axes = []
        for axis, value in changes.items():
            value = int(max(0, min(value, shape[axis] - 1)))
            if value != self.position[axis]:
                self.position[axis] = value
                changed_axes.append(axis)
        if not changed_axes:
            return
        for view in self.views:
            if view.fixed_axis in changed_axes:
                if view is not source_view:
                    view.show_slice(self.position[view.fixed_axis])
            elif view.axis_x in changed_axes or view.axis_y in changed_axes:
                view.redraw()
        if MAIN_PLANE_AXIS in changed_axes and notify_main:
            self.main_slice_changed.emit(self.position[MAIN_PLANE_AXIS])

    @pyqtSlot(int)
    def set_main_slice(self, slice_number):
        """ Follow a slice change of the main window (does not notify back). """
        self.set_position({MAIN_PLANE_AXIS: int(slice_number)}, notify_main=False)

    ### triplanar handling

    def reload_triplanar(self):
        if not self.isVisible():
            return
        for view in self.views:
            view.reload_volume()

    def refresh_triplanar(self):
        if not self.isVisible():
            return
        for view in self.views:
            view.redraw()

    ### anatomy handling

    @pyqtSlot(list, np.ndarray)
    def set_spacing_and_anatomy(self, spacing, anatomy):
        self.spacing = spacing
        shape_changed = self.anatomy is None or self.anatomy.shape != anatomy.shape
        self.anatomy = anatomy
        if shape_changed:
            # center the crosshair in-plane, but keep following the main window slice
            self.position = [anatomy.shape[0] // 2, anatomy.shape[1] // 2,
                             int(max(0, min(self.position[MAIN_PLANE_AXIS], anatomy.shape[2] - 1)))]
        self.reload_triplanar()
        self.visualize_anatomy()

    @pyqtSlot(int)
    def set_global_anat_opacity(self, value):
        self.anatomy_opacity = float(value)/100
        if not self.render_3d_enabled or self.actor_anatomy is None:
            return

        lut = pv.LookupTable(cmap=COLORMAP)
        lut.apply_opacity(OPACITY_ARRAY * self.anatomy_opacity)
        self.actor_anatomy.prop.apply_lookup_table(lut)
        self.plotter.render()

    def visualize_anatomy(self):
        if not self.render_3d_enabled or not self.isVisible():
            return
        self.plotter.remove_actor(self.actor_anatomy, render=False)
        self.actor_anatomy = None
        if self.anatomy is None or self.spacing is None or self.anatomy_opacity == 0:
            self.plotter.render()
            return
        anatomy = self.anatomy.astype(np.uint16)
        vol = pv.ImageData(dimensions=np.array(anatomy.shape)+1, spacing=self.spacing)
        vol.cell_data['values'] = anatomy.flatten(order='F')

        opacity = (OPACITY_ARRAY * self.anatomy_opacity)

        self.actor_anatomy = self.plotter.add_volume(vol,
                                                     scalars='values',
                                                     clim=[anatomy.min(), anatomy.max()],
                                                     opacity=opacity,
                                                     cmap=COLORMAP,
                                                     show_scalar_bar=False)
        self.plotter.render()

    ### active ROI mask handling

    @pyqtSlot(list, np.ndarray)
    def set_spacing_and_data(self, spacing, data):
        """
        Set the data and spacing.
        """
        self.spacing = spacing
        self.data = data
        self.update_data()
        self.refresh_triplanar()

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
        if not self.render_3d_enabled or not self.isVisible():
            return

        camera_position = self.plotter.camera_position
        self.plotter.remove_actor(self.actor_roi, reset_camera=False, render=False)
        if self.data is None or self.spacing is None or not np.any(self.data):
            print("No data to plot")
            self.plotter.render()
            return

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

    @pyqtSlot(np.ndarray)
    def set_data(self, data):
        """
        Set the data to be plotted.
        """
        self.data = data
        self.update_data()
        self.refresh_triplanar()

    @pyqtSlot(int, np.ndarray)
    def set_slice(self, slice_number, slice_data):
        """
        Set the slice to be plotted.
        """
        if self.data is None:
            return
        self.data[:, :, slice_number] = slice_data
        self.update_data()
        self.refresh_triplanar()

    ### other (non-active) ROI handling

    @pyqtSlot(list, np.ndarray)
    def set_other_masks(self, spacing, labeled_masks):
        """ Set the labeled volume of the non-active ROIs (0: background, 2..n: one label
            per ROI, matching the coloring of the main window). """
        self.spacing = spacing
        self.other_mask = labeled_masks
        n_labels = int(labeled_masks.max()) if labeled_masks is not None else 0
        if n_labels < 2:
            self.other_cmap = None
        elif GlobalConfig['USE_MULTIPLE_OTHER_COLORS']:
            self.other_cmap = hue_compass_colormap.generate_colormap(GlobalConfig['ROI_COLOR'],
                                                                     n_labels - 1)
        else:
            other_color = GlobalConfig['ROI_OTHER_COLOR']
            self.other_cmap = ListedColormap([[0, 0, 0, 0]] +
                                             [[other_color[0], other_color[1], other_color[2], 1.0]] * n_labels)
        self.visualize_other_masks()
        self.refresh_triplanar()

    def visualize_other_masks(self):
        if not self.render_3d_enabled or not self.isVisible():
            return
        for actor in self.other_actors:
            self.plotter.remove_actor(actor, render=False)
        self.other_actors = []
        if (not self.show_other_rois() or self.other_mask is None or self.other_cmap is None
                or self.spacing is None or not np.any(self.other_mask)):
            self.plotter.render()
            return
        camera_position = self.plotter.camera_position
        for label in np.unique(self.other_mask):
            if label == 0:
                continue
            binary = (self.other_mask == label).astype(np.uint8)
            grid = pv.ImageData(dimensions=binary.shape, spacing=self.spacing)
            surf = grid.contour([0.5], binary.flatten(order="F"), method='marching_cubes')
            if surf.n_points == 0:
                continue
            color = list(self.other_cmap.colors[min(int(label), self.other_cmap.N - 1)][:3])
            actor = self.plotter.add_mesh(surf,
                                          color=color,
                                          opacity=0.5,
                                          show_edges=False,
                                          smooth_shading=True,
                                          specular=0.5,
                                          show_scalar_bar=False,
                                          render=False
                                          )
            self.other_actors.append(actor)
        if np.max(np.abs(camera_position[0])) > 1:
            self.plotter.camera_position = camera_position
        self.plotter.render()

    ### window management

    def real_close(self):
        self.real_close_flag = True
        self.close()

    def closeEvent(self, event):
        if self.real_close_flag: # if the window is closed by the user
            event.accept()
        self.hide_signal.emit()
        event.ignore()
        self.hide()
