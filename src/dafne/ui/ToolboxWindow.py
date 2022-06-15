#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2021 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools

import os
import sys

from ..ui.ToolboxUI import Ui_SegmentationToolbox
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QMovie, QIcon, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QInputDialog, QFileDialog, QApplication, QDialog, \
                            QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem
from PyQt5.QtSvg import QSvgWidget
from . import GenericInputDialog
from .. import config
import platform
from . import BatchCalcTransforms
import webbrowser
from . import images as image_resources

from .LogWindow import LogWindow

from ..utils.resource_utils import get_resource_path

DOCUMENTATION_URL = 'https://www.dafne.network/documentation/'


SPLASH_ANIMATION_FILE = 'dafne_anim.gif'
ABOUT_SVG_FILE = 'about_paths.svg'

UPLOAD_DATA_TXT_1 = \
"""<h2>!!! This will upload your data to our servers !!!</h2>
<p>This is not necessary from your side, but it will help us improve our models and create new ones.</p>
<p><b>THE DATA WILL BE ANONYMOUS. NO PATIENT INFORMATION WILL BE SHARED.</b></p>
<p>Only the pixel values, the resolution, and the segmented masks will be sent.</p>
<p>However, make sure that you are complying with your local regulations before proceeding!</p>
<p>By proceeding, you are <b>relinquishing your rights to the uploaded data</b> and you are releasing them into the public domain.</p>
"""

UPLOAD_DATA_TXT_2 = \
"""<h2>Let us ask you one more time:</h2><h2>ARE YOU SURE?</h2>
By clicking "Yes" you are declaring that you are allowed to share the data according to your local regulations,
and that you are <b>relinquishing any right on this data</b>.
This is <b>NOT NECESSARY</b> for the function of the program.
"""

SHORTCUT_HELP = [
    ('Previous Image', '[Left Arrow], [Up Arrow]'),
    ('Next Image', '[Right Arrow], [Down Arrow]'),
    ('Paint/Add/Move', '[Shift]'),
    ('Erase/Delete', '[Ctrl/Cmd]'),
    ('Propagate forward', 'n'),
    ('Propagate back', 'b'),
    ('Reduce Brush Size', '-, y, z, [Ctrl/Cmd]+[Scroll down]'),
    ('Increase Brush Size', '+, x, [Ctrl/Cmd]+[Scroll up]'),
    ('Remove ROI overlap', 'r'),
    ('Undo', '[Ctrl/Cmd]+z'),
    ('Redo', '[Ctrl/Cmd]+y')
]

class AboutDialog(QDialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        myLayout = QVBoxLayout(self)
        self.setLayout(myLayout)
        self.setWindowTitle(f"About Dafne - version {config.VERSION}")
        self.setWindowModality(Qt.ApplicationModal)
        with get_resource_path(ABOUT_SVG_FILE) as svg_file:
            svg = QSvgWidget(svg_file)
        myLayout.addWidget(svg)
        btn = QPushButton("OK")
        btn.clicked.connect(self.close)
        myLayout.addWidget(btn)
        self.resize(640,480)
        self.show()

class ShortcutDialog(QDialog):
    def __init__(self, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        myLayout = QVBoxLayout(self)
        self.setLayout(myLayout)
        self.setWindowTitle("Keyboard shortcuts")
        self.setWindowModality(Qt.ApplicationModal)

        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setRowCount(len(SHORTCUT_HELP))
        item_h = QTableWidgetItem()
        item_h.setText("Keys")
        self.tableWidget.setHorizontalHeaderItem(0, item_h)
        for index, shortcut in enumerate(SHORTCUT_HELP):
            item_l = QTableWidgetItem()
            item_l.setText(shortcut[0])
            item_r = QTableWidgetItem()
            item_r.setText(shortcut[1])
            self.tableWidget.setVerticalHeaderItem(index, item_l)
            self.tableWidget.setItem(index, 0, item_r)

        myLayout.addWidget(self.tableWidget)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.horizontalHeader().setStretchLastSection(True)

        btn = QPushButton("OK")
        btn.clicked.connect(self.close)
        myLayout.addWidget(btn)
        self.resize(400, 400)
        self.show()


def ask_confirm(text):
    def decorator_confirm(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj.confirm(text):
                func(obj, *args, **kwargs)

        return wrapper

    return decorator_confirm


class ToolboxWindow(QMainWindow, Ui_SegmentationToolbox):

    reblit = pyqtSignal()
    do_autosegment = pyqtSignal(int, int)
    contour_optimize = pyqtSignal()
    contour_simplify = pyqtSignal()
    contour_propagate_fw = pyqtSignal()
    contour_propagate_bw = pyqtSignal()
    calculate_transforms = pyqtSignal()

    roi_added = pyqtSignal(str)
    roi_deleted = pyqtSignal(str)
    subroi_added = pyqtSignal(int)
    subroi_deleted = pyqtSignal(int)
    roi_changed = pyqtSignal(str, int)
    roi_clear = pyqtSignal()
    classification_changed = pyqtSignal(str)
    classification_change_all = pyqtSignal(str)

    editmode_changed = pyqtSignal(str)

    undo = pyqtSignal()
    redo = pyqtSignal()

    roi_import = pyqtSignal(str)
    roi_export = pyqtSignal(str)

    # signal to make a copy/rename of ROI. Parameters are old roi, new name, make copy y/n
    roi_copy = pyqtSignal(str, str, bool)
    # signal to combine two ROIs. Parameters are roi1, roi2, operator, dest_roi
    roi_combine = pyqtSignal(str, str, str, str)
    roi_multi_combine = pyqtSignal(list, str, str)
    roi_remove_overlap = pyqtSignal()

    masks_export = pyqtSignal(str, str)
    mask_import = pyqtSignal(str)

    data_open = pyqtSignal(str, str)

    statistics_calc = pyqtSignal(str)
    radiomics_calc = pyqtSignal(str, bool, int, int)
    incremental_learn = pyqtSignal()

    mask_grow = pyqtSignal()
    mask_shrink = pyqtSignal()

    brush_changed = pyqtSignal()

    config_changed = pyqtSignal()

    data_upload = pyqtSignal(str)

    model_import = pyqtSignal(str, str)

    NO_STATE = 0
    ADD_STATE = 1
    REMOVE_STATE = 2
    ROTATE_STATE = 3
    TRANSLATE_STATE = 4

    BRUSH_CIRCLE = 'Circle'
    BRUSH_SQUARE = 'Square'

    EDITMODE_MASK = 'Mask'
    EDITMODE_CONTOUR = 'Contour'


    def __init__(self, muscle_segmentation_window, activate_registration=True, activate_radiomics=True):
        super(ToolboxWindow, self).__init__()
        self.setupUi(self)

        self.state_buttons_dict = {
            self.addpaint_button: self.ADD_STATE,
            self.removeerase_button: self.REMOVE_STATE,
            self.translateContour_button: self.TRANSLATE_STATE,
            self.rotateContour_button: self.ROTATE_STATE
        }

        self.muscle_segmentation_window = muscle_segmentation_window

        self.setWindowFlag(Qt.WindowCloseButtonHint, False)

        if platform.system() == 'Darwin':
            self.menubar.setNativeMenuBar(False) # native menu bar behaves weirdly in Mac OS
            self.setWindowFlag(Qt.CustomizeWindowHint, True) # disable window decorations

        # reload the brush icons so that it works in with pyinstaller too. Check under windows!
        icon = QIcon()
        with get_resource_path('circle.png') as f:
            icon.addPixmap(QPixmap(f), QIcon.Normal, QIcon.Off)
        self.circlebrush_button.setIcon(icon)
        icon1 = QIcon()
        with get_resource_path('square.png') as f:
            icon1.addPixmap(QPixmap(f), QIcon.Normal, QIcon.Off)
        self.squarebrush_button.setIcon(icon1)


        
        self.setWindowTitle("Segmentation Toolbox")
        self.splashWidget.setVisible(False)
        self.all_rois = {}
        self.current_roi = ""
        self.current_subroi = 0
        self.suppress_roi_change_emit = False
        self.valid_roi_selected = False
        self.roi_combo.currentTextChanged.connect(self.send_roi_changed)
        self.roi_combo.currentTextChanged.connect(self.repopulate_subrois)
        self.subroi_combo.currentTextChanged.connect(self.send_roi_changed)
        self.roi_add_button.clicked.connect(self.add_roi)
        self.subroi_add_button.clicked.connect(self.subroi_added.emit)
        self.roi_remove_button.clicked.connect(self.delete_roi)
        self.subroi_remove_button.clicked.connect(self.delete_subroi)

        self.addpaint_button.clicked.connect(self.manage_edit_toggle)
        self.removeerase_button.clicked.connect(self.manage_edit_toggle)
        self.translateContour_button.clicked.connect(self.manage_edit_toggle)
        self.rotateContour_button.clicked.connect(self.manage_edit_toggle)

        self.eraseFromAllROIs_checkbox.setVisible(False)

        self.edit_state = self.NO_STATE
        self.temp_edit_state = None

        self.removeall_button.clicked.connect(self.clear_roi)

        self.classification_combo.currentTextChanged.connect(self.on_classification_changed)
        self.classification_all_button.clicked.connect(self.on_classification_change_all)
        self.autosegment_button.clicked.connect(self.on_do_segmentation)

        self.undoButton.clicked.connect(self.undo.emit)
        self.redoButton.clicked.connect(self.redo.emit)

        self.optimizeButton.clicked.connect(self.contour_optimize.emit)
        self.simplifyButton.clicked.connect(self.contour_simplify.emit)

        self.calcTransformsButton.clicked.connect(self.do_registration)
        self.propagateForwardButton.clicked.connect(self.contour_propagate_fw.emit)
        self.propagateBackButton.clicked.connect(self.contour_propagate_bw.emit)

        if not activate_registration:
            self.registrationGroup.setVisible(False)

        self.editmode_combo.currentTextChanged.connect(lambda : self.set_edit_mode(self.editmode_combo.currentText()))
        self.editmode_combo.setCurrentText('Mask')
        self.set_edit_mode('Mask')

        self.brushsize_slider.valueChanged.connect(self.brushsliderCB)
        self.brushsize_slider.setValue(5)
        self.brushsliderCB(5)

        self.grow_button.clicked.connect(self.mask_grow.emit)
        self.shrink_button.clicked.connect(self.mask_shrink.emit)

        with get_resource_path(SPLASH_ANIMATION_FILE) as f:
            self.splash_movie = QMovie(f)
        self.splash_label.setMovie(self.splash_movie)

        ## Menus

        self.actionImport_ROIs.triggered.connect(self.importROI_clicked)
        self.actionExport_ROIs.triggered.connect(self.exportROI_clicked)

        self.actionLoad_data.triggered.connect(self.loadData_clicked)

        self.actionSave_as_Dicom.triggered.connect(lambda: self.export_masks_dir('dicom'))
        self.actionSave_as_Compact_Dicom.triggered.connect(lambda: self.export_masks_dir('compact_dicom'))
        self.actionSaveNPY.triggered.connect(lambda: self.export_masks_dir('npy'))
        self.actionSave_as_Nifti.triggered.connect(lambda: self.export_masks_dir('nifti'))
        self.actionSaveNPZ.triggered.connect(self.export_masks_npz)
        self.actionSave_as_Compact_Nifti.triggered.connect(self.export_masks_compact_nifti)

        self.actionAbout.triggered.connect(self.about)
        self.actionOpen_online_documentation.triggered.connect(lambda : webbrowser.open(DOCUMENTATION_URL))
        self.actionHelp_shortcuts.triggered.connect(self.show_shortcuts)

        self.actionCalculate_statistics.triggered.connect(self.calculate_statistics)
        if not activate_radiomics:
            self.actionPyRadiomics.setVisible(False)

        self.actionPyRadiomics.triggered.connect(self.calculate_radiomics)

        self.actionIncremental_Learn.triggered.connect(self.do_incremental_learn)

        self.actionImport_masks.triggered.connect(self.load_mask_clicked)
        self.actionImport_multiple_masks.triggered.connect(self.load_multi_mask_clicked)

        self.actionPreferences.triggered.connect(self.edit_preferences)
        self.action_Restore_factory_settings.triggered.connect(self.clear_preferences)

        self.actionCopy_roi.triggered.connect(self.do_copy_roi)
        self.actionCombine_roi.triggered.connect(self.do_combine_roi)
        self.actionMultiple_combine.triggered.connect(self.do_combine_multiple_roi)
        self.actionRemove_overlap.triggered.connect(self.roi_remove_overlap.emit)

        self.actionCopy_roi.setEnabled(False)
        self.actionCombine_roi.setEnabled(False)
        self.actionMultiple_combine.setEnabled(False)
        self.actionRemove_overlap.setEnabled(False)

        self.action_Upload_data.triggered.connect(self.do_upload_data)

        self.actionImport_model.triggered.connect(self.do_import_model)

        self.actionOpen_transform_calculator.triggered.connect(self.open_transform_calculator)

        self.actionShowLogs.triggered.connect(self.show_logs)

        if not config.GlobalConfig['REDIRECT_OUTPUT']:
            self.actionShowLogs.setEnabled(False)
            self.actionShowLogs.setVisible(False)

        self.reload_config()
        self.config_changed.connect(self.reload_config)

        self.opacitySlider.setValue(config.GlobalConfig['MASK_LAYER_ALPHA'] * 100)
        self.opacitySlider.valueChanged.connect(self.set_opacity_config)

        self.general_enable(False)

        self.resize(self.mainUIWidget.sizeHint())

    @pyqtSlot()
    def show_logs(self):
        log_window = LogWindow(self)
        log_window.show()


    @pyqtSlot(int)
    def set_opacity_config(self, value):
        config.GlobalConfig['MASK_LAYER_ALPHA'] = float(value) / 100
        self.reblit.emit()

    @pyqtSlot()
    def general_enable(self, enabled = True):
        self.mainUIWidget.setEnabled(enabled)
        self.actionImport_ROIs.setEnabled(enabled)
        self.actionExport_ROIs.setEnabled(enabled)
        self.menuImport.setEnabled(enabled)
        self.menuSave_masks.setEnabled(enabled)
        self.action_Upload_data.setEnabled(enabled)
        self.actionCalculate_statistics.setEnabled(enabled)
        self.actionPyRadiomics.setEnabled(enabled)
        self.actionIncremental_Learn.setEnabled(enabled)

    @pyqtSlot()
    def reload_config(self):
        # all config-dependent UI elements go here
        self.actionSave_as_Nifti.setVisible(config.GlobalConfig['ENABLE_NIFTI'])
        self.actionSave_as_Compact_Nifti.setVisible(config.GlobalConfig['ENABLE_NIFTI'])
        self.actionImport_model.setVisible(config.GlobalConfig['MODEL_PROVIDER'] == 'Local')
        if config.GlobalConfig['ENABLE_DATA_UPLOAD'] and (config.GlobalConfig['MODEL_PROVIDER'] == 'Remote' or
                                                          config.GlobalConfig['FORCE_LOCAL_DATA_UPLOAD']):
            self.action_Upload_data.setVisible(True)
        else:
            self.action_Upload_data.setVisible(False)

    @pyqtSlot()
    def open_transform_calculator(self):
        #process = multiprocessing.Process(target=BatchCalcTransforms.run)
        #process.start()
        self.batchProcessWidget = BatchCalcTransforms.CalcTransformWindow()
        self.batchProcessWidget.show()

    @pyqtSlot()
    def do_import_model(self):
        modelFile, _ = QFileDialog.getOpenFileName(self, caption='Select model to import',
                                                   filter='Model files (*.model);;All files ()')
        if modelFile:
            accept, values = GenericInputDialog.show_dialog('Model Import', [
                GenericInputDialog.TextLineInput('Model name')
            ], self)
            if accept:
                modelName = values[0]
                self.model_import.emit(modelFile, modelName)

    @pyqtSlot()
    @ask_confirm(UPLOAD_DATA_TXT_1)
    @ask_confirm(UPLOAD_DATA_TXT_2)
    def do_upload_data(self):
        accept, values = GenericInputDialog.show_dialog('Add a comment', [
            GenericInputDialog.TextLineInput('Comment/description of the dataset')
        ], self)
        if accept:
            self.data_upload.emit(values[0])

    @pyqtSlot()
    def edit_preferences(self):
        if config.show_config_dialog(self, config.GlobalConfig['ADVANCED_CONFIG']):
            config.save_config()
            self.config_changed.emit()

    @pyqtSlot()
    def clear_preferences(self):
        accept, values = GenericInputDialog.show_dialog("Restore Factory Settings", [
            GenericInputDialog.BooleanInput("Keep the API key", True)
        ], parent=self, message="Warning! This will restore the default preferences!")
        if not accept: return
        if values[0]: # keep the API key
            api_key = config.GlobalConfig['API_KEY']
        config.delete_config()
        if values[0]:
            config.GlobalConfig['API_KEY'] = api_key
            config.save_config()
        self.config_changed.emit()


    def _make_roi_list_option_for_dialog(self, label):
        return GenericInputDialog.OptionInput(label,
                                              [self.roi_combo.itemText(i) for i in range(self.roi_combo.count())],
                                              self.roi_combo.currentText())

    @pyqtSlot()
    def do_copy_roi(self):
        if not self.roi_combo.currentText(): return
        accept, values = GenericInputDialog.show_dialog('Copy/Rename ROI', [self._make_roi_list_option_for_dialog('Original ROI'),
                                                                            GenericInputDialog.BooleanInput('Make copy', True),
                                                                            GenericInputDialog.TextLineInput('New name')], self)
        if accept:
            self.roi_copy.emit(values[0], values[2], values[1])

    @pyqtSlot()
    def do_combine_multiple_roi(self):
        if not self.roi_combo.currentText(): return
        roi_checkbox_list = []
        for i in range(self.roi_combo.count()):
            roi_name = self.roi_combo.itemText(i)
            roi_checkbox_list.append(GenericInputDialog.BooleanInput(roi_name))

        accept, rois = GenericInputDialog.show_dialog('ROIs', roi_checkbox_list, self)
        if not accept: return

        roi_list = [roi_name for roi_name, checked in rois.items() if checked]
        #print("Selected ROIs", roi_list)
        if len(roi_list) < 2:
            self.alert('Select at least 2 ROIs')
            return

        operator_option = GenericInputDialog.OptionInput('Operation',
                                                         ['Union', 'Subtraction', 'Intersection', 'Exclusion'])

        accept, values = GenericInputDialog.show_dialog('Combine ROIs', [operator_option,
                                                                         GenericInputDialog.TextLineInput('Output ROI')],
                                                        self, message='<b>Combining:</b> ' + '; '.join(roi_list))

        if not accept: return

        output_name = values['Output ROI']
        self.roi_multi_combine.emit(roi_list, values[0], output_name)

    @pyqtSlot()
    def do_combine_roi(self):
        if not self.roi_combo.currentText(): return
        input_roi_option_1 = self._make_roi_list_option_for_dialog('First ROI')
        input_roi_option_2 = self._make_roi_list_option_for_dialog('Second ROI')
        output_roi_option = self._make_roi_list_option_for_dialog('Output ROI')
        operator_option = GenericInputDialog.OptionInput('Operation',
                                                         ['Union', 'Subtraction', 'Intersection', 'Exclusion'])
        output_roi_option.add_option('Specify a different name')
        accept, values = GenericInputDialog.show_dialog('Combine ROIs', [input_roi_option_1,
                                                                         input_roi_option_2,
                                                                         operator_option,
                                                                         output_roi_option,
                                                                         GenericInputDialog.TextLineInput('New name')],
                                                        self)
        if accept:
            if values['Output ROI'] == 'Specify a different name':
                if values['New name'] == '':
                    return
                else:
                    output_name = values['New name']
            else:
                output_name = values['Output ROI']
            self.roi_combine.emit(values[0], values[1], values[2], output_name)

    @pyqtSlot()
    def show_shortcuts(self):
        ShortcutDialog(self)

    @pyqtSlot()
    def about(self):
        AboutDialog(self)

    @pyqtSlot(bool, int, int)
    @pyqtSlot(bool, int, int, str)
    def set_splash(self, is_splash, current_value, maximum_value, text= ''):
        if is_splash:
            self.mainUIWidget.setVisible(False)
            self.splash_progressbar.setMaximum(maximum_value)
            self.splash_progressbar.setValue(current_value)
            self.splash_movie.start()
            self.splash_text_label.setText(text)
            self.splashWidget.setVisible(True)
            self.menubar.setEnabled(False)
            QApplication.processEvents()
        else:
            self.splash_movie.stop()
            self.splashWidget.setVisible(False)
            self.mainUIWidget.setVisible(True)
            self.menubar.setEnabled(True)
            QApplication.processEvents()

    @pyqtSlot()
    def reduce_brush_size(self):
        self.brushsize_slider.setValue(max(self.brushsize_slider.value()-1, 1))
        self.brush_changed.emit()

    @pyqtSlot()
    def increase_brush_size(self):
        self.brushsize_slider.setValue(min(self.brushsize_slider.value() + 1, self.brushsize_slider.maximum()))
        self.brush_changed.emit()

    @pyqtSlot(int)
    def brushsliderCB(self, value):
        #self.brushsize_label.setText(str(value*2+1))
        self.brushsize_label.setText(str(value))
        self.brush_changed.emit()

    def get_brush(self):
        brush_size = int(self.brushsize_label.text())
        brush_type = self.BRUSH_SQUARE
        if self.circlebrush_button.isChecked():
            brush_type = self.BRUSH_CIRCLE
        return brush_type, brush_size

    def get_erase_from_all_rois(self):
        return self.eraseFromAllROIs_checkbox.isChecked()

    def get_edit_mode(self):
        if self.editmode_combo.currentText() == self.EDITMODE_MASK:
            return self.EDITMODE_MASK
        else:
            return self.EDITMODE_CONTOUR

    @pyqtSlot(str)
    def set_edit_mode(self, mode):
        if mode == self.EDITMODE_MASK:
            self.subroi_widget.setVisible(False)
            self.brush_group.setVisible(True)
            self.maskedit_group.setVisible(True)
            self.contouredit_widget.setVisible(False)
            self.addpaint_button.setText("Paint")
            self.removeerase_button.setText("Erase")
            if self.edit_state == self.TRANSLATE_STATE or self.edit_state == self.ROTATE_STATE:
                self.edit_state = self.NO_STATE
                self.temp_edit_state = None
            self.translateContour_button.setChecked(False)
            self.rotateContour_button.setChecked(False)
            #self.removeall_button.setVisible(False)
            self.eraseFromAllROIs_checkbox.setVisible(self.edit_state == self.REMOVE_STATE)
            self.editmode_changed.emit(self.EDITMODE_MASK)
        else:
            self.subroi_widget.setVisible(True)
            self.brush_group.setVisible(False)
            self.maskedit_group.setVisible(False)
            self.contouredit_widget.setVisible(True)
            self.addpaint_button.setText("Add/Move")
            self.removeerase_button.setText("Remove")
            #self.removeall_button.setVisible(True)
            self.eraseFromAllROIs_checkbox.setVisible(False)
            self.editmode_changed.emit(self.EDITMODE_CONTOUR)

    @pyqtSlot(bool)
    def undo_enable(self, enable):
        self.undoButton.setEnabled(enable)

    @pyqtSlot(bool)
    def redo_enable(self, enable):
        self.redoButton.setEnabled(enable)

    def confirm(self, text):
        w = QMessageBox.warning(self, "Warning", text, QMessageBox.Ok | QMessageBox.Cancel)
        return w == QMessageBox.Ok

    @pyqtSlot(str)
    def alert(self, text):
        QMessageBox.warning(self, "Warning", text, QMessageBox.Ok)

    @pyqtSlot(list)
    def set_available_classes(self, classes):
        self.classification_combo.clear()
        self.classification_combo.addItems(classes)
        self.classification_combo.addItem("None") # always add the "None" class

    def get_available_classes(self):
        return [self.classification_combo.itemText(i) for i in range(self.classification_combo.count())]

    @pyqtSlot(str)
    def set_class(self, class_str):
        self.classification_combo.setCurrentText(class_str)

    def get_class(self):
        return self.classification_combo.currentText()

    @pyqtSlot()
    def manage_edit_toggle(self):
        # make sure that only one add/remove knot button is pressed or translate/rotate
        senderObj = self.sender()
        if not senderObj.isChecked(): # this is a signal to uncheck the button: reset to no state
            self.edit_state = self.NO_STATE
        else:
            self.edit_state = self.state_buttons_dict[senderObj]
        self.manage_state_buttons(self.edit_state)

    def manage_state_buttons(self, state):
        for obj, obj_state in self.state_buttons_dict.items():
            if state == obj_state:
                obj.setChecked(True)
            else:
                obj.setChecked(False)
        if self.get_edit_mode() == self.EDITMODE_MASK and state == self.REMOVE_STATE:
            self.eraseFromAllROIs_checkbox.setVisible(True)
        else:
            self.eraseFromAllROIs_checkbox.setVisible(False)

    @pyqtSlot()
    def restore_edit_button_state(self):
        self.manage_state_buttons(self.edit_state)
        self.temp_edit_state = None
        self.brush_changed.emit()

    @pyqtSlot(int)
    def set_temp_edit_button_state(self, temp_state):
        self.temp_edit_state = temp_state
        self.manage_state_buttons(temp_state)
        self.brush_changed.emit()

    def get_edit_button_state(self):
        if self.temp_edit_state is not None: return self.temp_edit_state
        return self.edit_state

    @pyqtSlot(str)
    def set_class(self, class_str: str):
        self.classification_combo.setCurrentText(class_str)

    @pyqtSlot(dict)
    def set_rois_list(self, roi_dict):
        cur_roi = self.current_roi
        cur_subroi = self.current_subroi
        self.suppress_roi_change_emit = True
        self.all_rois = roi_dict
        self.roi_combo.clear()
        for roi_name in list(self.all_rois):
            if not roi_name:
                del self.all_rois[roi_name] # remove empty name
                continue
            self.roi_combo.addItem(roi_name)

        # try to reset the previous selection
        self.set_current_roi(cur_roi, cur_subroi)
        if not self.roi_combo.currentText():
            self.actionCopy_roi.setEnabled(False)
            self.actionCombine_roi.setEnabled(False)
            self.actionMultiple_combine.setEnabled(False)
            self.actionRemove_overlap.setEnabled(False)
        else:
            self.actionCopy_roi.setEnabled(True)
            self.actionCombine_roi.setEnabled(True)
            self.actionMultiple_combine.setEnabled(True)
            self.actionRemove_overlap.setEnabled(True)

    @pyqtSlot(str, int)
    def set_current_roi(self, current_roi_name, current_subroi_number=-1):
        if not self.all_rois:
            self.roi_combo.setEnabled(False)
            self.subroi_combo.setEnabled(False)
            self.roi_remove_button.setEnabled(False)
            self.subroi_remove_button.setEnabled(False)
            self.subroi_add_button.setEnabled(False)
            self.subroi_combo.setCurrentText("")
            self.valid_roi_selected = False
            return
        else:
            self.roi_combo.setEnabled(True)
            self.subroi_combo.setEnabled(True)
            self.roi_remove_button.setEnabled(True)
            self.subroi_remove_button.setEnabled(True)
            self.subroi_add_button.setEnabled(True)

        self.suppress_roi_change_emit = True

        if self.roi_combo.findText(current_roi_name) >= 0:
            self.current_roi = current_roi_name
            self.roi_combo.setCurrentText(current_roi_name)
        else:
            self.roi_combo.setCurrentIndex(0)
            self.current_roi = self.roi_combo.currentText()

        self.repopulate_subrois(current_subroi_number)

        self.suppress_roi_change_emit = False
        self.subroi_combo.setCurrentIndex(self.current_subroi)
        self.valid_roi_selected = True

    @pyqtSlot()
    def repopulate_subrois(self, current_subroi_number=-1):
        # populate subroi combo
        try:
            n_subrois = self.all_rois[self.current_roi]
        except KeyError:
            return
        # print("N_Subrois:", n_subrois)
        if n_subrois > current_subroi_number >= 0:
            self.current_subroi = current_subroi_number
        else:
            self.current_subroi = 0

        self.subroi_combo.clear()
        for n in range(n_subrois):
            self.subroi_combo.addItem(str(n))

    def valid_roi(self):
        return self.valid_roi_selected

    @pyqtSlot()
    def send_roi_changed(self):
        if self.suppress_roi_change_emit:
            return
        #print("Roi change:", self.roi_combo.currentText(), self.subroi_combo.currentIndex())
        self.roi_changed.emit(*self.get_current_roi_subroi())

    @pyqtSlot(name="delete_roi")  # it needs a specific name because of the decorator, Otherwise it will be overwritten by the next slot using the same decorator
    @ask_confirm("This will delete the ROI in all slices!")
    def delete_roi(self, *args, **kwargs):
        self.roi_deleted.emit(self.current_roi)

    @pyqtSlot(name="delete_subroi")
    @ask_confirm("This will delete the sub-ROI in the current slice!")
    def delete_subroi(self, *args, **kwargs):
        self.subroi_deleted.emit(self.current_subroi)

    def get_current_roi_subroi(self):
        self.current_roi = self.roi_combo.currentText()
        self.current_subroi = self.subroi_combo.currentIndex()
        return self.current_roi, self.current_subroi

    @pyqtSlot()
    def add_roi(self):
        new_roi_name, ok = QInputDialog.getText(self, "ROI Name", "Insert the name of the new ROI")
        if new_roi_name and ok:
            self.roi_added.emit(new_roi_name)

    @pyqtSlot(name="clear_roi")
    @ask_confirm("This will clear the ROI of the current slice")
    def clear_roi(self):
        self.roi_clear.emit()

    @pyqtSlot()
    def on_classification_changed(self):
        cur_class = self.classification_combo.currentText()
        if cur_class == 'None':
            self.autosegment_button.setEnabled(False)
        else:
            self.autosegment_button.setEnabled(True)
        self.classification_changed.emit(cur_class)

    @pyqtSlot()
    @ask_confirm("This will replace all the classifications in the dataset")
    def on_classification_change_all(self):
        self.classification_change_all.emit(self.classification_combo.currentText())

    @pyqtSlot()
    def on_do_segmentation(self, *args, **kwargs):
        cur_slice = self.muscle_segmentation_window.curImage
        min_slice = 0
        max_slice = len(self.muscle_segmentation_window.imList)-1
        accept, values = GenericInputDialog.show_dialog('Define slice range', [
                GenericInputDialog.IntSpinInput('Start slice', cur_slice, min_slice, max_slice),
                GenericInputDialog.IntSpinInput('End slice', cur_slice, min_slice, max_slice)
            ], self, message='Warning! This might replace the existing segmentation!')
        if accept:
            self.do_autosegment.emit(values[0], values[1])

    @pyqtSlot()
    @ask_confirm("This will calculate nonrigid transformations for all slices. It will take a few minutes")
    def do_registration(self):
        self.calculate_transforms.emit()

    @pyqtSlot()
    def load_mask_clicked(self):
        if config.GlobalConfig['ENABLE_NIFTI']:
            filter = 'Image files (*.dcm *.ima *.nii *.nii.gz *.npy *.npz);;Dicom files (*.dcm *.ima);;Nifti files (*.nii *.nii.gz);;Numpy files (*.npy *.npz);;All files ()'
        else:
            filter = 'Image files (*.dcm *.ima *.npy *.npz);;Dicom files (*.dcm *.ima);;Numpy files (*.npy *.npz);;All files ()'

        maskFile, _ = QFileDialog.getOpenFileName(self, caption='Select mask to import',
                                                  filter=filter)
        if maskFile:
            self.mask_import.emit(maskFile)

    def load_multi_mask_clicked(self):
        maskDir = QFileDialog.getExistingDirectory(self, caption='Select folder containing other DICOM folders or Nifti files')

        if maskDir:
            self.mask_import.emit(maskDir)

    @pyqtSlot()
    def loadData_clicked(self):
        if config.GlobalConfig['ENABLE_NIFTI']:
            filter = 'Image files (*.dcm *.ima *.nii *.nii.gz *.npy *.npz);;Dicom files (*.dcm *.ima);;Nifti files (*.nii *.nii.gz);;Numpy files (*.npy);;Data + Mask bundle (*npz);;All files ()'
        else:
            filter = 'Image files (*.dcm *.ima *.npy *.npz);;Dicom files (*.dcm *.ima);;Numpy files (*.npy);;Data + Mask bundle (*npz);;All files ()'

        dataFile, _ = QFileDialog.getOpenFileName(self, caption='Select dataset to import',
                                                  filter=filter)
        if dataFile:
            classifications = [(self.classification_combo.itemText(i), self.classification_combo.itemText(i)) for i in range(self.classification_combo.count())]
            if config.GlobalConfig['USE_CLASSIFIER']:
                classifications.insert(0, ('Automatic', ''))
            accepted, chosen_class = GenericInputDialog.show_dialog("Choose classification",
                                                                    [GenericInputDialog.OptionInput("Classification", classifications)])
            if not accepted:
                return
            self.data_open.emit(dataFile, chosen_class[0])

    @pyqtSlot()
    def importROI_clicked(self):
        roiFile, _ = QFileDialog.getOpenFileName(self, caption='Select ROI file to import',
                                                 filter='ROI Pickle files (*.p);;All files ()')
        if roiFile:
            self.roi_import.emit(roiFile)

    @pyqtSlot()
    def exportROI_clicked(self):
        roiFile, _ = QFileDialog.getSaveFileName(self, caption='Select ROI file to export',
                                                 filter='ROI Pickle files (*.p);;All files ()')
        if roiFile:
            self.roi_export.emit(roiFile)

    def set_exports_enabled(self, numpy=True, dicom=True, nifti=True):
        self.actionSave_as_Dicom.setEnabled(dicom)
        self.actionSave_as_Compact_Dicom.setEnabled(dicom)
        self.menuSave_as_Numpy.setEnabled(numpy)
        self.actionSave_as_Nifti.setEnabled(nifti)
        self.actionSave_as_Compact_Nifti.setEnabled(nifti)

    @pyqtSlot(str)
    def export_masks_dir(self, output_type):
        dir_out = QFileDialog.getExistingDirectory(self, caption=f'Select directory to export as {output_type}')
        if dir_out:
            self.masks_export.emit(dir_out, output_type)

    @pyqtSlot()
    def export_masks_npz(self):
        file_out, _ = QFileDialog.getSaveFileName(self, caption='Select npz file to export',
                                                  filter='Numpy array archive (*.npz);;All files ()')
        if file_out:
            self.masks_export.emit(file_out, 'npz')

    @pyqtSlot()
    def export_masks_compact_nifti(self):
        file_out, _ = QFileDialog.getSaveFileName(self, caption='Select Nifti file to export',
                                                  filter='Nifti files (*.nii *.nii.gz);;All files ()')
        if file_out:
            self.masks_export.emit(file_out, 'compact_nifti')

    @pyqtSlot()
    def calculate_statistics(self):
        file_out, _ = QFileDialog.getSaveFileName(self, caption='Select csv file to save the statistics',
                                                  filter='CSV File (*.csv);;All files ()')
        if file_out:
            self.statistics_calc.emit(file_out)

    @pyqtSlot()
    def calculate_radiomics(self):
        accept, radiomics_props = GenericInputDialog.show_dialog('PyRadiomics Options',
                                                                 [
                                                                     GenericInputDialog.BooleanInput('Quantize gray levels', True),
                                                                     GenericInputDialog.IntSpinInput('Quantization levels', 32, 0, 1024),
                                                                     GenericInputDialog.IntSliderInput('Mask erosion (px)', 0)], self)

        if not accept: return
        file_out, _ = QFileDialog.getSaveFileName(self, caption='Select csv file to save the statistics',
                                                  filter='CSV File (*.csv);;All files ()')
        if file_out:
            self.radiomics_calc.emit(file_out, radiomics_props[0], radiomics_props[1], radiomics_props[2])


    @pyqtSlot()
    @ask_confirm(f'This action will improve the model through incremental learning.\n' +
                 f'At least {config.GlobalConfig["IL_MIN_SLICES"]} slices are required for the operation.\n' +
                 f'It might take a few minutes. Continue?')
    def do_incremental_learn(self):
        self.incremental_learn.emit()
