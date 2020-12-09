#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:07:32 2020

@author: francesco
"""
import functools

from ui.ToolboxUI import Ui_SegmentationToolbox
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QInputDialog, QFileDialog
import traceback

def ask_confirm(text):
    def decorator_confirm(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj._confirm(text):
                func(obj, *args, **kwargs)

        return wrapper

    return decorator_confirm


class ToolboxWindow(QMainWindow, Ui_SegmentationToolbox):
    do_autosegment = pyqtSignal()
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

    undo = pyqtSignal()
    redo = pyqtSignal()

    roi_import = pyqtSignal(str)
    roi_export = pyqtSignal(str)

    masks_export = pyqtSignal(str, str)

    data_open = pyqtSignal(str)

    NO_STATE = 0
    ADD_STATE = 1
    REMOVE_STATE = 2

    BRUSH_CIRCLE = 'Circle'
    BRUSH_SQUARE = 'Square'

    EDITMODE_MASK = 'Mask'
    EDITMODE_CONTOUR = 'Contour'


    def __init__(self, activate_registration=True):
        super(ToolboxWindow, self).__init__()
        self.setupUi(self)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.setWindowTitle("Segmentation Toolbox")
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

        self.editState = self.NO_STATE
        self.tempEditState = None

        self.removeall_button.clicked.connect(self.clear_roi)

        self.classification_combo.currentTextChanged.connect(self.on_classification_changed)
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

        self.actionImport_ROIs.triggered.connect(self.importROI_clicked)
        self.actionExport_ROIs.triggered.connect(self.exportROI_clicked)

        self.actionLoad_data.triggered.connect(self.loadData_clicked)

        self.actionSave_as_Dicom.triggered.connect(lambda: self.export_masks_dir('dicom'))
        self.actionSaveNPY.triggered.connect(lambda: self.export_masks_dir('npy'))
        self.actionSave_as_Nifti.triggered.connect(lambda: self.export_masks_dir('nifti'))
        self.actionSaveNPZ.triggered.connect(self.export_masks_npz)

        self.editmode_combo.currentTextChanged.connect(lambda : self.setEditMode(self.editmode_combo.currentText()))
        self.editmode_combo.setCurrentText('Mask')
        self.setEditMode('Mask')

        self.brushsize_slider.valueChanged.connect(lambda : self.brushsize_label.setText(str(self.brushsize_slider.value())))

    def getBrush(self):
        brush_size = self.brushsize_slider.value()
        brush_type = self.BRUSH_SQUARE
        if self.circlebrush_button.isChecked():
            brush_type = self.BRUSH_CIRCLE
        return brush_type, brush_size

    def getEditMode(self):
        if self.editmode_combo.currentText() == self.EDITMODE_MASK:
            return self.EDITMODE_MASK
        else:
            return self.EDITMODE_CONTOUR

    @pyqtSlot(str)
    def setEditMode(self, mode):
        if mode == self.EDITMODE_MASK:
            self.subroi_widget.setVisible(False)
            self.brush_group.setVisible(True)
            self.contouredit_widget.setVisible(False)
            self.addpaint_button.setText("Paint")
            self.removeerase_button.setText("Erase")
            #self.removeall_button.setVisible(False)
        else:
            self.subroi_widget.setVisible(True)
            self.brush_group.setVisible(False)
            self.contouredit_widget.setVisible(True)
            self.addpaint_button.setText("Add/Move")
            self.removeerase_button.setText("Remove")
            #self.removeall_button.setVisible(True)

    @pyqtSlot(bool)
    def undoEnable(self, enable):
        self.undoButton.setEnabled(enable)

    @pyqtSlot(bool)
    def redoEnable(self, enable):
        self.redoButton.setEnabled(enable)

    def _confirm(self, text):
        w = QMessageBox.warning(self, "Warning", text, QMessageBox.Ok | QMessageBox.Cancel)
        return w == QMessageBox.Ok

    @pyqtSlot(list)
    def set_available_classes(self, classes):
        self.classification_combo.clear()
        self.classification_combo.addItems(classes)

    @pyqtSlot(str)
    def set_class(self, class_str):
        self.classification_combo.setCurrentText(class_str)

    def get_class(self):
        return self.classification_combo.currentText()

    @pyqtSlot()
    def manage_edit_toggle(self):
        # make sure that only one add/remove knot button is pressed
        if self.addpaint_button.isChecked() and self.removeerase_button.isChecked():
            if self.sender() == self.addpaint_button:
                self.removeerase_button.setChecked(False)
            else:
                self.addpaint_button.setChecked(False)

        # set the permanent state
        if self.addpaint_button.isChecked():
            self.editState = self.ADD_STATE
        elif self.removeerase_button.isChecked():
            self.editState = self.REMOVE_STATE
        else:
            self.editState = self.NO_STATE

    @pyqtSlot()
    def restore_edit_button_state(self):
        self.addpaint_button.setChecked(self.editState == self.ADD_STATE)
        self.removeerase_button.setChecked(self.editState == self.REMOVE_STATE)
        self.tempEditState = None

    @pyqtSlot(int)
    def set_temp_edit_button_state(self, tempState):
        self.tempEditState = tempState
        self.addpaint_button.setChecked(tempState == self.ADD_STATE)
        self.removeerase_button.setChecked(tempState == self.REMOVE_STATE)

    def get_edit_button_state(self):
        if self.tempEditState is not None: return self.tempEditState
        return self.editState

    @pyqtSlot(list)
    def set_classes_list(self, classes: list):
        self.classification_combo.clear()
        self.classification_combo.addItems(classes)

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
        for roi_name in self.all_rois:
            self.roi_combo.addItem(roi_name)

        # try to reset the previous selection
        self.set_current_roi(cur_roi, cur_subroi)

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
        print("Roi change:", self.roi_combo.currentText(), self.subroi_combo.currentIndex())
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
        self.classification_changed.emit(self.classification_combo.currentText())

    @pyqtSlot(name="on_do_segmentation")
    @ask_confirm("This might replace the existing segmentation")
    def on_do_segmentation(self, *args, **kwargs):
        self.do_autosegment.emit()

    @pyqtSlot()
    @ask_confirm("This will calculate nonrigid transformations for all slices. It will take a few minutes")
    def do_registration(self):
        self.do_registration.emit()

    @pyqtSlot()
    def loadData_clicked(self):
        dataFile, _ = QFileDialog.getOpenFileName(self, caption='Select dataset to import',
                                                  filter='Image files (*.dcm *.ima *.nii *.nii.gz *.npy);;Dicom files (*.dcm *.ima);;Nifti files (*.nii *.nii.gz);;Numpy files (*.npy);;All files (*.*)')
        if dataFile:
            self.data_open.emit(dataFile)

    @pyqtSlot()
    def importROI_clicked(self):
        roiFile, _ = QFileDialog.getOpenFileName(self, caption='Select ROI file to import',
                                                 filter='ROI Pickle files (*.p);;All files (*.*)')
        if roiFile:
            self.roi_import.emit(roiFile)

    @pyqtSlot()
    def exportROI_clicked(self):
        roiFile, _ = QFileDialog.getSaveFileName(self, caption='Select ROI file to export',
                                                 filter='ROI Pickle files (*.p);;All files (*.*)')
        if roiFile:
            self.roi_export.emit(roiFile)

    def set_exports_enabled(self, numpy=True, dicom=True, nifti=True):
        self.actionSave_as_Dicom.setEnabled(dicom)
        self.menuSave_as_Numpy.setEnabled(numpy)
        self.actionSave_as_Nifti.setEnabled(nifti)

    @pyqtSlot(str)
    def export_masks_dir(self, output_type):
        dir_out = QFileDialog.getExistingDirectory(self, caption=f'Select directory to export as {output_type}')
        if dir_out:
            self.masks_export.emit(dir_out, output_type)

    @pyqtSlot()
    def export_masks_npz(self):
        file_out, _ = QFileDialog.getSaveFileName(self, caption='Select npz file to export',
                                                  filter='Numpy array archive (*.npz);;All files (*.*)')
        if file_out:
            self.masks_export.emit(file_out, 'npz')
