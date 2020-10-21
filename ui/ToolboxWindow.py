#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:07:32 2020

@author: francesco
"""
import functools

from ui.ToolboxUI import Ui_SegmentationToolbox
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QInputDialog


def ask_confirm(text):
    def decorator_confirm(func):
        #@functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            if obj._confirm(text):
                func(obj, *args, **kwargs)
        return wrapper
    return decorator_confirm


class ToolboxWindow(QMainWindow, Ui_SegmentationToolbox):

    autosegment_triggered = pyqtSignal()
    roi_added = pyqtSignal(str)
    roi_deleted = pyqtSignal(str)
    subroi_added = pyqtSignal(int)
    subroi_deleted = pyqtSignal(int)
    roi_changed = pyqtSignal(str, int)
    roi_clear = pyqtSignal()
    classification_changed = pyqtSignal(str)

    NO_STATE=0
    ADD_STATE=1
    REMOVE_STATE=2

    def __init__(self):
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

        self.addknot_button.clicked.connect(self.manage_knot_toggle)
        self.removeknot_button.clicked.connect(self.manage_knot_toggle)

        self.knotState = self.NO_STATE
        self.tempKnotState = None

        self.removeall_button.clicked.connect(self.clear_roi)

        self.classification_combo.currentTextChanged.connect(self.on_classification_changed)
        self.autosegment_button.clicked.connect(self.on_do_segmentation)

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
    def manage_knot_toggle(self):
        # make sure that only one add/remove knot button is pressed
        if self.addknot_button.isChecked() and self.removeknot_button.isChecked():
            if self.sender() == self.addknot_button:
                self.removeknot_button.setChecked(False)
            else:
                self.addknot_button.setChecked(False)

        # set the permanent state
        if self.addknot_button.isChecked():
            self.knotState = self.ADD_STATE
        elif self.removeknot_button.isChecked():
            self.knotState = self.REMOVE_STATE
        else:
            self.knotState = self.NO_STATE

    @pyqtSlot()
    def restore_knot_button_state(self):
        self.addknot_button.setChecked(self.knotState == self.ADD_STATE)
        self.removeknot_button.setChecked(self.knotState == self.REMOVE_STATE)
        self.tempKnotState = None

    @pyqtSlot(int)
    def set_temp_knot_button_state(self, tempState):
        self.tempKnotState = tempState
        self.addknot_button.setChecked(tempState == self.ADD_STATE)
        self.removeknot_button.setChecked(tempState == self.REMOVE_STATE)

    def get_knot_button_state(self):
        if self.tempKnotState is not None: return self.tempKnotState
        return self.knotState

    @pyqtSlot(list)
    def set_classes_list(self, classes: list):
        self.classification_combo.clear()
        self.classification_combo.addItems(classes)

    @pyqtSlot(str)
    def set_class(self, class_str: str):
        self.classification_combo.setCurrentText(class_str)

    @pyqtSlot(dict)
    def set_rois_list(self, roi_dict):
        self.suppress_roi_change_emit = True
        self.all_rois = roi_dict
        self.roi_combo.clear()
        for roi_name in self.all_rois:
            self.roi_combo.addItem(roi_name)
        
        # try to reset the previous selection
        self.set_current_roi(self.current_roi, self.current_subroi)


    @pyqtSlot(str, int)
    def set_current_roi(self, current_roi_name, current_subroi_number = 0):
        if not self.all_rois:
            self.roi_combo.setEnabled(False)
            self.subroi_combo.setEnabled(False)
            self.roi_remove_button.setEnabled(False)
            self.subroi_remove_button.setEnabled(False)
            self.valid_roi_selected = False
            return
        else:
            self.roi_combo.setEnabled(True)
            self.subroi_combo.setEnabled(True)
            self.roi_remove_button.setEnabled(True)
            self.subroi_remove_button.setEnabled(True)

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
    def repopulate_subrois(self, current_subroi_number = 0):
        # populate subroi combo
        try:
            n_subrois = len(self.all_rois[self.current_roi])
        except:
            return
        print("N_Subrois:", n_subrois)
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
        print("Roi change:",self.roi_combo.currentText(), self.subroi_combo.currentIndex())
        self.roi_changed.emit(self.roi_combo.currentText(), self.subroi_combo.currentIndex())

    @pyqtSlot(name="delete_roi") # it needs a specific name because of the decorator, Otherwise it will be overwritten by the next slot using the same decorator
    @ask_confirm("This will delete the ROI in all slices!")
    def delete_roi(self, *args, **kwargs):
        self.roi_deleted.emit(self.current_roi)

    @pyqtSlot(name="delete_subroi")
    @ask_confirm("This will delete the sub-ROI in all slices!")
    def delete_subroi(self, *args, **kwargs):
        self.subroi_deleted.emit(self.current_subroi)

    def get_current_roi_subroi(self):
        self.current_roi = self.roi_combo.currentText()
        self.current_subroi = self.subroi_combo.currentIndex()
        return self.current_roi, self.current_subroi

    @pyqtSlot()
    def add_roi(self):
        newRoiName, ok = QInputDialog.getText(self, "ROI Name", "Insert the name of the new ROI")
        if newRoiName and ok:
            self.roi_added.emit(newRoiName)

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
        self.autosegment_triggered.emit()