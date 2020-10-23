# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'toolbox.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SegmentationToolbox(object):
    def setupUi(self, SegmentationToolbox):
        SegmentationToolbox.setObjectName("SegmentationToolbox")
        SegmentationToolbox.resize(267, 735)
        self.centralwidget = QtWidgets.QWidget(SegmentationToolbox)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.classification_combo = QtWidgets.QComboBox(self.centralwidget)
        self.classification_combo.setObjectName("classification_combo")
        self.verticalLayout.addWidget(self.classification_combo)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.undoButton = QtWidgets.QPushButton(self.groupBox)
        self.undoButton.setEnabled(False)
        self.undoButton.setObjectName("undoButton")
        self.horizontalLayout_3.addWidget(self.undoButton)
        self.redoButton = QtWidgets.QPushButton(self.groupBox)
        self.redoButton.setEnabled(False)
        self.redoButton.setObjectName("redoButton")
        self.horizontalLayout_3.addWidget(self.redoButton)
        self.verticalLayout.addWidget(self.groupBox)
        self.ROI_group = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ROI_group.sizePolicy().hasHeightForWidth())
        self.ROI_group.setSizePolicy(sizePolicy)
        self.ROI_group.setObjectName("ROI_group")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.ROI_group)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.roi_combo = QtWidgets.QComboBox(self.ROI_group)
        self.roi_combo.setObjectName("roi_combo")
        self.horizontalLayout.addWidget(self.roi_combo)
        self.roi_add_button = QtWidgets.QToolButton(self.ROI_group)
        self.roi_add_button.setObjectName("roi_add_button")
        self.horizontalLayout.addWidget(self.roi_add_button)
        self.roi_remove_button = QtWidgets.QToolButton(self.ROI_group)
        self.roi_remove_button.setObjectName("roi_remove_button")
        self.horizontalLayout.addWidget(self.roi_remove_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.ROI_group)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.subroi_combo = QtWidgets.QComboBox(self.ROI_group)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.subroi_combo.sizePolicy().hasHeightForWidth())
        self.subroi_combo.setSizePolicy(sizePolicy)
        self.subroi_combo.setObjectName("subroi_combo")
        self.horizontalLayout_2.addWidget(self.subroi_combo)
        self.subroi_add_button = QtWidgets.QToolButton(self.ROI_group)
        self.subroi_add_button.setObjectName("subroi_add_button")
        self.horizontalLayout_2.addWidget(self.subroi_add_button)
        self.subroi_remove_button = QtWidgets.QToolButton(self.ROI_group)
        self.subroi_remove_button.setObjectName("subroi_remove_button")
        self.horizontalLayout_2.addWidget(self.subroi_remove_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addWidget(self.ROI_group)
        self.autosegment_button = QtWidgets.QPushButton(self.centralwidget)
        self.autosegment_button.setObjectName("autosegment_button")
        self.verticalLayout.addWidget(self.autosegment_button)
        self.knots_group = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.knots_group.sizePolicy().hasHeightForWidth())
        self.knots_group.setSizePolicy(sizePolicy)
        self.knots_group.setObjectName("knots_group")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.knots_group)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.addknot_button = QtWidgets.QPushButton(self.knots_group)
        self.addknot_button.setCheckable(True)
        self.addknot_button.setChecked(False)
        self.addknot_button.setAutoExclusive(False)
        self.addknot_button.setObjectName("addknot_button")
        self.verticalLayout_3.addWidget(self.addknot_button)
        self.removeknot_button = QtWidgets.QPushButton(self.knots_group)
        self.removeknot_button.setCheckable(True)
        self.removeknot_button.setAutoExclusive(False)
        self.removeknot_button.setObjectName("removeknot_button")
        self.verticalLayout_3.addWidget(self.removeknot_button)
        self.removeall_button = QtWidgets.QPushButton(self.knots_group)
        self.removeall_button.setObjectName("removeall_button")
        self.verticalLayout_3.addWidget(self.removeall_button)
        self.verticalLayout.addWidget(self.knots_group)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.optimizeButton = QtWidgets.QPushButton(self.groupBox_2)
        self.optimizeButton.setObjectName("optimizeButton")
        self.verticalLayout_4.addWidget(self.optimizeButton)
        self.simplifyButton = QtWidgets.QPushButton(self.groupBox_2)
        self.simplifyButton.setObjectName("simplifyButton")
        self.verticalLayout_4.addWidget(self.simplifyButton)
        self.registrationGroup = QtWidgets.QGroupBox(self.groupBox_2)
        self.registrationGroup.setObjectName("registrationGroup")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.registrationGroup)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.propagateBackButton = QtWidgets.QPushButton(self.registrationGroup)
        self.propagateBackButton.setObjectName("propagateBackButton")
        self.horizontalLayout_4.addWidget(self.propagateBackButton)
        self.propagateForwardButton = QtWidgets.QPushButton(self.registrationGroup)
        self.propagateForwardButton.setObjectName("propagateForwardButton")
        self.horizontalLayout_4.addWidget(self.propagateForwardButton)
        self.verticalLayout_5.addLayout(self.horizontalLayout_4)
        self.calcTransformsButton = QtWidgets.QPushButton(self.registrationGroup)
        self.calcTransformsButton.setObjectName("calcTransformsButton")
        self.verticalLayout_5.addWidget(self.calcTransformsButton)
        self.verticalLayout_4.addWidget(self.registrationGroup)
        self.verticalLayout.addWidget(self.groupBox_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        SegmentationToolbox.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SegmentationToolbox)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 267, 28))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSave_masks = QtWidgets.QMenu(self.menuFile)
        self.menuSave_masks.setObjectName("menuSave_masks")
        SegmentationToolbox.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SegmentationToolbox)
        self.statusbar.setObjectName("statusbar")
        SegmentationToolbox.setStatusBar(self.statusbar)
        self.actionImport_ROIs = QtWidgets.QAction(SegmentationToolbox)
        self.actionImport_ROIs.setObjectName("actionImport_ROIs")
        self.actionExport_ROIs = QtWidgets.QAction(SegmentationToolbox)
        self.actionExport_ROIs.setObjectName("actionExport_ROIs")
        self.actionImport_masks = QtWidgets.QAction(SegmentationToolbox)
        self.actionImport_masks.setObjectName("actionImport_masks")
        self.actionSave_as_Numpy = QtWidgets.QAction(SegmentationToolbox)
        self.actionSave_as_Numpy.setObjectName("actionSave_as_Numpy")
        self.actionSave_as_Dicom = QtWidgets.QAction(SegmentationToolbox)
        self.actionSave_as_Dicom.setObjectName("actionSave_as_Dicom")
        self.actionSave_as_Nifti = QtWidgets.QAction(SegmentationToolbox)
        self.actionSave_as_Nifti.setObjectName("actionSave_as_Nifti")
        self.actionLoad_data = QtWidgets.QAction(SegmentationToolbox)
        self.actionLoad_data.setObjectName("actionLoad_data")
        self.menuSave_masks.addAction(self.actionSave_as_Numpy)
        self.menuSave_masks.addAction(self.actionSave_as_Dicom)
        self.menuSave_masks.addAction(self.actionSave_as_Nifti)
        self.menuFile.addAction(self.actionLoad_data)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionImport_ROIs)
        self.menuFile.addAction(self.actionExport_ROIs)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionImport_masks)
        self.menuFile.addAction(self.menuSave_masks.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(SegmentationToolbox)
        QtCore.QMetaObject.connectSlotsByName(SegmentationToolbox)

    def retranslateUi(self, SegmentationToolbox):
        _translate = QtCore.QCoreApplication.translate
        SegmentationToolbox.setWindowTitle(_translate("SegmentationToolbox", "MainWindow"))
        self.groupBox.setTitle(_translate("SegmentationToolbox", "History"))
        self.undoButton.setText(_translate("SegmentationToolbox", "Undo"))
        self.redoButton.setText(_translate("SegmentationToolbox", "Redo"))
        self.ROI_group.setTitle(_translate("SegmentationToolbox", "ROI"))
        self.roi_add_button.setText(_translate("SegmentationToolbox", "+"))
        self.roi_remove_button.setText(_translate("SegmentationToolbox", "-"))
        self.label.setText(_translate("SegmentationToolbox", " ⮡"))
        self.subroi_add_button.setText(_translate("SegmentationToolbox", "+"))
        self.subroi_remove_button.setText(_translate("SegmentationToolbox", "-"))
        self.autosegment_button.setText(_translate("SegmentationToolbox", "AutoSegment"))
        self.knots_group.setTitle(_translate("SegmentationToolbox", "Knots"))
        self.addknot_button.setText(_translate("SegmentationToolbox", "Add/Move"))
        self.removeknot_button.setText(_translate("SegmentationToolbox", "Remove"))
        self.removeall_button.setText(_translate("SegmentationToolbox", "Clear"))
        self.groupBox_2.setTitle(_translate("SegmentationToolbox", "Contour"))
        self.optimizeButton.setText(_translate("SegmentationToolbox", "Snap to edges"))
        self.simplifyButton.setText(_translate("SegmentationToolbox", "Simplify"))
        self.registrationGroup.setTitle(_translate("SegmentationToolbox", "Registration"))
        self.propagateBackButton.setText(_translate("SegmentationToolbox", "Propagate back"))
        self.propagateForwardButton.setText(_translate("SegmentationToolbox", "Propagate forward"))
        self.calcTransformsButton.setText(_translate("SegmentationToolbox", "Claculate registrations"))
        self.menuFile.setTitle(_translate("SegmentationToolbox", "File..."))
        self.menuSave_masks.setTitle(_translate("SegmentationToolbox", "Save masks..."))
        self.actionImport_ROIs.setText(_translate("SegmentationToolbox", "Import ROI file..."))
        self.actionExport_ROIs.setText(_translate("SegmentationToolbox", "Export ROI file..."))
        self.actionImport_masks.setText(_translate("SegmentationToolbox", "Import masks..."))
        self.actionSave_as_Numpy.setText(_translate("SegmentationToolbox", "Save as Numpy..."))
        self.actionSave_as_Dicom.setText(_translate("SegmentationToolbox", "Save as Dicom..."))
        self.actionSave_as_Nifti.setText(_translate("SegmentationToolbox", "Save as Nifti..."))
        self.actionLoad_data.setText(_translate("SegmentationToolbox", "Load data..."))
