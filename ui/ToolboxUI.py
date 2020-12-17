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
        SegmentationToolbox.resize(294, 832)
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
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_6.addWidget(self.label_2)
        self.editmode_combo = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.editmode_combo.sizePolicy().hasHeightForWidth())
        self.editmode_combo.setSizePolicy(sizePolicy)
        self.editmode_combo.setObjectName("editmode_combo")
        self.editmode_combo.addItem("")
        self.editmode_combo.addItem("")
        self.horizontalLayout_6.addWidget(self.editmode_combo)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
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
        self.subroi_widget = QtWidgets.QWidget(self.ROI_group)
        self.subroi_widget.setObjectName("subroi_widget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.subroi_widget)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtWidgets.QLabel(self.subroi_widget)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.subroi_combo = QtWidgets.QComboBox(self.subroi_widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.subroi_combo.sizePolicy().hasHeightForWidth())
        self.subroi_combo.setSizePolicy(sizePolicy)
        self.subroi_combo.setObjectName("subroi_combo")
        self.horizontalLayout_2.addWidget(self.subroi_combo)
        self.subroi_add_button = QtWidgets.QToolButton(self.subroi_widget)
        self.subroi_add_button.setObjectName("subroi_add_button")
        self.horizontalLayout_2.addWidget(self.subroi_add_button)
        self.subroi_remove_button = QtWidgets.QToolButton(self.subroi_widget)
        self.subroi_remove_button.setObjectName("subroi_remove_button")
        self.horizontalLayout_2.addWidget(self.subroi_remove_button)
        self.verticalLayout_2.addWidget(self.subroi_widget)
        self.verticalLayout.addWidget(self.ROI_group)
        self.autosegment_button = QtWidgets.QPushButton(self.centralwidget)
        self.autosegment_button.setObjectName("autosegment_button")
        self.verticalLayout.addWidget(self.autosegment_button)
        self.edit_group = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.edit_group.sizePolicy().hasHeightForWidth())
        self.edit_group.setSizePolicy(sizePolicy)
        self.edit_group.setObjectName("edit_group")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.edit_group)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.addpaint_button = QtWidgets.QPushButton(self.edit_group)
        self.addpaint_button.setCheckable(True)
        self.addpaint_button.setChecked(False)
        self.addpaint_button.setAutoExclusive(False)
        self.addpaint_button.setObjectName("addpaint_button")
        self.horizontalLayout_8.addWidget(self.addpaint_button)
        self.removeerase_button = QtWidgets.QPushButton(self.edit_group)
        self.removeerase_button.setCheckable(True)
        self.removeerase_button.setAutoExclusive(False)
        self.removeerase_button.setObjectName("removeerase_button")
        self.horizontalLayout_8.addWidget(self.removeerase_button)
        self.removeall_button = QtWidgets.QPushButton(self.edit_group)
        self.removeall_button.setObjectName("removeall_button")
        self.horizontalLayout_8.addWidget(self.removeall_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.verticalLayout.addWidget(self.edit_group)
        self.brush_group = QtWidgets.QGroupBox(self.centralwidget)
        self.brush_group.setObjectName("brush_group")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.brush_group)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.circlebrush_button = QtWidgets.QToolButton(self.brush_group)
        self.circlebrush_button.setMinimumSize(QtCore.QSize(24, 24))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.circlebrush_button.setFont(font)
        self.circlebrush_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui/images/circle.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.circlebrush_button.setIcon(icon)
        self.circlebrush_button.setCheckable(True)
        self.circlebrush_button.setChecked(True)
        self.circlebrush_button.setAutoExclusive(True)
        self.circlebrush_button.setObjectName("circlebrush_button")
        self.horizontalLayout_5.addWidget(self.circlebrush_button)
        self.squarebrush_button = QtWidgets.QToolButton(self.brush_group)
        self.squarebrush_button.setMinimumSize(QtCore.QSize(24, 24))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("ui/images/square.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.squarebrush_button.setIcon(icon1)
        self.squarebrush_button.setCheckable(True)
        self.squarebrush_button.setAutoExclusive(True)
        self.squarebrush_button.setObjectName("squarebrush_button")
        self.horizontalLayout_5.addWidget(self.squarebrush_button)
        self.brushsize_slider = QtWidgets.QSlider(self.brush_group)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.brushsize_slider.sizePolicy().hasHeightForWidth())
        self.brushsize_slider.setSizePolicy(sizePolicy)
        self.brushsize_slider.setMinimum(0)
        self.brushsize_slider.setMaximum(50)
        self.brushsize_slider.setSingleStep(1)
        self.brushsize_slider.setPageStep(10)
        self.brushsize_slider.setProperty("value", 5)
        self.brushsize_slider.setOrientation(QtCore.Qt.Horizontal)
        self.brushsize_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.brushsize_slider.setObjectName("brushsize_slider")
        self.horizontalLayout_5.addWidget(self.brushsize_slider)
        self.brushsize_label = QtWidgets.QLabel(self.brush_group)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.brushsize_label.sizePolicy().hasHeightForWidth())
        self.brushsize_label.setSizePolicy(sizePolicy)
        self.brushsize_label.setMinimumSize(QtCore.QSize(20, 0))
        self.brushsize_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.brushsize_label.setObjectName("brushsize_label")
        self.horizontalLayout_5.addWidget(self.brushsize_label)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.brush_group)
        self.contouredit_widget = QtWidgets.QWidget(self.centralwidget)
        self.contouredit_widget.setObjectName("contouredit_widget")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.contouredit_widget)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.groupBox_2 = QtWidgets.QGroupBox(self.contouredit_widget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.optimizeButton = QtWidgets.QPushButton(self.groupBox_2)
        self.optimizeButton.setObjectName("optimizeButton")
        self.horizontalLayout_9.addWidget(self.optimizeButton)
        self.simplifyButton = QtWidgets.QPushButton(self.groupBox_2)
        self.simplifyButton.setObjectName("simplifyButton")
        self.horizontalLayout_9.addWidget(self.simplifyButton)
        self.verticalLayout_4.addLayout(self.horizontalLayout_9)
        self.verticalLayout_7.addWidget(self.groupBox_2)
        self.verticalLayout.addWidget(self.contouredit_widget)
        self.registrationGroup = QtWidgets.QGroupBox(self.centralwidget)
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
        self.verticalLayout.addWidget(self.registrationGroup)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        SegmentationToolbox.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SegmentationToolbox)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 294, 30))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSave_masks = QtWidgets.QMenu(self.menuFile)
        self.menuSave_masks.setObjectName("menuSave_masks")
        self.menuSave_as_Numpy = QtWidgets.QMenu(self.menuSave_masks)
        self.menuSave_as_Numpy.setObjectName("menuSave_as_Numpy")
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
        self.actionSave_as_Dicom = QtWidgets.QAction(SegmentationToolbox)
        self.actionSave_as_Dicom.setObjectName("actionSave_as_Dicom")
        self.actionSave_as_Nifti = QtWidgets.QAction(SegmentationToolbox)
        self.actionSave_as_Nifti.setObjectName("actionSave_as_Nifti")
        self.actionLoad_data = QtWidgets.QAction(SegmentationToolbox)
        self.actionLoad_data.setObjectName("actionLoad_data")
        self.actionSaveNPZ = QtWidgets.QAction(SegmentationToolbox)
        self.actionSaveNPZ.setObjectName("actionSaveNPZ")
        self.actionSaveNPY = QtWidgets.QAction(SegmentationToolbox)
        self.actionSaveNPY.setObjectName("actionSaveNPY")
        self.menuSave_as_Numpy.addAction(self.actionSaveNPZ)
        self.menuSave_as_Numpy.addAction(self.actionSaveNPY)
        self.menuSave_masks.addAction(self.menuSave_as_Numpy.menuAction())
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
        self.label_2.setText(_translate("SegmentationToolbox", "Edit:"))
        self.editmode_combo.setItemText(0, _translate("SegmentationToolbox", "Mask"))
        self.editmode_combo.setItemText(1, _translate("SegmentationToolbox", "Contour"))
        self.ROI_group.setTitle(_translate("SegmentationToolbox", "ROI"))
        self.roi_add_button.setText(_translate("SegmentationToolbox", "+"))
        self.roi_remove_button.setText(_translate("SegmentationToolbox", "-"))
        self.label.setText(_translate("SegmentationToolbox", " ⮡"))
        self.subroi_add_button.setText(_translate("SegmentationToolbox", "+"))
        self.subroi_remove_button.setText(_translate("SegmentationToolbox", "-"))
        self.autosegment_button.setText(_translate("SegmentationToolbox", "AutoSegment"))
        self.edit_group.setTitle(_translate("SegmentationToolbox", "Edit"))
        self.addpaint_button.setText(_translate("SegmentationToolbox", "Add/Move"))
        self.removeerase_button.setText(_translate("SegmentationToolbox", "Remove"))
        self.removeall_button.setText(_translate("SegmentationToolbox", "Clear"))
        self.brush_group.setTitle(_translate("SegmentationToolbox", "Brush"))
        self.brushsize_label.setText(_translate("SegmentationToolbox", "11"))
        self.groupBox_2.setTitle(_translate("SegmentationToolbox", "Contour"))
        self.optimizeButton.setText(_translate("SegmentationToolbox", "Snap to edges"))
        self.simplifyButton.setText(_translate("SegmentationToolbox", "Simplify"))
        self.registrationGroup.setTitle(_translate("SegmentationToolbox", "Registration"))
        self.propagateBackButton.setText(_translate("SegmentationToolbox", "Propagate back"))
        self.propagateForwardButton.setText(_translate("SegmentationToolbox", "Propagate forward"))
        self.calcTransformsButton.setText(_translate("SegmentationToolbox", "Calculate transforms"))
        self.menuFile.setTitle(_translate("SegmentationToolbox", "File..."))
        self.menuSave_masks.setTitle(_translate("SegmentationToolbox", "Save masks..."))
        self.menuSave_as_Numpy.setTitle(_translate("SegmentationToolbox", "Save as Numpy..."))
        self.actionImport_ROIs.setText(_translate("SegmentationToolbox", "Import ROI file..."))
        self.actionExport_ROIs.setText(_translate("SegmentationToolbox", "Export ROI file..."))
        self.actionImport_masks.setText(_translate("SegmentationToolbox", "Import masks..."))
        self.actionSave_as_Dicom.setText(_translate("SegmentationToolbox", "Save as Dicom..."))
        self.actionSave_as_Nifti.setText(_translate("SegmentationToolbox", "Save as Nifti..."))
        self.actionLoad_data.setText(_translate("SegmentationToolbox", "Load data..."))
        self.actionSaveNPZ.setText(_translate("SegmentationToolbox", "Single file"))
        self.actionSaveNPY.setText(_translate("SegmentationToolbox", "Multiple files"))