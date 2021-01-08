from __future__ import annotations

from typing import Any
from collections import OrderedDict
from PyQt5.QtWidgets import QDialog, QWidget, QSpinBox, QDoubleSpinBox, QLineEdit, QFormLayout, QVBoxLayout, \
    QHBoxLayout, QGroupBox, QDialogButtonBox, QLabel, QApplication, QSlider, QCheckBox
from PyQt5.QtCore import Qt
import abc
import sys


class MixedDict(OrderedDict):
    """
    A class that behaves as a list and as a dictionary.
    If the key is integer, returns the item at that position, otherwise behave as a dictionary

    """

    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

    def key_at(self, pos):
        return list(self.keys())[pos]

    def __getitem__(self, key):
        if type(key) == int:
            return OrderedDict.__getitem__(self, self.key_at(key))
        else:
            return OrderedDict.__getitem__(self, key)

    def __setitem__(self, key, value):
        if type(key) == int:
            dict_key = self.key_at(key)  # if this fails, it will raise a keyerror, and we are fine with that
        else:
            dict_key = key
        OrderedDict.__setitem__(self, dict_key, value)

    def __iter__(self):
        return iter(self.values())


class InputClass(abc.ABC):
    """
    Abstract implementation of an input class
    """

    @abc.abstractmethod
    def get_widget(self) -> QWidget:
        pass

    @abc.abstractmethod
    def get_label(self) -> str:
        pass

    @abc.abstractmethod
    def get_value(self) -> Any:
        pass


## Default input classes

class TextLineInput(InputClass):

    def __init__(self, label):
        self.label = label
        self.line_edit = QLineEdit()

    def get_widget(self) -> QWidget:
        return self.line_edit

    def get_label(self) -> str:
        return self.label

    def get_value(self) -> Any:
        return self.line_edit.text()


class IntSpinInput(InputClass):

    def __init__(self, label, initial_value=0, min=0, max=99, increment=1):
        self.label = label
        self.widget = QSpinBox()
        self.widget.setMinimum(min)
        self.widget.setMaximum(max)
        self.widget.setValue(initial_value)
        self.widget.setSingleStep(increment)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.value()


class FloatSpinInput(InputClass):

    def __init__(self, label, initial_value=0.0, min=0.0, max=99.0, increment=1.0):
        self.label = label
        self.widget = QDoubleSpinBox()
        self.widget.setMinimum(min)
        self.widget.setMaximum(max)
        self.widget.setValue(initial_value)
        self.widget.setSingleStep(increment)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.value()


class IntSliderInput(InputClass):

    def __init__(self, label, initial_value=0, min=0, max=99, increment=1):
        self.label = label
        self.widget = QWidget()
        self.slider = QSlider(self.widget)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(min)
        self.slider.setMaximum(max)
        self.slider.setValue(initial_value)
        self.slider.setSingleStep(increment)
        self.slider_label = QLabel(self.widget)
        self.slider_label.setText(str(initial_value))
        layout = QHBoxLayout(self.widget)
        layout.addWidget(self.slider)
        layout.addWidget(self.slider_label)
        self.slider.valueChanged.connect(lambda value: self.slider_label.setText(str(value)))

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.slider.value()


class BooleanInput(InputClass):

    def __init__(self, label, default=False):
        self.label = label
        self.widget = QCheckBox()
        self.widget.setChecked(default)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.isChecked()


## Dialog class implementation

class GenericDialog(QDialog):

    def __init__(self, title, input_list: list[InputClass], parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.verticalLayout = QVBoxLayout(self)
        self.groupBox = QGroupBox(title, self)
        self.formLayout = QFormLayout(self.groupBox)

        self.buttonBox = QDialogButtonBox(self)
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)

        self.buttonBox.accepted.connect(lambda: self.exit_dialog(True))
        self.buttonBox.rejected.connect(lambda: self.exit_dialog(False))

        self.input_list = input_list

        for position, input_obj in enumerate(input_list):
            label = QLabel(self.groupBox)
            label.setText(input_obj.get_label())
            self.formLayout.setWidget(position, QFormLayout.LabelRole, label)
            input_field = input_obj.get_widget()
            self.formLayout.setWidget(position, QFormLayout.FieldRole, input_field)

        self.verticalLayout.addWidget(self.groupBox)
        self.verticalLayout.addWidget(self.buttonBox)

        self.resize(self.verticalLayout.sizeHint())

        self.output = None
        self.accepted = False
        self.setModal(True)

    def exit_dialog(self, accepted):
        self.accepted = accepted
        self.close()

    def closeEvent(self, event):
        self.output = MixedDict()
        for input_obj in self.input_list:
            self.output[input_obj.get_label()] = input_obj.get_value()


## This is the main function that should be called
def show_dialog(title, input_list: list[InputClass], parent=None) -> (bool, MixedDict):
    dialog = GenericDialog(title, input_list, parent)
    dialog.exec()
    # this will stop until the dialog is closed
    if dialog.accepted is None:
        return None
    return dialog.accepted, dialog.output


## Testing
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    accepted, values = show_dialog("Test", [TextLineInput('Text input'),
                                            IntSpinInput('My Int', 10, -100, 100),
                                            FloatSpinInput('My Float'),
                                            IntSliderInput('My slider'),
                                            BooleanInput('Bool value')])

    # values can be accessed by key or by position
    print(values['My Int'])
    print(values[2])

    # they can be iterated like a list
    for v in values:
        print(v)
