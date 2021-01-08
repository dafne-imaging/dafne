from __future__ import annotations

from typing import Any, Union
from collections import OrderedDict
from PyQt5.QtWidgets import QDialog, QWidget, QSpinBox, QDoubleSpinBox, QLineEdit, QFormLayout, QVBoxLayout, \
    QHBoxLayout, QGroupBox, QDialogButtonBox, QLabel, QApplication, QSlider, QCheckBox, QComboBox
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

    def __init__(self, label:str, initial_value:str =''):
        self.label = label
        self.line_edit = QLineEdit()
        self.line_edit.setText(initial_value)

    def get_widget(self) -> QWidget:
        return self.line_edit

    def get_label(self) -> str:
        return self.label

    def get_value(self) -> Any:
        return self.line_edit.text()


class IntSpinInput(InputClass):

    def __init__(self, label:str, initial_value:int=0, min:int=0, max:int=99, increment:int=1):
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

    def __init__(self, label:str, initial_value:float=0.0, min:float=0.0, max:float=99.0, increment:float=1.0):
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

    def __init__(self, label:str, initial_value:int=0, min:int=0, max:int=99, increment:int=1):
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

    def __init__(self, label: str, default:bool = False):
        self.label = label
        self.widget = QCheckBox()
        self.widget.setChecked(default)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.widget.isChecked()


class OptionInput(InputClass):

    def __init__(self, label:str, value_list:list[Union[str, tuple[str, Any]]], default=None):
        self.label = label
        self.widget = QComboBox()

        self.output_list = []
        default_set = False

        for index, v in enumerate(value_list):
            if type(v) == str:
                self.widget.addItem(v)
                self.output_list.append(v)
                if default is not None:
                    if default == v:
                        default_set = True
                        self.widget.setCurrentIndex(index)
            else: # it is a tuple, specified by the type hint in the constructor
                self.widget.addItem(v[0])
                self.output_list.append(v[1])
                if default is not None:
                    if default in v:
                        default_set = True
                        self.widget.setCurrentIndex(index)

        if not default_set and type(default) == int:
            self.widget.setCurrentIndex(default)

    def get_label(self) -> str:
        return self.label

    def get_widget(self) -> QWidget:
        return self.widget

    def get_value(self) -> Any:
        return self.output_list[self.widget.currentIndex()]


## Dialog class implementation

class GenericDialog(QDialog):

    def __init__(self, title: str, input_list: list[InputClass], parent=None):
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
def show_dialog(title: str, input_list: list[InputClass], parent=None) -> (bool, MixedDict):
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
                                            BooleanInput('Bool value'),
                                            OptionInput('My string options', [
                                                'option 1',
                                                'option 2',
                                                'option 3'
                                            ], 'option 3'),
                                            OptionInput('My int options', [
                                                            ('option 1', 1.1),
                                                            ('option 2', 2.2),
                                                            ('option 3', 3.3)
                                                        ], 2.2)
                                            ])
    # Note: for option inputs, the value list can be a list of strings, and then the output is the string itself, or a
    # list of tuples, where the first element is a string (the label) and the second is the returned value (any).
    # The default value for options can be the label string, the default returned value, or an integer index


    # returned values can be accessed by key or by position
    print(values['My Int'])
    print(values[2])

    # they can be iterated like a list
    for v in values:
        print(v)
