import os
import pickle
from ui import GenericInputDialog

CONFIG_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.pickle')

## Defaults
## This dictionary is used also for interface visualization
## It contains tuples where the first element is the default value, the second is the type (for interface visualization)
## Then optional limits (min/max/increment), and the final element is the visualization label.
## Only elements with a visualization label are shown in the normal interface.

defaults = {
    'SERVER_URL': ('http://www.dafne.network:5000/', 'string', None),
    'API_KEY': ('abc123', 'string', 'Personal server access key'),
    'MODEL_PROVIDER': ('Local', 'option', ['Local', 'Remote'], 'Location of the deep learning models'),
    'MODEL_PATH': ('models', 'string', None),
    'DO_INCREMENTAL_LEARNING':  (True, 'bool', None),
    'ROI_CIRCLE_SIZE':  (2, 'int', 1, 200, 1, None),
    'SIMPLIFIED_ROI_POINTS':  (20, 'int', 1, 200, 1, None),
    'SIMPLIFIED_ROI_SPACING':  (15, 'int', 1, 200, 1, None),
    'HIDE_ROIS_RIGHTCLICK':  (True, 'bool', 'Hide ROIs with right click'),
    'INTERPOLATION': ('spline36', 'option', ['none', 'nearest', 'bilinear', 'bicubic', 'spline36', 'catrom', 'lanczos'], 'Image interpolation method'),
    'COLORMAP': ('gray', 'option', ['gray', 'viridis', 'magma', 'gist_yarg', 'hsv'], 'Image colormap'),
    'ROI_COLOR':  ((1, 0, 0, 0.5), 'color', 'Color for active subROI'),  # red with 0.5 opacity,
    'ROI_SAME_COLOR':  ((1, 1, 0, 0.5), 'color', 'Color for inactive subROIs'),  # yellow with 0.5 opacity,
    'ROI_OTHER_COLOR':  ((0, 0, 1, 0.4), 'color', 'Color for inactive ROIs'),
    'MASK_LAYER_ALPHA':  (0.4, 'float_slider', 0.0, 1.0, 0.1, 'Alpha channel of masks'),
    'ROI_COLOR_WACOM':  ((1, 0, 0, 1), 'color', None),  # red with 1 opacity,
    'ROI_SAME_COLOR_WACOM':  ((1, 1, 0, 1), 'color', None),  # yellow with 1 opacity,
    'ROI_OTHER_COLOR_WACOM':  ((0, 0, 1, 0.8), 'color', None),
    'BRUSH_PAINT_COLOR':  ((1, 0, 0, 0.6), 'color', 'Brush color - paint'),
    'BRUSH_ERASE_COLOR':  ((0, 0, 1, 0.6), 'color', 'Brush color - erase'),
    'ROI_FILENAME':  ('rois.p', 'string', None),
    'AUTOSAVE_INTERVAL':  (30, 'int_slider', 1, 1000, 1, 'Interval for autosave (s)'),
    'HISTORY_LENGTH':  (20, 'int_slider', 1, 1000, 1, None),
    'SPLIT_LATERALITY': (True, 'bool', 'Separate L/R in autosegment'),
    'ENABLE_DATA_UPLOAD': (False, 'bool', None)
}

GlobalConfig = { k: v[0] for k,v in defaults.items() }

def load_config():
    global GlobalConfig
    try:
        with open(CONFIG_FILE, 'rb') as f:
            stored_config = pickle.load(f)
    except OSError:
        print("Warning no stored config file found!")
        stored_config = {}

    for k, v in stored_config.items():
        GlobalConfig[k] = v


def save_config():
    with open(CONFIG_FILE, 'wb') as f:
        pickle.dump(GlobalConfig, f)

def show_config_dialog(parent=None, show_all=False):
    option_list = []
    display_key_map = {}
    for key, value in defaults.items():
        current_value = GlobalConfig[key]
        display_string = value[-1]
        if not display_string:
            if not show_all:
                continue
            display_string = key
        type = value[1]
        display_key_map[display_string] = key # remember the mapping between the
        if type == 'bool':
            option = GenericInputDialog.BooleanInput(display_string, current_value)
        elif type == 'int' or type == 'int_spin':
            option = GenericInputDialog.IntSpinInput(display_string, current_value, value[2], value[3], value[4])
        elif type == 'int_slider':
            option = GenericInputDialog.IntSliderInput(display_string, current_value, value[2], value[3], value[4])
        elif type == 'float' or type == 'float_spin':
            option = GenericInputDialog.FloatSpinInput(display_string, current_value, value[2], value[3], value[4])
        elif type == 'float_slider':
            option = GenericInputDialog.FloatSliderInput(display_string, current_value, value[2], value[3], value[4])
        elif type == 'string':
            option = GenericInputDialog.TextLineInput(display_string, current_value)
        elif type == 'color':
            option = GenericInputDialog.ColorSpinInput(display_string, current_value)
        elif type == 'option': # at the moment not used
            option = GenericInputDialog.OptionInput(display_string, value[2], current_value)

        option_list.append(option)

    accepted, values = GenericInputDialog.show_dialog("Configuration", option_list, parent=parent, entries_per_page=11)
    if not accepted:
        return False

    # update the config
    for key, value in values.items():
        GlobalConfig[ display_key_map[key] ] = value

    return True