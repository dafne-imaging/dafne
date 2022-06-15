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

import os
import pickle

from ..ui import GenericInputDialog
from appdirs import AppDirs

APP_NAME='Dafne'
APP_DEVELOPER='Dafne-imaging'

APP_DIR = os.path.abspath(os.getcwd())

DEBUG_ENVIRONMENT = os.path.isfile(os.path.join(APP_DIR, 'use_local_directories')) # this changes all the directory locations.
# True means local directories, False means system standard

if DEBUG_ENVIRONMENT:

    print('Using local directories for configuration')

    class AppDirTemp:
        pass

    app_dirs = AppDirTemp()
    app_dirs.user_config_dir = os.path.join(APP_DIR, 'config')
    app_dirs.user_data_dir = APP_DIR
    app_dirs.user_cache_dir = APP_DIR
    app_dirs.user_log_dir = APP_DIR
else:
    print('Using system directories for configuration')
    app_dirs = AppDirs(APP_NAME, APP_DEVELOPER)

CONFIG_DIR = app_dirs.user_config_dir
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.pickle')

## Defaults
## This dictionary is used also for interface visualization
## It contains tuples where the first element is the default value, the second is the type (for interface visualization)
## Then optional limits (min/max/increment), and the final element is the visualization label.
## Only elements with a visualization label are shown in the normal interface.

defaults = {
    'SERVER_URL': ('https://www.dafne.network/api/', 'string', None),
    'USE_CLASSIFIER': (False, 'bool', None),
    'MODEL_PROVIDER': ('Local', 'option', ['Local', 'Remote'], 'Model location'),
    'API_KEY': ('', 'string', 'Personal server access key'),
    'SPLIT_LATERALITY': (True, 'bool', 'Separate L/R in autosegment'),
    'FORCE_MODEL_DOWNLOAD': (False, 'bool', 'Force download of models from server'),
    'IL_MIN_SLICES': (5, 'int_spin', 1, 50, 1, None),
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
    'FORCE_LOCAL_DATA_UPLOAD': (False, 'bool', None),
    'DELETE_OLD_MODELS': (True, 'bool', None),
    'ECHO_OUTPUT': (False, 'bool', None),
    'ADVANCED_CONFIG': (False, 'bool', 'Show advanced configuration'),
}

# This part of the config is only stored here and can be changed by new software releases
static_config = {
    'ENABLE_DATA_UPLOAD': True,
    'MODEL_PATH': os.path.join(app_dirs.user_data_dir, 'models'),
    'TEMP_UPLOAD_DIR': os.path.join(app_dirs.user_cache_dir, 'upload_temp'),
    'TEMP_DIR': os.path.join(app_dirs.user_cache_dir, 'temp'),
    'ENABLE_NIFTI': True,
    'OUTPUT_LOG_FILE': os.path.join(app_dirs.user_log_dir, 'dafne_output.log'),
    'ERROR_LOG_FILE': os.path.join(app_dirs.user_log_dir, 'dafne_error.log'),
    'REDIRECT_OUTPUT': True # redirect stdout/stderr to logfiles
}

## Initialization

GlobalConfig = { k: v[0] for k,v in defaults.items() }
for k, v in static_config.items():
    GlobalConfig[k] = v

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(app_dirs.user_log_dir, exist_ok=True)
os.makedirs(GlobalConfig['MODEL_PATH'], exist_ok=True)
os.makedirs(GlobalConfig['TEMP_UPLOAD_DIR'], exist_ok=True)
os.makedirs(GlobalConfig['TEMP_DIR'], exist_ok=True)


def load_config():
    # load defaults
    for k, v in defaults.items():
        GlobalConfig[k] = v[0]

    try:
        with open(CONFIG_FILE, 'rb') as f:
            stored_config = pickle.load(f)
    except OSError:
        print("Warning no stored config file found!")
        stored_config = {}

    # overwrite with stored configuration
    for k, v in stored_config.items():
        GlobalConfig[k] = v

    # static config always supersedes stored config
    for k, v in static_config.items():
        GlobalConfig[k] = v


def save_config():
    with open(CONFIG_FILE, 'wb') as f:
        pickle.dump(GlobalConfig, f)


def delete_config():
    os.remove(CONFIG_FILE)
    load_config()


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