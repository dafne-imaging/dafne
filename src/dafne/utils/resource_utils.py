#  Copyright (c) 2022 Dafne-Imaging Team
import os
import sys

assert sys.version_info.major == 3, "This software is only compatible with Python 3.x"

if sys.version_info.minor < 10:
    import importlib_resources as pkg_resources
else:
    import importlib.resources as pkg_resources

from contextlib import contextmanager

from .. import resources

@contextmanager
def get_resource_path(resource_name):
    if getattr(sys, '_MEIPASS', None):
        yield os.path.join(sys._MEIPASS, 'resources', resource_name)  # PyInstaller support. If _MEIPASS is set, we are in a Pyinstaller environment
    else:
        with pkg_resources.as_file(pkg_resources.files(resources).joinpath(resource_name)) as resource:
            yield str(resource)
