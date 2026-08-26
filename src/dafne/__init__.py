#  Copyright (c) 2022 Dafne-Imaging Team

from . import resources

import sys
import flexidep

assert sys.version_info.major == 3, "This software is only compatible with Python 3.x"

if sys.version_info.minor < 10:
    import importlib_resources as pkg_resources
else:
    import importlib.resources as pkg_resources

if sys.platform == 'linux':
    print("Linux detected. Uninstalling triton to avoid tensorflow crash on Linux when using the GPU.")
    # Fix for Linux - triton makes tensorflow crash on Linux when using the GPU. This is a known issue with TensorFlow and Triton.
    flexidep.uninstall_package(flexidep.PackageManagers.pip, 'triton')

# install the required resources
if not flexidep.is_frozen():
    with pkg_resources.files(resources).joinpath('runtime_dependencies.cfg').open() as f:
        dm = flexidep.DependencyManager(config_file=f)
    dm.install_interactive()

# the following need to be imported before .config initializes the Qt environment, otherwise under windows there are DLL conflicts
import tensorflow as tf
import torch

from .config.version import VERSION
__version__ = VERSION
