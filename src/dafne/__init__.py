#  Copyright (c) 2022 Dafne-Imaging Team

from .config.version import VERSION
__version__ = VERSION

from . import resources

import sys
import flexidep

assert sys.version_info.major == 3, "This software is only compatible with Python 3.x"

if sys.version_info.minor < 10:
    import importlib_resources as pkg_resources
else:
    import importlib.resources as pkg_resources

# install the required resources
if not flexidep.is_frozen():
    with pkg_resources.files(resources).joinpath('runtime_dependencies.cfg').open() as f:
        dm = flexidep.DependencyManager(config_file=f)
    dm.install_interactive()
