#  Copyright (c) 2025 Dafne-Imaging Team
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

"""Set the name and icon shown by the desktop environment.

Without this, dafne started from a console script shows up as "python".
"""

import sys

from .resource_utils import get_resource_path

APP_NAME = 'Dafne'
APP_ICON = 'dafne_logo.png'

# must stay alive: if garbage collected, the C++ objects it owns (e.g. the
# global QThreadPool) are destroyed too
_qapp = None


def _set_macos_bundle_name(app_name):
    """Patch the process' Info.plist. Qt reads it when the QApplication is created."""
    try:
        from Foundation import NSBundle
    except ImportError:
        return

    bundle = NSBundle.mainBundle()
    if bundle is None:
        return

    info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
    if info is None:
        return

    # mutable for a non-bundled process; a no-op inside a real .app bundle
    try:
        info['CFBundleName'] = app_name
        info['CFBundleDisplayName'] = app_name
    except (TypeError, ValueError):
        pass


def _set_macos_dock_icon(icon_path):
    """Set the Dock icon of the running process."""
    try:
        from AppKit import NSApplication, NSImage
    except ImportError:
        return

    image = NSImage.alloc().initByReferencingFile_(icon_path)
    if image is None or not image.isValid():
        return

    NSApplication.sharedApplication().setApplicationIconImage_(image)


def setup_app_identity():
    """Create the QApplication with the correct name and icon.

    Must be called before any other Qt object is instantiated.
    """
    global _qapp

    if _qapp is not None:
        return _qapp

    if sys.platform == 'darwin':
        _set_macos_bundle_name(APP_NAME)

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QIcon

    # matplotlib only sets these when it creates the QApplication itself
    for attribute in ('AA_EnableHighDpiScaling', 'AA_UseHighDpiPixmaps'):
        if hasattr(Qt, attribute):
            QApplication.setAttribute(getattr(Qt, attribute))
    if hasattr(Qt, 'HighDpiScaleFactorRoundingPolicy'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    QApplication.setApplicationName(APP_NAME)
    QApplication.setApplicationDisplayName(APP_NAME)
    QApplication.setOrganizationName(APP_NAME)
    # matches the window to its .desktop entry on Wayland/X11
    QApplication.setDesktopFileName(APP_NAME.lower())

    # argv[0] only: dafne's own options are not meant for Qt
    _qapp = QApplication.instance() or QApplication(sys.argv[:1])
    _qapp.setQuitOnLastWindowClosed(True)

    with get_resource_path(APP_ICON) as icon_path:
        QApplication.setWindowIcon(QIcon(icon_path))
        if sys.platform == 'darwin':
            _set_macos_dock_icon(icon_path)

    return _qapp
