# Hook file for pyinstaller

from PyInstaller.utils.hooks import collect_submodules

hiddenimports=collect_submodules('dl') + \
              collect_submodules('skimage.filters')