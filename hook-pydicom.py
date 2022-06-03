#  Copyright (c) 2022 Dafne-Imaging Team
# Hook file for pyinstaller

from PyInstaller.utils.hooks import collect_submodules

hiddenimports=collect_submodules('pydicom')