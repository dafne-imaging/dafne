#  Copyright (c) 2022 Dafne-Imaging Team
#  Copyright (c) 2022 Dafne-Imaging Team
# Hook file for pyinstaller

from PyInstaller.utils.hooks import collect_submodules

hiddenimports=collect_submodules('dafne') + \
              collect_submodules('skimage.segmentation')