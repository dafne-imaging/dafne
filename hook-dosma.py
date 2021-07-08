# Hook file for pyinstaller

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports=collect_submodules('dosma')
datas = collect_data_files('dosma')