import subprocess
import sys
import os


def open_folder(folder_path):
    if not os.path.isdir(folder_path):
        folder_path = os.path.dirname(folder_path)

    if sys.platform == 'win32':
        proc_name = 'explorer'
    elif sys.platform == 'darwin':
        proc_name = 'open'
    elif sys.platform.startswith('linux'):
        proc_name = 'xdg-open'
    else:
        raise NotImplementedError('Unsupported platform')

    try:
        subprocess.run([proc_name, folder_path])
    except Exception as e:
        print(f'Error while opening folder: {e}')
