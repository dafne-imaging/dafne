#!/usr/bin/env python3
#  Copyright (c) 2022 Dafne-Imaging Team
import shutil
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'src')))

from dafne.config.version import VERSION

shutil.move('dafne_win.iss', 'dafne_win.iss.old')

with open('dafne_win.iss.old', 'r') as orig_file:
    with open('dafne_win.iss', 'w') as new_file:
        for line in orig_file:
            if line.startswith('#define MyAppVersion'):
                new_file.write(f'#define MyAppVersion "{VERSION}"\n')
            elif line.startswith('OutputBaseFilename='):
                new_file.write(f'OutputBaseFilename=dafne_windows_setup_{VERSION}\n')
            else:
                new_file.write(line)

shutil.move('dafne_mac.spec', 'dafne_mac.spec.old')
with open('dafne_mac.spec.old', 'r') as orig_file:
    with open('dafne_mac.spec', 'w') as new_file:
        for line in orig_file:
            if 'version=' in line:
                new_file.write(f"    version='{VERSION}')\n")
            else:
                new_file.write(line)

print(VERSION)