#!/usr/bin/env python3
#  Copyright (c) 2022 Dafne-Imaging Team
import shutil

from config import VERSION

shutil.move('dafne_win.iss', 'dafne_win.iss.old')

with open('dafne_win.iss.old', 'r') as orig_file:
    with open('dafne_win.iss', 'w') as new_file:
        for line in orig_file:
            if line.startswith('#define MyAppVersion'):
                new_file.write(f'#define MyAppVersion "{VERSION}"\n')
            else:
                new_file.write(line)

print(VERSION)