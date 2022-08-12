#  Copyright (c) 2022 Dafne-Imaging Team
# generic stub for executable scripts residing in bin.
# This code will execute the main function of a script residing in bin having the same name as the script.
# The main function must be named "main" and must be in the global scope.

import os
import sys
import importlib

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

this_script = os.path.basename(__file__)
this_script_name = os.path.splitext(this_script)[0]

import_module_name = f'dafne.bin.{this_script_name}'

i = importlib.import_module(import_module_name)

if __name__ == '__main__':
    i.main()