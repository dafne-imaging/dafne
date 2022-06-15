#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2022 Dafne-Imaging Team
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

from ..config.config import show_config_dialog, save_config, load_config

import sys
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    load_config()
    accepted = show_config_dialog(None, True)
    if accepted:
        save_config()
        print('Configuration saved')
    else:
        print('Aborted')