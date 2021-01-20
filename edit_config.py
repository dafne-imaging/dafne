#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from config import show_config_dialog, save_config, load_config

import sys
from PyQt5.QtWidgets import QApplication

app = QApplication(sys.argv)
app.setQuitOnLastWindowClosed(True)

load_config()
accepted = show_config_dialog(None, True)
if accepted:
    save_config()
    print('Configuration saved')
else:
    print('Aborted')