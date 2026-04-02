#  Copyright (c) 2026 Dafne-Imaging Team
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

from __future__ import annotations

from flexidep.utils import (get_installed_packages_with_available_versions,
                             get_pypi_available_versions, is_conda_environment)
from flexidep.installers import install_package_version, uninstall_package
from flexidep.config import PackageManagers

from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QTableWidget, QTableWidgetItem,
                              QComboBox, QLabel, QLineEdit, QMessageBox,
                              QApplication, QStackedWidget, QHeaderView,
                              QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QThread
from PyQt5.QtGui import QColor

from . import GenericInputDialog
from .. import config

class _PackageLoaderThread(QThread):
    packages_loaded = pyqtSignal(object)
    loading_start = pyqtSignal(object)
    progress_signal = pyqtSignal(int, int)

    def run(self):
        def emit_progress(current, total):
            self.progress_signal.emit(current, total)
        packages = get_installed_packages_with_available_versions(callback=emit_progress)
        self.packages_loaded.emit(packages)


class _InstallThread(QThread):
    install_done = pyqtSignal(bool)

    def __init__(self, package_manager, package_name, version, parent=None):
        QThread.__init__(self, parent)
        self.package_manager = package_manager
        self.package_name = package_name
        self.version = version

    def run(self):
        result = install_package_version(self.package_manager, self.package_name, self.version)
        self.install_done.emit(result)


class _UninstallThread(QThread):
    uninstall_done = pyqtSignal(bool)

    def __init__(self, package_manager, package_name, parent=None):
        QThread.__init__(self, parent)
        self.package_manager = package_manager
        self.package_name = package_name

    def run(self):
        result = uninstall_package(self.package_manager, self.package_name)
        self.uninstall_done.emit(result)


class _VersionFetcherThread(QThread):
    versions_fetched = pyqtSignal(list)

    def __init__(self, package_name, parent=None):
        QThread.__init__(self, parent)
        self.package_name = package_name

    def run(self):
        versions = get_pypi_available_versions(self.package_name)
        self.versions_fetched.emit(versions)


class PackageManagerWindow(QDialog):

    busy_start = pyqtSignal(str)
    busy_end = pyqtSignal()
    progress = pyqtSignal(int, int, str)

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle("Package Manager")
        self.resize(800, 600)

        self.packages = {}
        self._update_queue = []
        self._is_busy = False

        self.busy_start.connect(lambda _: setattr(self, '_is_busy', True))
        self.busy_end.connect(lambda: setattr(self, '_is_busy', False))

        if is_conda_environment():
            self.package_manager = PackageManagers.conda
        else:
            self.package_manager = PackageManagers.pip

        self._setup_ui()

    def exec(self):
        self._load_packages()
        return QDialog.exec(self)

    def closeEvent(self, event):
        if self._is_busy:
            event.ignore()
        else:
            event.accept()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.stacked_widget = QStackedWidget(self)

        # --- Loading page ---
        loading_page = QWidget()
        loading_layout = QVBoxLayout(loading_page)
        loading_layout.setAlignment(Qt.AlignCenter)

        self.loading_text = QLabel("Loading packages, please wait...")
        self.loading_text.setAlignment(Qt.AlignCenter)

        loading_layout.addStretch()
        loading_layout.addWidget(self.loading_text)
        loading_layout.addStretch()

        # --- Main page ---
        main_page = QWidget()
        main_page_layout = QVBoxLayout(main_page)

        self.hide_uptodate_checkbox = QCheckBox("Show outdated packages only")
        self.hide_uptodate_checkbox.stateChanged.connect(self._apply_filter)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Package', 'Version'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)

        self.update_all_btn = QPushButton("Update All")
        self.update_all_btn.clicked.connect(self._update_all)

        # Install package section
        install_group = QGroupBox("Install Package")
        install_layout = QHBoxLayout(install_group)
        self.install_package_edit = QLineEdit()
        self.install_package_edit.setPlaceholderText("Package name...")
        self.install_package_edit.returnPressed.connect(self._install_package)
        install_btn = QPushButton("Install")
        install_btn.clicked.connect(self._install_package)
        install_layout.addWidget(self.install_package_edit)
        install_layout.addWidget(install_btn)

        main_page_layout.addWidget(self.hide_uptodate_checkbox)
        main_page_layout.addWidget(self.table)
        main_page_layout.addWidget(self.update_all_btn)
        main_page_layout.addWidget(install_group)

        self.stacked_widget.addWidget(loading_page)  # index 0
        self.stacked_widget.addWidget(main_page)      # index 1

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)

        main_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(close_btn)

    def _show_loading(self, message="Loading packages, please wait..."):
        self.loading_text.setText(message)
        self.stacked_widget.setCurrentIndex(0)
        QApplication.processEvents()

    def _show_main(self):
        self.stacked_widget.setCurrentIndex(1)

    def _load_packages(self):
        self.busy_start.emit("Loading packages")
        QApplication.processEvents()
        self._show_loading()
        self.loader_thread = _PackageLoaderThread(self)
        self.loader_thread.packages_loaded.connect(self._on_packages_loaded)
        self.loader_thread.progress_signal.connect(lambda i,j: self.progress.emit(i, j, 'Loading Packages'), Qt.QueuedConnection)
        self.loader_thread.start()

    @pyqtSlot(object)
    def _on_packages_loaded(self, packages):
        self.busy_end.emit()
        QApplication.processEvents()
        self.packages = packages
        self._populate_table()
        self._show_main()

    def _populate_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)

        for package_name, info in sorted(self.packages.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QTableWidgetItem(package_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            if not info['latest']:
                name_item.setForeground(QColor('red'))
            self.table.setItem(row, 0, name_item)

            combo = QComboBox()
            installed_str = str(info['installed_version'])
            max_versions = config.GlobalConfig['SHOW_MAX_PACKAGE_VER']
            for v in info['available_versions'][:max_versions]:
                combo.addItem(str(v))

            installed_index = combo.findText(installed_str)
            if installed_index >= 0:
                combo.setCurrentIndex(installed_index)

            combo.addItem("Uninstall")

            combo.setProperty('installed_version', installed_str)
            combo.setProperty('package_name', package_name)
            combo.activated[str].connect(self._on_version_activated)

            self.table.setCellWidget(row, 1, combo)

        self.table.setSortingEnabled(True)
        self._apply_filter()

    def _apply_filter(self):
        hide_uptodate = self.hide_uptodate_checkbox.isChecked()
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            is_latest = name_item.foreground().color() != QColor('red')
            self.table.setRowHidden(row, hide_uptodate and is_latest)

    @pyqtSlot(str)
    def _on_version_activated(self, selected_version):
        combo = self.sender()
        if combo is None:
            return

        installed_version = combo.property('installed_version')
        package_name = combo.property('package_name')

        if selected_version == installed_version:
            return

        if selected_version == "Uninstall":
            reply = QMessageBox.question(
                self,
                "Uninstall Package",
                f"Uninstall {package_name}?",
                QMessageBox.Yes | QMessageBox.No
            )
            really_uninstalled = False
            if reply == QMessageBox.Yes:
                reply = QMessageBox.warning(self, "Uninstall Package",
                                            f"Warning! Uninstalling packages might break Dafne and prevent it from starting! Do you really want to continue?",
                                            QMessageBox.Yes | QMessageBox.No,
                                            QMessageBox.No)
                if reply == QMessageBox.Yes:
                    really_uninstalled = True
                    self._do_uninstall(package_name)

            if not really_uninstalled:
                idx = combo.findText(installed_version)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            return

        reply = QMessageBox.question(
            self,
            "Install Package Version",
            f"Install {package_name}=={selected_version}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._do_install(package_name, selected_version, reload=True)
        else:
            # Revert combo to installed version
            idx = combo.findText(installed_version)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _do_install(self, package_name, version, reload=True):
        self.busy_start.emit(f"Installing {package_name}=={version}")
        QApplication.processEvents()
        self._reload_after_install = reload
        self._show_loading(f"Installing {package_name}=={version}...")
        self.install_thread = _InstallThread(
            self.package_manager, package_name, version, self
        )
        self.install_thread.install_done.connect(self._on_install_done)
        self.install_thread.start()

    @pyqtSlot(bool)
    def _on_install_done(self, success):
        self.busy_end.emit()
        QApplication.processEvents()
        if not success:
            QMessageBox.warning(self, "Installation Failed",
                                "The package installation failed. Check the log for details.")
            self._update_queue = []
            self._original_queue_length = 0
            self._load_packages()
            return

        if self._update_queue:
            self._install_next_in_queue()
        else:
            QMessageBox.information(self, "Done", "Package installed successfully.")
            self._load_packages()

    def _do_uninstall(self, package_name):
        self.busy_start.emit(f"Uninstalling {package_name}")
        QApplication.processEvents()
        self._show_loading(f"Uninstalling {package_name}...")
        self.uninstall_thread = _UninstallThread(self.package_manager, package_name, self)
        self.uninstall_thread.uninstall_done.connect(self._on_uninstall_done)
        self.uninstall_thread.start()

    @pyqtSlot(bool)
    def _on_uninstall_done(self, success):
        self.busy_end.emit()
        QApplication.processEvents()
        if not success:
            QMessageBox.warning(self, "Uninstall Failed",
                                "The package uninstallation failed. Check the log for details.")
        else:
            QMessageBox.information(self, "Done", "Package uninstalled successfully.")
        self._load_packages()

    # --- Update All ---

    def _update_all(self):
        outdated = [
            (name, str(info['available_versions'][0]))
            for name, info in self.packages.items()
            if not info['latest'] and info['available_versions']
        ]

        if not outdated:
            QMessageBox.information(self, "Up to Date", "All packages are already at the latest version.")
            return

        packages_list = "\n".join(f"  {name}  →  {ver}" for name, ver in outdated)
        reply = QMessageBox.question(
            self,
            "Update All Packages",
            f"Update {len(outdated)} package(s)?\n\n{packages_list}",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self._update_queue = outdated
        self._original_queue_length = len(self._update_queue)
        self._install_next_in_queue()

    def _install_next_in_queue(self):
        if not self._update_queue:
            QMessageBox.information(self, "Done", "All packages updated successfully.")
            self._original_queue_length = 0
            self._load_packages()
            return

        package_name, version = self._update_queue.pop(0)
        remaining = len(self._update_queue)
        self._show_loading(f"Updating {package_name}=={version}..."
                           + (f" ({remaining} remaining)" if remaining else ""))
        self.install_thread = _InstallThread(
            self.package_manager, package_name, version, self
        )
        self.install_thread.install_done.connect(self._on_update_one_done)
        self.install_thread.start()

    @pyqtSlot(bool)
    def _on_update_one_done(self, success):
        self.progress.emit(self._original_queue_length - len(self._update_queue), self._original_queue_length, 'Installing Packages')
        if not success:
            QMessageBox.warning(self, "Update Failed",
                                "A package update failed. Remaining updates cancelled.")
            self._update_queue = []
            self._original_queue_length = 0
            self._load_packages()
            return
        self._install_next_in_queue()

    # --- Install new package ---

    def _install_package(self):
        package_name = self.install_package_edit.text().strip()
        if not package_name:
            QMessageBox.warning(self, "Missing Input", "Please enter a package name.")
            return

        self._pending_package = package_name
        self._show_loading(f"Fetching available versions for '{package_name}'...")
        self.version_fetcher = _VersionFetcherThread(package_name, self)
        self.version_fetcher.versions_fetched.connect(self._on_versions_fetched)
        self.version_fetcher.start()

    @pyqtSlot(list)
    def _on_versions_fetched(self, versions):
        self._show_main()
        package_name = self._pending_package

        if not versions:
            QMessageBox.warning(self, "Package Not Found",
                                f"Could not find package '{package_name}' on PyPI.")
            return

        version_strings = [str(v) for v in versions]

        accepted, values = GenericInputDialog.show_dialog(
            f"Install {package_name}",
            [GenericInputDialog.OptionInput(
                "Version",
                version_strings,
                default=version_strings[0]
            )],
            parent=self,
            message=f"Select a version to install for package '{package_name}':"
        )

        if not accepted:
            return

        selected_version = values['Version']
        self.install_package_edit.clear()
        self._do_install(package_name, selected_version, reload=True)
