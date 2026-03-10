#!/bin/bash

DAFNE_VERSION="1.9a2"
PKG_NAME="dafne_mac_${DAFNE_VERSION}.pkg"
CODESIGN_IDENTITY="Francesco Santini"


NOTARYTOOL() {
  if [ -f /Library/Developer/CommandLineTools/usr/bin/notarytool ]
  then
    /Library/Developer/CommandLineTools/usr/bin/notarytool "$@"
  else 
    xcrun notarytool "$@"
  fi
}


# Create necessary directories
mkdir -p /tmp/pkg_root/tmp/resources
mkdir -p /tmp/scripts

# Copy your existing icon file to the resources directory
cp ./dafne.icns /tmp/pkg_root/tmp/resources

# Create the postinstall script
echo "#!/bin/bash" > /tmp/scripts/postinstall
echo "DAFNE_VERSION=$DAFNE_VERSION" >> /tmp/scripts/postinstall
cat >> /tmp/scripts/postinstall << 'EOF'
# Enable error handling
set -e

# Enable command logging
set -x

# Set variables
PYTHON_VERSION="3.11.9"
PYTHON_MAJOR_MINOR="3.11"
VENV_DIR="/usr/local/dafne"
PIP_PACKAGE="dafne"
APP_BUNDLE="/Applications/Dafne.app"
LOG_FILE="/tmp/dafne_install.log"

cpu_brand=$(sysctl -n machdep.cpu.brand_string)
    
# Check if it's an Apple Silicon processor
if [[ "$cpu_brand" == *"Apple"* ]]; then
    echo "This Mac has ARM64 capability (Apple Silicon)"
    ARCH_CMD="arch -arm64 "
else
    ARCH_CMD="arch -x86_64 "
fi

echo "Starting installation..." | tee -a "$LOG_FILE"

# Function to check if Python framework is installed
check_python_framework() {
    [ -d "/Library/Frameworks/Python.framework/Versions/${PYTHON_MAJOR_MINOR}" ]
}

# Function to check if Python installer package is already downloaded
check_python_installer() {
    [ -f "/tmp/python-${PYTHON_VERSION}-macos11.pkg" ]
}

# Function to download Python installer
download_python_installer() {
    curl -L "https://www.python.org/ftp/python/${PYTHON_VERSION}/python-${PYTHON_VERSION}-macos11.pkg" \
         -o "/tmp/python-${PYTHON_VERSION}-macos11.pkg"
}

# Function to install Python framework
install_python_framework() {
    installer -pkg "/tmp/python-${PYTHON_VERSION}-macos11.pkg" -target /
}

# Check and install Python if needed
if ! check_python_framework; then
    echo "Python ${PYTHON_VERSION} framework not found. Installing..." | tee -a "$LOG_FILE"
    
    if ! check_python_installer; then
        echo "Downloading Python installer..." | tee -a "$LOG_FILE"
        if ! download_python_installer; then
            echo "Failed to download Python installer" | tee -a "$LOG_FILE"
            exit 1
        fi
    fi
    
    echo "Installing Python framework..." | tee -a "$LOG_FILE"
    if ! install_python_framework; then
        echo "Failed to install Python framework" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    # Clean up installer
    rm -f "/tmp/python-${PYTHON_VERSION}-macos11.pkg"
else
    echo "Python ${PYTHON_VERSION} framework already installed" | tee -a "$LOG_FILE"
fi

# Verify Python installation
PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/${PYTHON_MAJOR_MINOR}/bin/python3"

if ! [ -x "$PYTHON_BIN" ]; then
    echo "Python installation verification failed" | tee -a "$LOG_FILE"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..." | tee -a "$LOG_FILE"
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing virtual environment..." | tee -a "$LOG_FILE"
    rm -rf "$VENV_DIR"
fi

if ! "$PYTHON_BIN" -m venv "$VENV_DIR" 2>&1 | tee -a "$LOG_FILE"; then
    echo "Failed to create virtual environment" | tee -a "$LOG_FILE"
    exit 1
fi

# Install pip package in virtual environment
echo "Installing $PIP_PACKAGE==${DAFNE_VERSION}..." | tee -a "$LOG_FILE"
if ! $ARCH_CMD "$VENV_DIR/bin/pip" install "$PIP_PACKAGE==${DAFNE_VERSION}" 2>&1 | tee -a "$LOG_FILE"; then
    echo "Failed to install pip package" | tee -a "$LOG_FILE"
    exit 1
fi

# Make site-packages writable by all
SITE_PACKAGES_DIR="$VENV_DIR/lib/"
echo "Making site-packages writable..." | tee -a "$LOG_FILE"
chmod -R a+w "$SITE_PACKAGES_DIR" 2>&1 | tee -a "$LOG_FILE"

BIN_PACKAGES_DIR="$VENV_DIR/bin"
echo "Making bin writable..." | tee -a "$LOG_FILE"
chmod -R a+w "$BIN_PACKAGES_DIR" 2>&1 | tee -a "$LOG_FILE"

# Create application bundle
echo "Creating application bundle..." | tee -a "$LOG_FILE"
mkdir -p "${APP_BUNDLE}/Contents/"{MacOS,Resources}

# Create Info.plist
cat > "${APP_BUNDLE}/Contents/Info.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>Dafne</string>
    <key>CFBundleIconFile</key>
    <string>dafne.icns</string>
    <key>CFBundleIdentifier</key>
    <string>network.dafne.dafne</string>
    <key>CFBundleName</key>
    <string>Dafne</string>
    <key>CFBundleVersion</key>
    <string>$DAFNE_VERSION</string>
</dict>
</plist>
PLIST_EOF

# Create launcher script
cat > "${APP_BUNDLE}/Contents/MacOS/Dafne" << 'LAUNCHER_EOF'
#!/bin/bash

# Set up environment for the virtual environment
VENV_DIR="/usr/local/dafne"

# Check if it's an Apple Silicon processor
cpu_brand=$(sysctl -n machdep.cpu.brand_string)
if [[ "$cpu_brand" == *"Apple"* ]]; then
    echo "This Mac has ARM64 capability (Apple Silicon)"
    ARCH_CMD="arch -arm64 "
else
    ARCH_CMD="arch -x86_64 "
fi

exec $ARCH_CMD "$VENV_DIR/bin/dafne" "$@" >/tmp/dafne_out.log 2>/tmp/dafne_err.log
LAUNCHER_EOF

# Make launcher executable
chmod a+x "${APP_BUNDLE}/Contents/MacOS/Dafne"

# Copy icon from package resources
cp "/tmp/resources/dafne.icns" "${APP_BUNDLE}/Contents/Resources/dafne.icns"

# Set proper permissions and attributes
chmod -R 777 "${APP_BUNDLE}"
#xattr -cr "${APP_BUNDLE}"

# Touch the app bundle to refresh Finder
touch "${APP_BUNDLE}"

echo "Installation completed successfully!" | tee -a "$LOG_FILE"
echo "Log file available at: $LOG_FILE" | tee -a "$LOG_FILE"

# Print installation details for verification
echo "Virtual environment location: $VENV_DIR" | tee -a "$LOG_FILE"
echo "Python location: $VENV_DIR/bin/python" | tee -a "$LOG_FILE"
echo "Application bundle: $APP_BUNDLE" | tee -a "$LOG_FILE"
EOF

# Make the postinstall script executable
chmod +x /tmp/scripts/postinstall

# Create the preinstall script
cat > /tmp/scripts/preinstall << 'EOF'
#!/bin/bash

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    echo "This package is only for macOS"
    exit 1
fi

# Check if we have enough disk space (let's say 1GB to be safe)
available_space=$(df -k / | tail -1 | awk '{print $4}')
if [ "$available_space" -lt 1048576 ]; then  # 1GB in KB
    echo "Not enough disk space. Need at least 1GB"
    exit 1
fi

exit 0
EOF

# Make the preinstall script executable
chmod +x /tmp/scripts/preinstall

# Create the license file
cat > /tmp/LICENSE.txt << 'EOF'
End User License Agreement

Copyright (C) 2024 Dafne Imaging Team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

By installing this software, you agree to the terms and conditions outlined above.
EOF

# Create distribution XML
cat > /tmp/distribution.xml << 'EOF'
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>Dafne</title>
    <organization>network.dafne</organization>
    <domains enable_localSystem="true"/>
    <options customize="never" require-scripts="true" rootVolumeOnly="true"/>
    <license file="LICENSE.txt"/>
    <choices-outline>
        <line choice="network.dafne.dafne"/>
    </choices-outline>
    <choice id="network.dafne.dafne" visible="false">
        <pkg-ref id="network.dafne.dafne"/>
    </choice>
    <pkg-ref id="network.dafne.dafne" version="1.0">dafne-component.pkg</pkg-ref>
</installer-gui-script>
EOF

codesign --force --sign "$CODESIGN_IDENTITY" --timestamp /tmp/scripts/preinstall
codesign --force --sign "$CODESIGN_IDENTITY" --timestamp /tmp/scripts/postinstall


# Build the component package
echo "Building component package..."
pkgbuild --root /tmp/pkg_root \
         --scripts /tmp/scripts \
         --identifier network.dafne.dafne \
         --version $DAFNE_VERSION \
         --install-location "/" \
         --sign "$CODESIGN_IDENTITY" \
         dafne-component.pkg 

codesign --force --sign "$CODESIGN_IDENTITY" --timestamp dafne-component.pkg 

# Build the final product package with license
echo "Building final package with license..."
productbuild --distribution /tmp/distribution.xml \
             --resources /tmp \
             --package-path . \
             --sign "$CODESIGN_IDENTITY" \
             $PKG_NAME

# Clean up
rm -rf /tmp/pkg_root /tmp/scripts /tmp/LICENSE.txt /tmp/distribution.xml dafne-component.pkg

echo "Signing and notarizing..."

codesign --force --sign "$CODESIGN_IDENTITY" --timestamp $PKG_NAME
NOTARYTOOL submit $PKG_NAME --wait --keychain-profile "AC_PASSWORD"
xcrun stapler staple $PKG_NAME

echo "Package built. To install with verbose output, run:"
echo "sudo installer -verbose -dumplog -pkg dafne.pkg -target /"
