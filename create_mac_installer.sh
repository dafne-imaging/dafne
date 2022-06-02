#!/bin/zsh

git pull
cd dl
git checkout master
git pull
cd ..
VERSION=`python update_version.py | tail -n 1`
echo $VERSION
pyinstaller dafne_mac.spec --noconfirm
cd dist
echo "Signing code"
codesign --deep -s "Francesco" Dafne.app
echo "Creating DMG"
create-dmg --volname "Dafne" --volicon Dafne.app/Contents/Resources/dafne_icon.icns \
	 --eula Dafne.app/Contents/Resources/LICENSE --background ../mac_installer_bg.png \
	 --window-size 420 220 --icon-size 64 --icon dafne.app 46 31 \
	 --app-drop-link 236 90 "dafne_$VERSION.dmg" dafne.app
#rm /tmp/dafne.dmg
#cp calc_transforms/calc_transforms dafne
#zip -r dafne_mac_$VERSION.zip dafne
