#!/bin/sh

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
codesign --deep -s "Francesco" dafne.app
echo "Creating DMG"
hdiutil create /tmp/dafne.dmg -ov -volname "Dafne" -fs HFS+ -srcfolder "dafne.app"
hdiutil convert /tmp/dafne.dmg -format UDZO -o "dafne_$VERSION.dmg"
rm /tmp/dafne.dmg
#cp calc_transforms/calc_transforms dafne
#zip -r dafne_mac_$VERSION.zip dafne
