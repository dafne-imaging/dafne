#!/bin/sh

git pull
cd dl
git checkout master
git pull
cd ..
VERSION=`python update_version.py | tail -n 1`
pyinstaller dafne_mac.spec --noconfirm
cd dist
cp calc_transforms/calc_transforms dafne
zip -r dafne_mac_$VERSION.zip dafne
