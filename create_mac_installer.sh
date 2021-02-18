#!/bin/sh

git pull
cd dl
git checkout master
git pull
cd ..
pyinstaller dafne_mac.spec --noconfirm
cd dist
cp calc_transforms/calc_transforms dafne
zip -r dafne_mac.zip dafne 
