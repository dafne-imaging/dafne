#!/bin/bash
#
# Copyright (c) 2022 Dafne-Imaging Team
#

rm -rf build
rm dist/dafne
VERSION=`python update_version.py | tail -n1`
echo $VERSION
pyinstaller dafne_linux.spec --noconfirm
cd dist
tar czf "dafne_linux_$VERSION.tar.gz" dafne