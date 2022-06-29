#!/bin/bash
#
# Copyright (c) 2022 Dafne-Imaging Team
#

VERSION=`python update_version.py | tail -n1`
echo $VERSION
pyinstaller dafne_linux.spec --noconfirm
cd dist
mv dafne "dafne_linux_$VERSION"