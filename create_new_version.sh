#!/bin/bash
#
# Copyright (c) 2022 Dafne-Imaging Team
#

new_version=$1
version_file=src/dafne/config/version.py
current_version=`tail -n 1 $version_file | sed "s/VERSION='\(.*\)'/\1/"`

if [ "$new_version" == "" ]; then
    echo "Usage: $0 <new_version>"
    echo "Current version: $current_version"
    exit 1
fi

echo "#  Copyright (c) 2022 Dafne-Imaging Team" > $version_file
echo "# This file was auto-generated. Any changes might be overwritten" >> $version_file
echo "VERSION='$new_version'" >> $version_file

rm dist/*
python -m build --sdist --wheel
python -m twine upload dist/*
