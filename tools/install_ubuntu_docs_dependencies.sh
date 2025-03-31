#!/bin/sh
#
# Prepare an Ubuntu CI machine for running 'tox -e docs'.  Assumes that Python is available.

set -e

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade tox

sudo apt-get update
sudo apt-get install -y graphviz pandoc doxygen
