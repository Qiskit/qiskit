#!/bin/sh
#
# Prepare an Ubuntu CI machine for running 'tox -e docs'.  Assumes that Python is available.

if [[ -n "$1" ]]
then
    set -e

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --upgrade tox

    sudo apt-get update
    sudo apt-get install -y graphviz pandoc 

    # This command fetches the latest release of doxygen and its linux binaries
    curl -H "Authorization: token $$1" \
        https://api.github.com/repos/doxygen/doxygen/releases/latest | \
        jq '.["assets"][2]["browser_download_url"]' | \
        xargs -I {} wget {} -O ./doxygen.tar.gz

    # The following commands install the binaries for doxygen
    tar -zxvf ./doxygen.tar.gz
    cd ./doxygen
    ./configure
    make
    sudo make install
else
    echo "No github token provided"
    exit 1
fi
