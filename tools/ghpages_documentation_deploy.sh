#!/bin/bash

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# Authors: Diego M. Rodriguez <diego.moreda@ibm.com>

# Script for generating the sphinx documentation and deploying it in the
# Github Pages repository. Please note that this depends on having the
# following variable set on travis containing a valid token with permissions
# for pushing into the Github Pages repository:
# GH_TOKEN

# Non-travis variables used by this script.
TARGET_REPOSITORY_USER="Qiskit"
TARGET_REPOSITORY_NAME="qiskit.github.io"
TARGET_DOC_DIR="documentation"
TARGET_DOC_DIR_DE="documentation/de"
TARGET_DOC_DIR_JA="documentation/ja"
SOURCE_DOC_DIR="doc/_build/html"
SOURCE_DOC_DIR_DE="doc/_build/de/html"
SOURCE_DOC_DIR_JA="doc/_build/ja/html"
SOURCE_DIR=`pwd`

# Build the documentation.
make -C out doc

echo "Cloning the Github Pages repository ..."
cd ..
git clone https://github.com/$TARGET_REPOSITORY_USER/$TARGET_REPOSITORY_NAME.git
cd $TARGET_REPOSITORY_NAME

echo "Replacing $TARGET_DOC_DIR with the new contents ..."
git rm -rf $TARGET_DOC_DIR/_* $TARGET_DOC_DIR/de/* $TARGET_DOC_DIR/ja/* $TARGET_DOC_DIR/*.html
mkdir -p $TARGET_DOC_DIR $TARGET_DOC_DIR_DE $TARGET_DOC_DIR_JA
cp -r $SOURCE_DIR/$SOURCE_DOC_DIR/* $TARGET_DOC_DIR/
cp -r $SOURCE_DIR/$SOURCE_DOC_DIR_DE/* $TARGET_DOC_DIR_DE/
cp -r $SOURCE_DIR/$SOURCE_DOC_DIR_JA/* $TARGET_DOC_DIR_JA/
git add $TARGET_DOC_DIR

echo "Commiting and pushing changes ..."
git commit -m "Automated documentation update from SDK" -m "Commit: $TRAVIS_COMMIT" -m "Travis build: https://travis-ci.org/$TRAVIS_REPO_SLUG/builds/$TRAVIS_BUILD_ID"
git push --quiet https://$GH_TOKEN@github.com/$TARGET_REPOSITORY_USER/$TARGET_REPOSITORY_NAME.git > /dev/null 2>&1
