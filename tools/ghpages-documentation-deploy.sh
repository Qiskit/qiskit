#!/bin/bash

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Authors: Diego M. Rodriguez <diego.moreda@ibm.com>

# Script for generating the sphinx documentation and deploying it in the
# Github Pages repository. Please note that this depends on having the
# following variable set on travis containing a valid token with permissions
# for pushing into the Github Pages repository:
# GH_TOKEN

# Non-travis variables used by this script.
TARGET_REPOSITORY_USER="QISKit"
TARGET_REPOSITORY_NAME="qiskit.github.io"
TARGET_DOC_DIR="documentation"
TARGET_DOC_DIR_JA="documentation/ja"
SOURCE_DOC_DIR="doc/_build/html"
SOURCE_DOC_DIR_JA="doc/_build/ja/html"
SOURCE_DIR=`pwd`

# Build the documentation.
make doc

echo "Cloning the Github Pages repository ..."
cd ..
git clone https://github.com/$TARGET_REPOSITORY_USER/$TARGET_REPOSITORY_NAME.git
cd $TARGET_REPOSITORY_NAME

echo "Replacing $TARGET_DOC_DIR with the new contents ..."
git rm -rf $TARGET_DOC_DIR
git rm -rf $TARGET_DOC_DIR_JA
cp -r $SOURCE_DIR/$SOURCE_DOC_DIR $TARGET_DOC_DIR
cp -r $SOURCE_DIR/$SOURCE_DOC_DIR_JA $TARGET_DOC_DIR_JA
git add $TARGET_DOC_DIR

echo "Commiting and pushing changes ..."
git commit -m "Automated documentation update from SDK" -m "Commit: $TRAVIS_COMMIT" -m "Travis build: https://travis-ci.org/$TRAVIS_REPO_SLUG/builds/$TRAVIS_BUILD_ID"
git push --quiet https://$GH_TOKEN@github.com/$TARGET_REPOSITORY_USER/$TARGET_REPOSITORY_NAME.git > /dev/null 2>&1
