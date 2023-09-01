#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Script for pushing the translatable messages to poBranch.


# Variables used by this script.
# From github actions the docs/locale/en directory from the sphinx build
# gets downloaded from the github actions artifacts as deploy directory
TARGET_REPOSITORY="git@github.com:qiskit-community/qiskit-translations.git"
SOURCE_DIR=`pwd`
SOURCE_LANG='en'

SOURCE_REPOSITORY="git@github.com:Qiskit/qiskit.git"
TARGET_BRANCH_PO="main"
DOC_DIR_PO="deploy"
TARGET_DOCS_DIR_PO="docs/locale"

echo "show current dir: "
pwd

echo "Setup ssh keys"
pwd
set -e
# Add poBranch push key to ssh-agent
openssl enc -aes-256-cbc -d -in ../tools/github_poBranch_update_key.enc -out github_poBranch_deploy_key -K $encrypted_deploy_po_branch_key -iv $encrypted_deploy_po_branch_iv
chmod 600 github_poBranch_deploy_key
eval $(ssh-agent -s)
ssh-add github_poBranch_deploy_key

# Clone to the working repository for .po and pot files
popd
pwd
echo "git clone for working repo"
git clone --depth 1 $TARGET_REPOSITORY temp --single-branch --branch $TARGET_BRANCH_PO
pushd temp

git config user.name "Qiskit Autodeploy"
git config user.email "qiskit@qiskit.org"

echo "git rm -rf for the translation po files"
git rm -rf --ignore-unmatch $TARGET_DOC_DIR_PO/$SOURCE_LANG/LC_MESSAGES/*.po \
    $TARGET_DOCS_DIR_PO/$SOURCE_LANG/LC_MESSAGES/api \
    $TARGET_DOCS_DIR_PO/$SOURCE_LANG/LC_MESSAGES/apidoc \
    $TARGET_DOCS_DIR_PO/$SOURCE_LANG/LC_MESSAGES/apidoc_legacy \
    $TARGET_DOCS_DIR_PO/$SOURCE_LANG/LC_MESSAGES/theme \
    $TARGET_DOCS_DIR_PO/$SOURCE_LANG/LC_MESSAGES/_*

# Remove api/ and apidoc/ to avoid confusion while translating
rm -rf $SOURCE_DIR/$DOC_DIR_PO/LC_MESSAGES/api/ \
    $SOURCE_DIR/$DOC_DIR_PO/LC_MESSAGES/apidoc/ \
    $SOURCE_DIR/$DOC_DIR_PO/LC_MESSAGES/apidoc_legacy/ \
    $SOURCE_DIR/$DOC_DIR_PO/LC_MESSAGES/stubs/ \
    $SOURCE_DIR/$DOC_DIR_PO/LC_MESSAGES/theme/

# Copy the new rendered files and add them to the commit.
echo "copy directory"
cp -r $SOURCE_DIR/$DOC_DIR_PO/. $TARGET_DOCS_DIR_PO/$SOURCE_LANG
cp $SOURCE_DIR/qiskit_pkg/setup.py .
cp $SOURCE_DIR/requirements-dev.txt .
# Append optional requirements to the dev list as some are needed for
# docs builds
cat $SOURCE_DIR/requirements-optionals.txt >> requirements-dev.txt
cp $SOURCE_DIR/constraints.txt .

echo "add to po files to target dir"
git add docs/
git add setup.py
git add requirements-dev.txt constraints.txt

# Commit and push the changes.
git commit -m "Automated documentation update to add .po files from qiskit" -m "skip ci" -m "Commit: $GITHUB_SHA" -m "Github Actions Run: $GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"
echo "git push"
git push --quiet origin $TARGET_BRANCH_PO
echo "********** End of pushing po to working repo! *************"
popd
