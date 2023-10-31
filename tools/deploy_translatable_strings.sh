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

# DO NOT `set -x`.  We have to pass secrets to `openssl` on the command line,
# and we don't want them appearing in the log.  This script instead manually
# 'echo's its status at various points.
set -eu -o pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: deploy_translatable_string.sh /path/to/translations/artifact" >&2
    exit 1
fi

# Variables used by this script.
# From github actions the docs/locale/en directory from the sphinx build
# gets downloaded from the github actions artifacts as deploy directory

TARGET_REPO="git@github.com:qiskit-community/qiskit-translations.git"
TARGET_REPO_BRANCH="main"

SOURCE_TOOLS_DIR="$(dirname "$(realpath "$0")")"
# Absolute paths to the git repository roots for the source repository (which
# this file lives in) and where we're going to clone the target repository.
SOURCE_REPO_ROOT="$(dirname "$SOURCE_TOOLS_DIR")"
TARGET_REPO_ROOT="${SOURCE_REPO_ROOT}/_qiskit_translations"

SOURCE_LANG="en"
# Absolute paths to the source and target directories for the translations
# files.  CI should feed the source in for us - it depends on the particulars of
# how it was built in a previous job.  The target is under our control.
SOURCE_PO_DIR="$1"
TARGET_PO_DIR="${TARGET_REPO_ROOT}/docs/locale/${SOURCE_LANG}"

# Add the SSH key needed to verify ourselves when pushing to the target remote.
echo "+ setup ssh keys"
eval "$(ssh-agent -s)"
openssl enc -aes-256-cbc -d \
    -in "${SOURCE_REPO_ROOT}/tools/github_poBranch_update_key.enc" \
    -K "$encrypted_deploy_po_branch_key" \
    -iv "$encrypted_deploy_po_branch_iv" \
    | ssh-add -

# Clone the target repository so we can build our commit in it.
echo "+ 'git clone' translations target repository"
git clone --depth 1 "$TARGET_REPO" "$TARGET_REPO_ROOT" --single-branch --branch "$TARGET_REPO_BRANCH"
pushd "$TARGET_REPO_ROOT"

echo "+ setup git configuration for commit"
git config user.name "Qiskit Autodeploy"
git config user.email "qiskit@qiskit.org"

echo "+ 'git rm' current translations files"
# Remove existing versions of the translations, to ensure deletions in the source repository are recognised.
git rm -rf --ignore-unmatch \
    "$TARGET_PO_DIR/LC_MESSAGES/"*.po \
    "$TARGET_PO_DIR/LC_MESSAGES/api" \
    "$TARGET_PO_DIR/LC_MESSAGES/apidoc" \
    "$TARGET_PO_DIR/LC_MESSAGES/apidoc_legacy" \
    "$TARGET_PO_DIR/LC_MESSAGES/theme" \
    "$TARGET_PO_DIR/LC_MESSAGES/"_*

echo "+ 'rm' unwanted files from source documentation"
# Remove files from the deployment that we don't want translating.
rm -rf \
    "$SOURCE_PO_DIR/LC_MESSAGES/api/" \
    "$SOURCE_PO_DIR/LC_MESSAGES/apidoc/" \
    "$SOURCE_PO_DIR/LC_MESSAGES/apidoc_legacy/" \
    "$SOURCE_PO_DIR/LC_MESSAGES/stubs/" \
    "$SOURCE_PO_DIR/LC_MESSAGES/theme/"

echo "+ 'cp' wanted files from source to target"
# Copy the new rendered files and add them to the commit.
cp -r "$SOURCE_PO_DIR/." "$TARGET_PO_DIR"
# Copy files necessary to build the Qiskit metapackage.
cp "$SOURCE_REPO_ROOT/qiskit_pkg/setup.py" "${TARGET_REPO_ROOT}"
cat "$SOURCE_REPO_ROOT/requirements-dev.txt" "$SOURCE_REPO_ROOT/requirements-optional.txt" \
    > "${TARGET_REPO_ROOT}/requirements-dev.txt"
cp "$SOURCE_REPO_ROOT/constraints.txt" "${TARGET_REPO_ROOT}"
# Add commit hash to be able to run the build with the commit hash before the actual release
echo $GITHUB_SHA > "${TARGET_REPO_ROOT}/qiskit-commit-hash"

echo "+ 'git add' files to target commit"
git add docs/ setup.py requirements-dev.txt constraints.txt

echo "+ 'git commit' wanted files"
# Commit and push the changes.
git commit \
    -m "Automated documentation update to add .po files from qiskit" \
    -m "skip ci" \
    -m "Commit: $GITHUB_SHA" \
    -m "Github Actions Run: $GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID"

echo "+ 'git push' to target repository"
git push --quiet origin "$TARGET_REPO_BRANCH"
echo "********** End of pushing po to working repo! *************"
popd
