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

import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from distutils import dir_util

apidocs_exists = False
apidocs_master = None


def _get_current_versions(app):
    setup_py_path = os.path.join(os.path.dirname(app.srcdir), "setup.py")
    with open(setup_py_path, "r") as fd:
        setup_py = fd.read()
    version_regex = re.compile("qiskit-terra" + '[=|>]=(.*)"')
    match = version_regex.search(setup_py)
    return match[1]


def _install_from_master():
    github_url = "git+https://github.com/Qiskit/qiskit-terra"
    cmd = [sys.executable, "-m", "pip", "install", "-U", github_url]
    subprocess.run(cmd)


def _git_copy(sha1, meta_package_docs_dir, sub_package_docs_folder):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            github_source = "https://github.com/Qiskit/qiskit-terra"
            subprocess.run(["git", "clone", github_source, temp_dir], capture_output=True)
            subprocess.run(["git", "checkout", sha1], cwd=temp_dir, capture_output=True)
            dir_util.copy_tree(
                os.path.join(temp_dir, "docs", sub_package_docs_folder), meta_package_docs_dir
            )

    except FileNotFoundError:
        warnings.warn(
            f"Copy from git failed for qiskit-terra at {sha1}, skipping...", RuntimeWarning
        )


def load_api_sources(app):
    """Git clones and sets up Qiskit repos so that we can generate their API docs."""
    api_docs_dir = os.path.join(app.srcdir, "apidoc")
    migration_guides_dir = os.path.join(app.srcdir, "migration_guides")
    if os.getenv("DOCS_FROM_MASTER"):
        global apidocs_master
        apidocs_master = tempfile.mkdtemp()
        shutil.move(api_docs_dir, apidocs_master)
        _install_from_master()
        _git_copy("HEAD", api_docs_dir, "apidocs")
        _git_copy("HEAD", migration_guides_dir, "migration_guides")
        return
    elif os.path.isdir(api_docs_dir):
        global apidocs_exists
        apidocs_exists = True
        warnings.warn("docs/apidocs already exists skipping source clone")
        return

    version = _get_current_versions(app)
    _git_copy(version, api_docs_dir, "apidocs")
    _git_copy(version, migration_guides_dir, "migration_guides")


def load_tutorials(app):
    """Git clones the tutorials repo so that we can generate their docs."""
    tutorials_dir = os.path.join(app.srcdir, "tutorials")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            github_source = "https://github.com/Qiskit/qiskit-tutorials"
            subprocess.run(["git", "clone", github_source, temp_dir], capture_output=True)
            dir_util.copy_tree(os.path.join(temp_dir, "tutorials"), tutorials_dir)
    except FileNotFoundError:
        warnings.warn(
            "Copy from git failed for qiskit-tutorials, skipping...",
            RuntimeWarning,
        )


def clean_api_source(app, exc):
    """Deletes the Git cloned repos used for API doc generation."""
    api_docs_dir = os.path.join(app.srcdir, "apidoc")
    global apidocs_exists
    global apidocs_master
    if apidocs_exists:
        return
    elif apidocs_master:
        shutil.rmtree(api_docs_dir)
        shutil.move(os.path.join(apidocs_master, "apidoc"), api_docs_dir)
        return
    shutil.rmtree(api_docs_dir)


def clean_tutorials(app, exc):
    """Deletes the Git cloned tutorials repo used for doc generation."""
    tutorials_dir = os.path.join(app.srcdir, "tutorials")
    shutil.rmtree(tutorials_dir)
