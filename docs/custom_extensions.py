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

# Elements with api doc sources
qiskit_elements = ['qiskit-terra', 'qiskit-aer', 'qiskit-ibmq-provider']
apidocs_exists = False
apidocs_master = None


def _get_current_versions(app):
    versions = {}
    setup_py_path = os.path.join(os.path.dirname(app.srcdir), 'setup.py')
    with open(setup_py_path, 'r') as fd:
        setup_py = fd.read()
        for package in qiskit_elements:
            version_regex = re.compile(package + '[=|>]=(.*)\"')
            match = version_regex.search(setup_py)
            if match:
                ver = match[1]
                versions[package] = ver
    return versions


def _install_from_master():
    for package in qiskit_elements + ['qiskit-ignis']:
        github_url = 'git+https://github.com/Qiskit/%s' % package
        cmd = [sys.executable, '-m', 'pip', 'install', '-U', github_url]
        subprocess.run(cmd)


def _git_copy(package, sha1, api_docs_dir):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            github_source = 'https://github.com/Qiskit/%s' % package
            subprocess.run(['git', 'clone', github_source, temp_dir],
                           capture_output=True)
            subprocess.run(['git', 'checkout', sha1], cwd=temp_dir,
                           capture_output=True)
            dir_util.copy_tree(
                os.path.join(temp_dir, 'docs', 'apidocs'),
                api_docs_dir)

    except FileNotFoundError:
        warnings.warn('Copy from git failed for %s at %s, skipping...' %
                      (package, sha1), RuntimeWarning)


def load_api_sources(app):
    """Git clones and sets up Qiskit repos so that we can generate their API docs."""
    api_docs_dir = os.path.join(app.srcdir, 'apidoc')
    if os.getenv('DOCS_FROM_MASTER'):
        global apidocs_master
        apidocs_master = tempfile.mkdtemp()
        shutil.move(api_docs_dir, apidocs_master)
        _install_from_master()
        for package in qiskit_elements:
            _git_copy(package, 'HEAD', api_docs_dir)
        return
    elif os.path.isdir(api_docs_dir):
        global apidocs_exists
        apidocs_exists = True
        warnings.warn('docs/apidocs already exists skipping source clone')
        return
    meta_versions = _get_current_versions(app)
    for package in qiskit_elements:
        _git_copy(package, meta_versions[package], api_docs_dir)


def load_tutorials(app):
    """Git clones the tutorials repo so that we can generate their docs."""
    tutorials_dir = os.path.join(app.srcdir, 'tutorials')
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            github_source = 'https://github.com/Qiskit/qiskit-tutorials'
            subprocess.run(['git', 'clone', github_source, temp_dir],
                           capture_output=True)
            dir_util.copy_tree(
                os.path.join(temp_dir, 'tutorials'),
                tutorials_dir)
    except FileNotFoundError:
        warnings.warn(
            'Copy from git failed for qiskit-tutorials, skipping...',
            RuntimeWarning,
        )


def clean_api_source(app, exc):
    """Deletes the Git cloned repos used for API doc generation."""
    api_docs_dir = os.path.join(app.srcdir, 'apidoc')
    global apidocs_exists
    global apidocs_master
    if apidocs_exists:
        return
    elif apidocs_master:
        shutil.rmtree(api_docs_dir)
        shutil.move(os.path.join(apidocs_master, 'apidoc'), api_docs_dir)
        return
    shutil.rmtree(api_docs_dir)


def clean_tutorials(app, exc):
    """Deletes the Git cloned tutorials repo used for doc generation."""
    tutorials_dir = os.path.join(app.srcdir, 'tutorials')
    shutil.rmtree(tutorials_dir)


def deprecate_ibmq_provider(app, docname, source):
    """Adds a deprecation message to the top of every qiskit-ibmq-provider page."""
    message = """.. warning::
       The package ``qiskit-ibmq-provider`` is being deprecated and its repo is going to be
       archived soon. Please transition to the new packages. More information in
       https://ibm.biz/provider_migration_guide\n\n"""
    if 'apidoc/ibmq' in docname or 'qiskit.providers.ibmq' in docname:
        source[0] = message + source[0]
