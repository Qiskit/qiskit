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

import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

TERRA_FILES = []
TERRA_DIRS = [
    "apidoc",
    "migration_guides",
    "_templates",
    "source_images",
]


def _get_current_terra_version(app):
    setup_py = Path(app.srcdir, "../", "setup.py").read_text()
    version_regex = re.compile("qiskit-terra" + '[=|>]=(.*)"')
    match = version_regex.search(setup_py)
    return match[1]


def load_terra_docs(app):
    """Git clones Qiskit Terra docs."""
    if all(Path(app.srcdir, fp).exists() for fp in [*TERRA_DIRS, *TERRA_FILES]):
        warnings.warn(
            "Terra docs already exist. Skipping Git clone. These docs may be out of date! "
            "Run `tox -e docs-clean` to remove these docs."
        )
        return

    version = _get_current_terra_version(app)
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "https://github.com/Qiskit/qiskit-terra", temp_dir],
            capture_output=True,
        )
        subprocess.run(["git", "checkout", version], cwd=temp_dir, capture_output=True)
        for d in TERRA_DIRS:
            src = Path(temp_dir, "docs", "apidocs" if d == "apidoc" else d)
            shutil.copytree(src, Path(app.srcdir, d))
        for f in TERRA_FILES:
            src = Path(temp_dir, "docs", f)
            shutil.copy(src, Path(app.srcdir, f))


def load_tutorials(app):
    """Git clones the tutorials repo so that we can generate their docs."""
    if Path(app.srcdir, "tutorials").exists():
        warnings.warn(
            "Tutorials already exist. Skipping Git clone. These docs may be out of date! "
            "Run `tox -e docs-clean` to remove these docs."
        )
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "https://github.com/Qiskit/qiskit-tutorials", temp_dir],
            capture_output=True,
        )
        shutil.copytree(Path(temp_dir, "tutorials"), Path(app.srcdir, "tutorials"))


def clean_docs(app, exc):
    """Deletes the Git cloned docs."""
    for d in [*TERRA_DIRS, "tutorials"]:
        shutil.rmtree(Path(app.srcdir, d))
    for f in TERRA_FILES:
        Path(app.srcdir, f).unlink()
