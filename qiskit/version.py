# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=no-name-in-module,broad-except,cyclic-import

"""Contains Qiskit (terra) version."""

import os
import subprocess
from collections.abc import Mapping

import warnings

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ["SYSTEMROOT", "PATH"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=os.path.join(os.path.dirname(ROOT_DIR)),
    ) as proc:
        stdout, stderr = proc.communicate()
        if proc.returncode > 0:
            error_message = stderr.strip().decode("ascii")
            raise OSError(f"Command {cmd} exited with code {proc.returncode}: {error_message}")
    return stdout


def git_version():
    """Get the current git head sha1."""
    # Determine if we're at main
    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip().decode("ascii")
    except OSError:
        git_revision = "Unknown"

    return git_revision


with open(os.path.join(ROOT_DIR, "VERSION.txt")) as version_file:
    VERSION = version_file.read().strip()


def get_version_info():
    """Get the full version string."""
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of scipy.version messes
    # up the build under Python 3.
    full_version = VERSION

    if not os.path.exists(os.path.join(os.path.dirname(ROOT_DIR), ".git")):
        return full_version
    try:
        release = _minimal_ext_cmd(["git", "tag", "-l", "--points-at", "HEAD"])
    except Exception:  # pylint: disable=broad-except
        return full_version
    git_revision = git_version()
    if not release:
        full_version += ".dev0+" + git_revision[:7]

    return full_version


__version__ = get_version_info()


class QiskitVersion(Mapping):
    """DEPRECATED in 0.25.0 use qiskit.__version__"""

    __slots__ = ["_version_dict", "_loaded"]

    def __init__(self):
        warnings.warn(
            "qiskit.__qiskit_version__ is deprecated since "
            "Qiskit Terra 0.25.0, and will be removed 3 months or more later. "
            "Instead, you should use qiskit.__version__. The other packages listed in "
            "former qiskit.__qiskit_version__ have their own __version__ module level dunder, "
            "as standard in PEP 8.",
            category=DeprecationWarning,
        )
        self._version_dict = {
            "qiskit-terra": __version__,
            "qiskit": None,
        }
        self._loaded = False

    def _load_versions(self):
        from importlib.metadata import version

        try:
            # TODO: Update to use qiskit_aer instead when we remove the
            # namespace redirect
            from qiskit.providers import aer

            self._version_dict["qiskit-aer"] = aer.__version__
        except Exception:
            self._version_dict["qiskit-aer"] = None
        try:
            from qiskit import ignis

            self._version_dict["qiskit-ignis"] = ignis.__version__
        except Exception:
            self._version_dict["qiskit-ignis"] = None
        try:
            from qiskit.providers import ibmq

            self._version_dict["qiskit-ibmq-provider"] = ibmq.__version__
        except Exception:
            self._version_dict["qiskit-ibmq-provider"] = None
        try:
            import qiskit_nature

            self._version_dict["qiskit-nature"] = qiskit_nature.__version__
        except Exception:
            self._version_dict["qiskit-nature"] = None
        try:
            import qiskit_finance

            self._version_dict["qiskit-finance"] = qiskit_finance.__version__
        except Exception:
            self._version_dict["qiskit-finance"] = None
        try:
            import qiskit_optimization

            self._version_dict["qiskit-optimization"] = qiskit_optimization.__version__
        except Exception:
            self._version_dict["qiskit-optimization"] = None
        try:
            import qiskit_machine_learning

            self._version_dict["qiskit-machine-learning"] = qiskit_machine_learning.__version__
        except Exception:
            self._version_dict["qiskit-machine-learning"] = None
        try:
            self._version_dict["qiskit"] = version("qiskit")
        except Exception:
            self._version_dict["qiskit"] = None
        self._loaded = True

    def __repr__(self):
        if not self._loaded:
            self._load_versions()
        return repr(self._version_dict)

    def __str__(self):
        if not self._loaded:
            self._load_versions()
        return str(self._version_dict)

    def __getitem__(self, key):
        if not self._loaded:
            self._load_versions()
        return self._version_dict[key]

    def __iter__(self):
        if not self._loaded:
            self._load_versions()
        return iter(self._version_dict)

    def __len__(self):
        return len(self._version_dict)


__qiskit_version__ = QiskitVersion()
