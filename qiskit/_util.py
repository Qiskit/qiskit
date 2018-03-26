# -*- coding: utf-8 -*-

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
"""Common utilities for QISKit."""

import logging
import re
import sys
import warnings

API_NAME = 'IBMQuantumExperience'
logger = logging.getLogger(__name__)


def _check_python_version():
    """Check for Python version 3.5+
    """
    if sys.version_info < (3, 5):
        raise Exception('QISKit requires Python version 3.5 or greater.')


def _check_ibmqx_version():
    """Check if the available IBMQuantumExperience version is the required one.

    Check that the installed "IBMQuantumExperience" package version matches the
    version required by the package, emitting a warning if it is not present.

    Note:
        The check is only performed when `qiskit` is installed via `pip`
        (available under `pkg_resources.working_set`). For other configurations
        (such as local development, etc), the check is skipped silently.
    """
    try:
        # Use a local import, as in very specific environments setuptools
        # might not be available or updated (conda with specific setup).
        import pkg_resources
        working_set = pkg_resources.working_set
        qiskit_pkg = working_set.by_key['qiskit']
    except (ImportError, KeyError):
        # If 'qiskit' was not found among the installed packages, silently
        # return.
        return

    # Find the IBMQuantumExperience version specified in this release of qiskit
    # based on pkg_resources (in turn, based on setup.py::install_requires).
    ibmqx_require = next(r for r in qiskit_pkg.requires() if
                         r.name == API_NAME)

    # Finally, compare the versions.
    try:
        # First try to use IBMQuantumExperience.__version__ directly.
        from IBMQuantumExperience import __version__ as ibmqx_version

        if ibmqx_version in ibmqx_require:
            return
    except ImportError:
        # __version__ was not available, so try to compare using the
        # working_set. This assumes IBMQuantumExperience is installed as a
        # library (using pip, etc).
        try:
            working_set.require(str(ibmqx_require))
            return
        except pkg_resources.DistributionNotFound:
            # IBMQuantumExperience was not found among the installed libraries.
            # The warning is not printed, assuming the user is using a local
            # version and takes responsibility of handling the versions.
            return
        except pkg_resources.VersionConflict:
            pass

    logger.warning('The installed IBMQuantumExperience package does '
                   'not match the required version - some features might '
                   'not work as intended. Please install %s.',
                   str(ibmqx_require))


def _enable_deprecation_warnings():
    """
    Force the `DeprecationWarning` warnings to be displayed for the qiskit
    module, overriding the system configuration as they are ignored by default
    [1] for end-users.

    TODO: on Python 3.7, this might not be needed due to PEP-0565 [2].

    [1] https://docs.python.org/3/library/warnings.html#default-warning-filters
    [2] https://www.python.org/dev/peps/pep-0565/
    """
    # pylint: disable=invalid-name
    deprecation_filter = ('always', None, DeprecationWarning,
                          re.compile(r'^qiskit\.*', re.UNICODE), 0)

    # Instead of using warnings.simple_filter() directly, the internal
    # _add_filter() function is used for being able to match against the
    # module.
    warnings._add_filter(*deprecation_filter, append=False)


_check_python_version()
_check_ibmqx_version()
_enable_deprecation_warnings()
