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

API_NAME = 'IBMQuantumExperience'
logger = logging.getLogger(__name__)


def _check_ibmqe_version():
    """Check if the available IBMQuantumExperience version is the required one.

    Check that the version of the available "IBMQuantumExperience" package
    matches the version required by the package, emitting a warning if it is
    not present.
    """
    try:
        # Use a local import, as in very specific environments setuptools
        # might not be available or updated (conda with specific setup).
        import pkg_resources
    except ImportError:
        return

    working_set = pkg_resources.working_set
    # Find if qiskit is installed and the current execution is using the
    # installed package; or if it is a local environment.
    qiskit_local = True
    try:
        qiskit_pkg = working_set.by_key['qiskit']
        if __file__.startswith(qiskit_pkg.location):
            qiskit_local = False
    except KeyError:
        pass

    # Find the IBMQuantumExperience version specified in qiskit.
    if qiskit_local:
        try:
            with open('requirements.txt') as reqfile:
                ibmqe_require_line = next(line for line in reqfile if
                                          line.startswith(API_NAME))
                ibmqe_require = pkg_resources.Requirement(ibmqe_require_line)
        except (FileNotFoundError, StopIteration, pkg_resources.RequirementParseError):
            logger.warning(
                'Could not find %s in requirements.txt or the requirements.txt \
                file was not found or unparsable', API_NAME)
            return
    else:
        # Retrieve the requirement line from pkg_resources
        ibmqe_require = next(r for r in qiskit_pkg.requires() if
                             r.name == API_NAME)

    # Finally, compare the versions.
    try:
        # First try to use IBMQuantumExperience.__version__ directly.
        from IBMQuantumExperience import __version__ as ibmqe_version

        if ibmqe_version in ibmqe_require:
            return
    except ImportError:
        # __version__ was not available, so try to compare using the
        # working_set. This assumes IBMQuantumExperience is installed as a
        # library (using pip, etc).
        try:
            working_set.require(str(ibmqe_require))
            return
        except pkg_resources.DistributionNotFound:
            # IBMQuantumExperience was not found among the installed libraries.
            # The warning is not printed, assuming the user is using a local
            # version and takes responsability of handling the versions.
            return
        except pkg_resources.VersionConflict:
            pass

    logger.warning('The installed IBMQuantumExperience package does '
                   'not match the required version - some features might '
                   'not work as intended. Please install %s.',
                   str(ibmqe_require))
