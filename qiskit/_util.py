# -*- coding: utf-8 -*-
# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=too-many-ancestors

"""Common utilities for QISKit."""

import logging
import re
import sys
import platform
import warnings
import socket
from collections import UserDict
import psutil

API_NAME = 'IBMQuantumExperience'
logger = logging.getLogger(__name__)

FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')


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
    try:
        warnings._add_filter(*deprecation_filter, append=False)
    except AttributeError:
        # ._add_filter is internal and not available in some Python versions.
        pass


def _camel_case_to_snake_case(identifier):
    """Return a `snake_case` string from a `camelCase` string.

    Args:
        identifier (str): a `camelCase` string.

    Returns:
        str: a `snake_case` string.
    """
    string_1 = FIRST_CAP_RE.sub(r'\1_\2', identifier)
    return ALL_CAP_RE.sub(r'\1_\2', string_1).lower()


_check_python_version()
_check_ibmqx_version()
_enable_deprecation_warnings()


class AvailableToOperationalDict(UserDict):
    """
    TEMPORARY class for transitioning from `status['available']` to
    `status['operational']`.

    FIXME: Remove this class as soon as the API is updated, please.
    """
    def __getitem__(self, key):
        if key == 'available':
            warnings.warn(
                "status['available'] has been renamed to status['operational'] "
                " since 0.5.5. Please use status['operational'] accordingly.",
                DeprecationWarning)

        return super(AvailableToOperationalDict, self).__getitem__(key)


def _dict_merge(dct, merge_dct):
    """
    TEMPORARY method for merging backend.calibration & backend.parameters
    into backend.properties.

    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        dct (dict): the dictionary to merge into
        merge_dct (dict): the dictionary to merge
    """
    for k, _ in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):
            _dict_merge(dct[k], merge_dct[k])
        elif (k in dct and isinstance(dct[k], list) and isinstance(merge_dct[k], list)):
            for i in range(len(dct[k])):
                _dict_merge(dct[k][i], merge_dct[k][i])
        else:
            dct[k] = merge_dct[k]


def _parse_ibmq_credentials(url, hub=None, group=None, project=None):
    """Converts old Q network credentials to new url only
    format, if needed.
    """
    if any([hub, group, project]):
        url = "https://q-console-api.mybluemix.net/api/" + \
              "Hubs/{hub}/Groups/{group}/Projects/{project}"
        url = url.format(hub=hub, group=group, project=project)
        warnings.warn(
            "Passing hub/group/project as parameters is deprecated in qiskit "
            "0.6+. Please use the new URL format provided in the q-console.",
            DeprecationWarning)
    return url


def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.

    Returns:
        dict: The hardware information.

    """
    results = {'os': platform.system()}
    results['memory'] = psutil.virtual_memory().total / (1024**3)
    results['cpus'] = psutil.cpu_count(logical=False)
    return results


def _has_connection(hostname, port):
    """Checks to see if internet connection exists to host
    via specified port

    Args:
        hostname (str): Hostname to connect to.
        port (int): Port to connect to

    Returns:
        bool: Has connection or not

    Raises:
        gaierror: No connection established.
    """
    try:
        host = socket.gethostbyname(hostname)
        socket.create_connection((host, port), 2)
        return True
    except socket.gaierror:
        pass
    return False
