# -*- coding: utf-8 -*-
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Common utilities for Qiskit."""

import platform
import re
import socket
import sys
import warnings
import functools

import psutil


def _check_python_version():
    """Check for Python version 3.5+."""
    if sys.version_info < (3, 5):
        raise Exception('Qiskit requires Python version 3.5 or greater.')


def _filter_deprecation_warnings():
    """Apply filters to deprecation warnings.

    Force the `DeprecationWarning` warnings to be displayed for the qiskit
    module, overriding the system configuration as they are ignored by default
    [1] for end-users. Additionally, silence the `ChangedInMarshmallow3Warning`
    messages.

    TODO: on Python 3.7, this might not be needed due to PEP-0565 [2].

    [1] https://docs.python.org/3/library/warnings.html#default-warning-filters
    [2] https://www.python.org/dev/peps/pep-0565/
    """
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


_check_python_version()
_filter_deprecation_warnings()


def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on. CPU count defaults to 1 when true count can't be determined.

    Returns:
        dict: The hardware information.
    """
    results = {
        'os': platform.system(),
        'memory': psutil.virtual_memory().total / (1024 ** 3),
        'cpus': psutil.cpu_count(logical=False) or 1
    }
    return results


def _has_connection(hostname, port):
    """Checks if internet connection exists to host via specified port.

    If any exception is raised while trying to open a socket this will return
    false.

    Args:
        hostname (str): Hostname to connect to.
        port (int): Port to connect to

    Returns:
        bool: Has connection or not

    """
    try:
        host = socket.gethostbyname(hostname)
        socket.create_connection((host, port), 2).close()
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def deprecate_arguments(kwarg_map):
    """Decorator to automatically alias deprecated agrument names and warn upon use."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                _rename_kwargs(func.__name__, kwargs, kwarg_map)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _rename_kwargs(func_name, kwargs, kwarg_map):
    for old_arg, new_arg in kwarg_map.items():
        if old_arg in kwargs:
            if new_arg in kwargs:
                raise TypeError('{} received both {} and {} (deprecated).'.format(
                    func_name, new_arg, old_arg))

            warnings.warn('{} keyword argument {} is deprecated and '
                          'replaced with {}.'.format(
                              func_name, old_arg, new_arg),
                          DeprecationWarning, stacklevel=3)

            kwargs[new_arg] = kwargs.pop(old_arg)
