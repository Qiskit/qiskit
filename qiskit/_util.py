# -*- coding: utf-8 -*-
# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Common utilities for Qiskit."""

import logging
import platform
import re
import socket
import os
import sys
import warnings
import inspect
import pkgutil
import importlib
from math import log2
import psutil
import numpy
import scipy
import networkx as nx
from marshmallow.warnings import ChangedInMarshmallow3Warning
import qiskit.providers

logger = logging.getLogger(__name__)


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

    # Add a filter for ignoring ChangedInMarshmallow3Warning, as we depend on
    # marhsmallow 2 explicitly. 2.17.0 introduced new deprecation warnings that
    # are useful for eventually migrating, but too verbose for our purposes.
    warnings.simplefilter('ignore', category=ChangedInMarshmallow3Warning)


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
        socket.create_connection((host, port), 2)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def about():
    """ Returns information about the Qiskit installation.
    """
    hardware_info = local_hardware_info()
    n_qubits = int(log2(local_hardware_info()['memory']*(1024**3)/16))

    print("")
    print("Qiskit Details")
    print("==============")
    print("Package         Version Number")
    print("------------------------------")
    print("Qiskit-terra:       %s" % qiskit.__version__)
    print("Numpy:              %s" % numpy.__version__)
    print("Scipy:              %s" % scipy.__version__)
    print("Networkx:           %s" % nx.__version__)
    print("Python:             %d.%d.%d" % sys.version_info[0:3])
    print("")
    print("Available Providers")
    print("-------------------")
    offset = 20
    for pro in get_available_providers():
        print(pro[0]+' '*(offset-len(pro[0]))+(pro[1] if pro[1] else ''))

    print("")
    print("Additional Information")
    print("----------------------")
    print("Memory:             %s GB [%s qubits]" % (hardware_info['memory'], n_qubits))
    print("Number of CPUs:     %s" % hardware_info['cpus'])
    print("Platform Info:      %s (%s)" % (platform.system(),
                                           platform.machine()))
    qiskit_install_path = os.path.dirname(inspect.getsourcefile(qiskit))
    print("Install path:       %s" % qiskit_install_path)


def get_available_providers():
    """Returns a list of available providers and their
    versioning (if any).

    Returns:
        list: List of (provider_name, version) tuples.
    """
    package = qiskit.providers
    providers = []
    for _, modname, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg and modname != 'models':
            mod = importlib.import_module("qiskit.providers."+modname)
            try:
                version = mod.__version__
            except AttributeError:
                version = None
            providers.append((modname, version))
    return providers
