# -*- coding: utf-8 -*-
# pylint: disable=import-error
# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=too-many-ancestors

"""Common utilities for QISKit."""

import os
import logging
import re
import sys
import warnings
from collections import UserDict
import multiprocessing
import numpy as np
from qiskit._qiskiterror import QISKitError

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


def _mac_hardware_info():
    """Returns system info on OSX.
    """
    info = dict()
    results = dict()
    for line in [l.split(':') for l in os.popen('sysctl hw').readlines()[1:]]:
        info[line[0].strip(' "').replace(' ', '_').lower().strip('hw.')] = \
            line[1].strip('.\n ')
    results.update({'cpus': int(info['physicalcpu'])})
    results.update({'cpu_freq': int(
        float(os.popen('sysctl -n machdep.cpu.brand_string')
              .readlines()[0].split('@')[1][:-4])*1000)})
    results.update({'memsize': int(int(info['memsize']) / (1024 ** 2))})
    # add OS information
    results.update({'os': 'Mac OSX'})
    return results


def _linux_hardware_info():
    """Returns system info on Linux.
    """
    results = {}
    # get cpu number
    sockets = 0
    cores_per_socket = 0
    frequency = 0.0
    for line in [l.split(':') for l in open("/proc/cpuinfo").readlines()]:
        if line[0].strip() == "physical id":
            sockets = max(sockets, int(line[1].strip())+1)
        if line[0].strip() == "cpu cores":
            cores_per_socket = int(line[1].strip())
        if line[0].strip() == "cpu MHz":
            frequency = float(line[1].strip()) / 1000.
    results.update({'cpus': sockets * cores_per_socket})
    # get cpu frequency directly (bypasses freq scaling)
    try:
        file = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        line = open(file).readlines()[0]
        frequency = float(line.strip('\n')) / 1000000.
    except IOError:
        pass
    results.update({'cpu_freq': frequency})

    # get total amount of memory
    mem_info = dict()
    for line in [l.split(':') for l in open("/proc/meminfo").readlines()]:
        mem_info[line[0]] = line[1].strip('.\n ').strip('kB')
    results.update({'memsize': int(mem_info['MemTotal']) / 1024})
    # add OS information
    results.update({'os': 'Linux'})
    return results


def _win_hardware_info():
    """Returns system info on Windows.
    """
    results = {'os': 'Windows'}
    try:
        from comtypes.client import CoGetObject

    except ImportError:
        ncpus = int(multiprocessing.cpu_count())
        results.update({'cpus': ncpus})

    else:
        winmgmts_root = CoGetObject("winmgmts:root\\cimv2")
        cpus = winmgmts_root.ExecQuery("Select * from Win32_Processor")
        ncpus = 0
        freq = 0
        for cpu in cpus:
            ncpus += int(cpu.Properties_['NumberOfCores'].Value)
            if not freq:
                freq = int(cpu.Properties_['MaxClockSpeed'].Value)
        results.update({'cpu_freq': freq})
        ncpus = int(multiprocessing.cpu_count())
        results.update({'cpus': ncpus})
        mem = winmgmts_root.ExecQuery("Select * from Win32_ComputerSystem")
        tot_mem = 0
        for item in mem:
            tot_mem += int(item.Properties_['TotalPhysicalMemory'].Value)
        tot_mem = int(tot_mem / 1024**2)
        results.update({'memsize': tot_mem})

    return results


def local_hardware_info():
    """Basic hardware information about the local machine.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.

    Returns:
        dict: The hardware information.

    """
    if sys.platform == 'darwin':
        out = _mac_hardware_info()
    elif sys.platform == 'win32':
        out = _win_hardware_info()
    elif sys.platform in ['linux', 'linux2']:
        out = _linux_hardware_info()
    else:
        out = {}
    return out


def verify_qubit_number(num_qubits):
    """Determines if an user can run a simulation
    with a given number of qubits, as set by their
    system hardware.
    """
    local_hardware = local_hardware_info()
    if 'memsize' in local_hardware.keys():
        # system memory in MB
        sys_mem = local_hardware['memsize']
    else:
        raise QISKitError('Cannot determine local memory size.')
    max_qubits = np.log2(sys_mem*(1024**2)/128)
    if num_qubits > max_qubits:
        raise QISKitError("Number of qubits exceeds local memory.")
