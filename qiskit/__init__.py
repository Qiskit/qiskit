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

# pylint: disable=wrong-import-order,invalid-name,wrong-import-position

"""Main Qiskit public functionality."""

import sys
import warnings
import os

# First, check for required Python and API version
from . import util

# Extend namespace for backwards compat
from qiskit import namespace
new_meta_path = []
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aer', 'qiskit.providers.aer'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_ignis', 'qiskit.ignis'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua', 'qiskit.aqua'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.ai', 'qiskit.ai'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.chemistry', 'qiskit.chemistry'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.finance', 'qiskit.finance'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.ml', 'qiskit.ml'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.optimizations', 'qiskit.optimizations'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_ibmq_provider', 'qiskit.providers.ibmq'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqt_provider', 'qiskit.providers.aqt'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_honeywell_provider', 'qiskit.providers.honeywell'))
# Add Qiskit importers to meta_path before PathFinder in the default
# sys.meta_path to avoid the miss penalty on trying to import a module
# which does not exist
old_meta_path = sys.meta_path
sys.meta_path = old_meta_path[:-1] + new_meta_path + [old_meta_path[-1]]

# qiskit errors operator
from qiskit.exceptions import QiskitError  # noqa

# The main qiskit operators
from qiskit.circuit import ClassicalRegister  # noqa
from qiskit.circuit import QuantumRegister  # noqa
from qiskit.circuit import AncillaRegister  # noqa
from qiskit.circuit import QuantumCircuit  # noqa

# user config
from qiskit import user_config as _user_config  # noqa

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions  # noqa
import qiskit.circuit.measure  # noqa
import qiskit.circuit.reset  # noqa

# Please note these are global instances, not modules.
from qiskit.providers.basicaer import BasicAer  # noqa

_config = _user_config.get_config()

# Try to import the Aer provider if installed.
try:
    from qiskit.providers.aer import Aer
except ImportError:
    suppress_warnings = os.environ.get('QISKIT_SUPPRESS_PACKAGING_WARNINGS', '')
    if suppress_warnings.upper() != 'Y':
        if not _config.get('suppress_packaging_warnings') or suppress_warnings.upper() == 'N':
            warnings.warn('Could not import the Aer provider from the qiskit-aer '
                          'package. Install qiskit-aer or check your installation.',
                          RuntimeWarning)

# Try to import the IBMQ provider if installed.
try:
    from qiskit.providers.ibmq import IBMQ
except ImportError:
    suppress_warnings = os.environ.get('QISKIT_SUPPRESS_PACKAGING_WARNINGS', '')
    if suppress_warnings.upper() != 'Y':
        if not _config.get('suppress_packaging_warnings') or suppress_warnings.upper() == 'N':
            warnings.warn('Could not import the IBMQ provider from the '
                          'qiskit-ibmq-provider package. Install '
                          'qiskit-ibmq-provider or check your installation.',
                          RuntimeWarning)

from qiskit.execute import execute  # noqa
from qiskit.compiler import transpile, assemble, schedule  # noqa

from .version import __version__  # noqa
from .version import _get_qiskit_versions  # noqa


if sys.version_info[0] == 3 and sys.version_info[1] == 5:
    warnings.warn(
        "Using Qiskit with Python 3.5 is deprecated as of the 0.12.0 release. "
        "Support for running Qiskit with Python 3.5 will be removed at the "
        "Python 3.5 EoL on 09/13/2020.", DeprecationWarning)


__qiskit_version__ = _get_qiskit_versions()
