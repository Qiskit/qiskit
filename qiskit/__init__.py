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
    'qiskit_aqua.chemistry', 'qiskit.chemistry'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.finance', 'qiskit.finance'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.ml', 'qiskit.ml'))
new_meta_path.append(namespace.QiskitElementImport(
    'qiskit_aqua.optimization', 'qiskit.optimization'))
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

# Moved to after IBMQ and Aer imports due to import issues
# with other modules that check for IBMQ (tools)
from qiskit.execute_function import execute  # noqa
from qiskit.compiler import transpile, assemble, schedule, sequence  # noqa

from .version import __version__  # noqa
from .version import QiskitVersion  # noqa


__qiskit_version__ = QiskitVersion()


if sys.version_info[0] == 3 and sys.version_info[1] == 6:
    warnings.warn(
        "Using Qiskit with Python 3.6 is deprecated as of the 0.17.0 release. "
        "Support for running Qiskit with Python 3.6 will be removed in a "
        "future release.", DeprecationWarning)


class AerWrapper:
    """Lazy loading wrapper for Aer provider."""

    def __init__(self):
        self.aer = None

    def __bool__(self):
        if self.aer is None:
            try:
                from qiskit_aer
                self.aer = qiskit_aer.Aer
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.aer:
            try:
                import qiskit_aer
                self.aer = qiskit_aer.Aer
            except ImportError as exc:
                raise ImportError('Could not import the Aer provider from the '
                                  'qiskit-aer package. Install qiskit-aer or '
                                  'check your installation.') from exc
        return getattr(self.aer, attr)


class IBMQWrapper:
    """Lazy loading wraooer for IBMQ provider."""

    def __init__(self):
        self.ibmq = None

    def __bool__(self):
        if self.ibmq is None:
            try:
                from qiskit_ibmq_provider as ibmq
                self.ibmq = ibmq.IBMQ
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.ibmq:
            try:
                from qiskit_ibmq_provider as ibmq
                self.ibmq = ibmq.IBMQ
            except ImportError as exc:
                raise ImportError('Could not import the IBMQ provider from the '
                                  'qiskit-ibmq-provider package. Install '
                                  'qiskit-ibmq-provider or check your  '
                                  'installation.') from exc
        return getattr(self.ibmq, attr)


Aer = AerWrapper()
IBMQ = IBMQWrapper()
