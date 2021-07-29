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

# pylint: disable=wrong-import-position

"""Main Qiskit public functionality."""

import pkgutil
import sys
import warnings
import os

# qiskit errors operator
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import AncillaRegister
from qiskit.circuit import QuantumCircuit

# user config
from qiskit import user_config as _user_config

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions
import qiskit.circuit.measure
import qiskit.circuit.reset

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code AND *before*
# importing the package you want to allow extensions for (in this case `backends`).
__path__ = pkgutil.extend_path(__path__, __name__)

# Please note these are global instances, not modules.
from qiskit.providers.basicaer import BasicAer

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
        "future release.",
        DeprecationWarning,
    )


class AerWrapper:
    """Lazy loading wrapper for Aer provider."""

    def __init__(self):
        self.aer = None

    def __bool__(self):
        if self.aer is None:
            try:
                from qiskit.providers import aer

                self.aer = aer.Aer
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.aer:
            try:
                from qiskit.providers import aer

                self.aer = aer.Aer
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    "qiskit-aer", "Aer provider", "pip install qiskit-aer"
                ) from ex
        return getattr(self.aer, attr)


class IBMQWrapper:
    """Lazy loading wrapper for IBMQ provider."""

    def __init__(self):
        self.ibmq = None

    def __bool__(self):
        if self.ibmq is None:
            try:
                from qiskit.providers import ibmq

                self.ibmq = ibmq.IBMQ
            except ImportError:
                return False
        return True

    def __getattr__(self, attr):
        if not self.ibmq:
            try:
                from qiskit.providers import ibmq

                self.ibmq = ibmq.IBMQ
            except ImportError as ex:
                raise MissingOptionalLibraryError(
                    "qiskit-ibmq-provider", "IBMQ provider", "pip install qiskit-ibmq-provider"
                ) from ex
        return getattr(self.ibmq, attr)


Aer = AerWrapper()
IBMQ = IBMQWrapper()
