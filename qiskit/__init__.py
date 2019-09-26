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

# pylint: disable=wrong-import-order, wrong-import-position


"""Main Qiskit public functionality."""
import sys
import pkgutil
import warnings

# First, check for required Python and API version
from . import util

import qiskit.version

# qiskit errors operator
from .exceptions import QiskitError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.execute import execute
from qiskit.compiler import transpile, assemble

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions
import qiskit.circuit.measure
import qiskit.circuit.reset

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code AND *before*
# importing the package you want to allow extensions for (in this case `backends`).
__path__ = pkgutil.extend_path(__path__, __name__)


class QiskitBaseModule():
    """Emulate qiskit base package as a class so that __version__ can be a
    @property. Can be converted to a module-level __getattr__ once support is
    available on all supported python versions (See PEP 562)."""

    __path__ = __path__
    QiskitError = QiskitError
    ClassicalRegister = ClassicalRegister
    QuantumRegister = QuantumRegister
    QuantumCircuit = QuantumCircuit
    execute = execute
    transpile = transpile
    assemble = assemble

    __version__cache = None

    @property
    def __version__(self):
        if self.__version__cache is None:
            # pylint: disable=no-member
            self.__version__cache = qiskit.version.__version__

        return self.__version__cache

    __qiskit_version__cache = None

    @property
    def __qiskit_version__(self):
        if self.__qiskit_version__cache is None:
            # pylint: disable=no-member
            self.__qiskit_version__cache = qiskit.version.__qiskit_version__

        return self.__qiskit_version__cache


sys.modules[__name__] = QiskitBaseModule()

# Please note these are global instances, not modules.
from qiskit.providers.basicaer import BasicAer  # noqa
sys.modules[__name__].BasicAer = BasicAer

# Try to import the Aer provider if installed.
try:
    from qiskit.providers.aer import Aer
    sys.modules[__name__].Aer = Aer
except ImportError:
    warnings.warn('Could not import the Aer provider from the qiskit-aer '
                  'package. Install qiskit-aer or check your installation.',
                  RuntimeWarning)
# Try to import the IBMQ provider if installed.
try:
    from qiskit.providers.ibmq import IBMQ
    sys.modules[__name__].IBMQ = IBMQ
except ImportError:
    warnings.warn('Could not import the IBMQ provider from the '
                  'qiskit-ibmq-provider package. Install qiskit-ibmq-provider '
                  'or check your installation.',
                  RuntimeWarning)
