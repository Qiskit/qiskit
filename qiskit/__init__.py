# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=wrong-import-order


"""Main Qiskit public functionality."""

import pkgutil

# First, check for required Python and API version
from . import util

# qiskit errors operator
from .exceptions import QiskitError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import QuantumCircuit
# pylint: disable=redefined-builtin
from qiskit.tools.compiler import compile  # TODO remove after 0.8
from qiskit.execute import execute
from qiskit.compiler.transpiler import transpile
from qiskit.compiler.assembler import assemble

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

# Try to import the Aer provider if installed.
try:
    from qiskit.providers.aer import Aer
except ImportError:
    pass
# Try to import the IBMQ provider if installed.
try:
    from qiskit.providers.ibmq import IBMQ
except ImportError:
    pass

from .version import __version__
from .version import __qiskit_version__
