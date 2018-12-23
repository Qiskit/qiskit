# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=wrong-import-order
# pylint: disable=redefined-builtin

"""Main Qiskit public functionality."""

import os
import pkgutil

# First, check for required Python and API version
from . import _util

# qiskit errors operator
from .qiskiterror import QiskitError, QISKitError

# The main qiskit operators
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from .tools.compiler import (compile, execute)

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions.standard
import qiskit.extensions.quantum_initializer
import qiskit.circuit.measure
import qiskit.circuit.reset

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code AND *before*
# importing the package you want to allow extensions for (in this case `backends`).
__path__ = pkgutil.extend_path(__path__, __name__)

# Please note these are global instances, not modules.
from qiskit.providers.ibmq import IBMQ
from qiskit.providers.builtinsimulators import BasicAer
from qiskit.providers.legacysimulators import LegacySimulators

# Try to import the Aer provider if th Aer element is installed.
try:
    from qiskit.providers.aer import Aer
except ImportError:
    pass

# TODO: Remove
from .wrapper._wrapper import (load_qasm_string, load_qasm_file)

# Import the wrapper, to make it available when doing "import qiskit".
from . import wrapper
from . import tools

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r") as version_file:
    __version__ = version_file.read().strip()
