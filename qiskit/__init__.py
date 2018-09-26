# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=wrong-import-order
# pylint: disable=redefined-builtin

"""Main QISKit public functionality."""

import os
import pkgutil

# First, check for required Python and API version
from . import _util

from ._qiskiterror import QISKitError
from ._classicalregister import ClassicalRegister
from ._quantumregister import QuantumRegister
from ._quantumcircuit import QuantumCircuit
from ._gate import Gate
from ._compositegate import CompositeGate
from ._instruction import Instruction
from ._instructionset import InstructionSet
from ._reset import Reset
from ._measure import Measure
from ._schema_validation import (validate_json_against_schema,
                                 SchemaValidationError)
from .result import Result

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions.standard
import qiskit.extensions.quantum_initializer

import qiskit.backends.ibmq
import qiskit.backends.local

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code.
__path__ = pkgutil.extend_path(__path__, __name__)

from .wrapper._wrapper import (compile, execute, load_qasm_string,
                               load_qasm_file, least_busy, qobj_to_circuits)

# To be deprecated methods
from .wrapper._wrapper import (available_backends, get_backend, register,
                               unregister, registered_providers)


# Import the wrapper, to make it available when doing "import qiskit".
from . import wrapper

# Global objects. They are assigned to the specific module-level variables to
# minimize the risk of them falling out of sync - please use with care.
IBMQ = qiskit.backends.ibmq.IBMQ
Aer = qiskit.backends.local.Aer  # pylint: disable=invalid-name

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r") as version_file:
    __version__ = version_file.read().strip()
