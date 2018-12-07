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

from qiskit.circuit import ClassicalRegister
from qiskit.circuit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit import CompositeGate
from qiskit.circuit import Instruction
from qiskit.circuit import InstructionSet
from qiskit.circuit import Reset
from qiskit.circuit import Measure

from ._qiskiterror import QiskitError, QISKitError
from ._schema_validation import (validate_json_against_schema,
                                 SchemaValidationError)
from .result import Result
from ._pubsub import Publisher, Subscriber

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions.standard
import qiskit.extensions.quantum_initializer

# Please note these are global instances, not modules.
from qiskit.backends.ibmq import IBMQ
from qiskit.backends.aer import Aer  # pylint: disable=invalid-name

# Allow extending this namespace. Please note that currently this line needs
# to be placed *before* the wrapper imports or any non-import code.
__path__ = pkgutil.extend_path(__path__, __name__)

from .wrapper._wrapper import (load_qasm_string, load_qasm_file)
from .tools._compiler import (compile, execute)

# Import the wrapper, to make it available when doing "import qiskit".
from . import wrapper
from . import tools

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r") as version_file:
    __version__ = version_file.read().strip()
