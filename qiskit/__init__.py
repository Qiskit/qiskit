# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Main QISKit public functionality."""

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

# The qiskit.extensions.x imports needs to be placed here due to the
# mechanism for adding gates dynamically.
import qiskit.extensions.standard
import qiskit.extensions.quantum_initializer

from ._jobprocessor import JobProcessor
from ._quantumjob import QuantumJob
from ._quantumprogram import QuantumProgram
from ._result import Result

from ._compiler import compile  # pylint: disable=redefined-builtin
from ._compiler import execute

__version__ = '0.5.0'
