# -*- coding: utf-8 -*-

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

"""
Quantum computer instruction.
"""
from sympy import Number, Basic

from ._qiskiterror import QISKitError
from ._register import Register


class Instruction(object):
    """Generic quantum computer instruction."""

    def __init__(self, name, param, arg, circuit=None):
        """Create a new instruction.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circuit = QuantumCircuit or CompositeGate containing this instruction
        """
        for i in arg:
            if not isinstance(i[0], Register):
                raise QISKitError("argument not (Register, int) tuple")
        self.name = name
        self.param = []
        for single_param in param:
            if not isinstance(single_param, (Basic, complex)):
                # If the item in param is not symbolic and not complex (used
                # by InitializeGate), make it symbolic.
                self.param.append(Number(single_param))
            else:
                self.param.append(single_param)
        self.arg = arg
        self.control = None  # tuple (ClassicalRegister, int) for "if"
        self.circuit = circuit

    def check_circuit(self):
        """Raise exception if self.circuit is None."""
        if self.circuit is None:
            raise QISKitError("Instruction's circuit not assigned")

    def c_if(self, classical, val):
        """Add classical control on register clasical and value val."""
        self.check_circuit()
        self.circuit._check_creg(classical)
        if val < 0:
            raise QISKitError("control value should be non-negative")
        self.control = (classical, val)
        return self

    def _modifiers(self, gate):
        """Apply any modifiers of this instruction to another one."""
        if self.control is not None:
            self.check_circuit()
            if not gate.circuit.has_register(self.control[0]):
                raise QISKitError("control register %s not found"
                                  % self.control[0].name)
            gate.c_if(self.control[0], self.control[1])

    def _qasmif(self, string):
        """Print an if statement if needed."""
        if self.control is None:
            return string
        return "if(%s==%d) " % (self.control[0].name, self.control[1]) + string
