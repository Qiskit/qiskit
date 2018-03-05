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
Unitary gate.
"""
from ._instruction import Instruction
from ._quantumregister import QuantumRegister
from ._qiskiterror import QISKitError


class Gate(Instruction):
    """Unitary gate."""

    def __init__(self, name, param, args, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters (will converted to symbolic)
        arg = list of pairs (Register, index)
        circuit = QuantumCircuit or CompositeGate containing this gate
        """
        for argument in args:
            if not isinstance(argument[0], QuantumRegister):
                raise QISKitError("argument not (QuantumRegister, int) "
                                  + "tuple")
        super().__init__(name, param, args, circuit)

    def inverse(self):
        """Invert this gate."""
        raise QISKitError("inverse not implemented")

    def q_if(self, *qregs):
        """Add controls to this gate."""
        # pylint: disable=unused-argument
        raise QISKitError("control not implemented")
