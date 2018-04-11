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
Composite gate, a container for a sequence of unitary gates.
"""
from ._gate import Gate
from ._qiskiterror import QISKitError


class CompositeGate(Gate):
    """Composite gate, a sequence of unitary gates."""

    def __init__(self, name, param, args, circuit=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters
        arg = list of pairs (Register, index)
        circ = QuantumCircuit or CompositeGate containing this gate
        """
        super().__init__(name, param, args, circuit)
        self.data = []  # gate sequence defining the composite unitary
        self.inverse_flag = False

    def instruction_list(self):
        """Return a list of instructions for this CompositeGate.

        If the CompositeGate itself contains composites, call
        this method recursively.
        """
        instruction_list = []
        for instruction in self.data:
            if isinstance(instruction, CompositeGate):
                instruction_list.extend(instruction.instruction_list())
            else:
                instruction_list.append(instruction)
        return instruction_list

    def has_register(self, register):
        """Test if this gate's circuit has the register r."""
        self.check_circuit()
        return self.circuit.has_register(register)

    def _modifiers(self, gate):
        """Apply any modifiers of this gate to another composite g."""
        if self.inverse_flag:
            gate.inverse()
        super()._modifiers(gate)

    def _attach(self, gate):
        """Attach a gate."""
        self.data.append(gate)
        return gate

    def _check_qubit(self, qubit):
        """Raise exception if q is not an argument or not qreg in circuit."""
        self.check_circuit()
        self.circuit._check_qubit(qubit)
        if (qubit[0].name, qubit[1]) not in map(
                lambda x: (x[0].name, x[1]), self.arg):
            raise QISKitError("qubit '%s[%d]' not argument of gate"
                              % (qubit[0].name, qubit[1]))

    def _check_qreg(self, register):
        """Raise exception.

        if quantum register is not in this gate's circuit.
        """
        self.check_circuit()
        self.circuit._check_qreg(register)

    def _check_creg(self, register):
        """Raise exception.

        if classical register is not in this gate's circuit.
        """
        self.check_circuit()
        self.circuit._check_creg(register)

    def _check_dups(self, qubits):
        """Raise exception.

        if list of qubits contains duplicates.
        """
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise QISKitError("duplicate qubit arguments")

    def qasm(self):
        """Return OPENQASM string."""
        return "\n".join([g.qasm() for g in self.data])

    def inverse(self):
        """Invert this gate."""
        self.data = [gate.inverse() for gate in reversed(self.data)]
        self.inverse_flag = not self.inverse_flag
        return self

    def q_if(self, *qregs):
        """Add controls to this gate."""
        self.data = [gate.q_if(qregs) for gate in self.data]
        return self

    def c_if(self, classical, val):
        """Add classical control register."""
        self.data = [gate.c_if(classical, val) for gate in self.data]
        return self
