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
Quantum circuit object.
"""
import itertools
from collections import OrderedDict
from ._qiskiterror import QISKitError
from ._register import Register
from ._quantumregister import QuantumRegister
from ._classicalregister import ClassicalRegister
from ._measure import Measure
from ._reset import Reset
from ._instructionset import InstructionSet


class QuantumCircuit(object):
    """Quantum circuit."""

    # Class variable OPENQASM header
    header = "OPENQASM 2.0;"

    # Class variable with gate definitions
    # This is a dict whose values are dicts with the
    # following keys:
    #   "print" = True or False
    #   "opaque" = True or False
    #   "n_args" = number of real parameters
    #   "n_bits" = number of qubits
    #   "args"   = list of parameter names
    #   "bits"   = list of qubit names
    #   "body"   = GateBody AST node
    definitions = OrderedDict()

    def __init__(self, *regs, name=None):
        """Create a new circuit."""
        self.name = name
        # Data contains a list of instructions in the order they were applied.
        self.data = []
        # This is a map of registers bound to this circuit, by name.
        self.regs = OrderedDict()
        self.add(*regs)

    def has_register(self, register):
        """
        Test if this circuit has the register r.

        Return True or False.
        """
        if register.name in self.regs:
            registers = self.regs[register.name]
            if registers.size == register.size:
                if ((isinstance(register, QuantumRegister) and
                     isinstance(registers, QuantumRegister)) or
                        (isinstance(register, ClassicalRegister) and
                         isinstance(registers, ClassicalRegister))):
                    return True
        return False

    def get_qregs(self):
        """Get the qregs from the registers."""
        qregs = {}
        for name, register in self.regs.items():
            if isinstance(register, QuantumRegister):
                qregs[name] = register
        return qregs

    def get_cregs(self):
        """Get the cregs from the registers."""
        cregs = {}
        for name, register in self.regs.items():
            if isinstance(register, ClassicalRegister):
                cregs[name] = register
        return cregs

    def combine(self, rhs):
        """
        Append rhs to self if self contains rhs's registers.

        Return self + rhs as a new object.
        """
        if self.name is not None and rhs.name is not None:
            circuit_name = "c%sN%s" % (self.name, rhs.name)
        else:
            circuit_name = None
        for register in rhs.regs.values():
            if not self.has_register(register):
                raise QISKitError("circuits are not compatible")
        circuit = QuantumCircuit(
            *[register for register in self.regs.values()], name=circuit_name)
        for gate in itertools.chain(self.data, rhs.data):
            gate.reapply(circuit)
        return circuit

    def extend(self, rhs):
        """
        Append rhs to self if self contains rhs's registers.

        Modify and return self.
        """
        for register in rhs.regs.values():
            if not self.has_register(register):
                raise QISKitError("circuits are not compatible")
        for gate in rhs.data:
            gate.reapply(self)
        return self

    def __add__(self, rhs):
        """Overload + to implement self.concatenate."""
        return self.combine(rhs)

    def __iadd__(self, rhs):
        """Overload += to implement self.extend."""
        return self.extend(rhs)

    def __len__(self):
        """Return number of operations in circuit."""
        return len(self.data)

    def __getitem__(self, item):
        """Return indexed operation."""
        return self.data[item]

    def _attach(self, gate):
        """Attach a gate."""
        self.data.append(gate)
        return gate

    def add(self, *regs):
        """Add registers."""
        for register in regs:
            if not isinstance(register, Register):
                raise QISKitError("expected a register")
            if register.name not in self.regs:
                self.regs[register.name] = register
            else:
                raise QISKitError("register name \"%s\" already exists"
                                  % register.name)

    def _check_qreg(self, register):
        """Raise exception if r is not in this circuit or not qreg."""
        if not isinstance(register, QuantumRegister):
            raise QISKitError("expected quantum register")
        if not self.has_register(register):
            raise QISKitError(
                "register '%s' not in this circuit" %
                register.name)

    def _check_qubit(self, qubit):
        """Raise exception if qubit is not in this circuit or bad format."""
        self._check_qreg(qubit[0])
        qubit[0].check_range(qubit[1])

    def _check_creg(self, register):
        """Raise exception if r is not in this circuit or not creg."""
        if not isinstance(register, ClassicalRegister):
            raise QISKitError("expected classical register")
        if not self.has_register(register):
            raise QISKitError(
                "register '%s' not in this circuit" %
                register.name)

    def _check_dups(self, qubits):
        """Raise exception if list of qubits contains duplicates."""
        squbits = set(qubits)
        if len(squbits) != len(qubits):
            raise QISKitError("duplicate qubit arguments")

    def _gate_string(self, name):
        """Return a QASM string for the named gate."""
        out = ""
        if self.definitions[name]["opaque"]:
            out = "opaque " + name
        else:
            out = "gate " + name
        if self.definitions[name]["n_args"] > 0:
            out += "(" + ",".join(self.definitions[name]["args"]) + ")"
        out += " " + ",".join(self.definitions[name]["bits"])
        if self.definitions[name]["opaque"]:
            out += ";"
        else:
            out += "\n{\n" + self.definitions[name]["body"].qasm() + "}"
        return out

    def qasm(self):
        """Return OPENQASM string."""
        string = self.header + "\n"
        for gate_name in self.definitions:
            if self.definitions[gate_name]["print"]:
                string += self._gate_string(gate_name)
        for register in self.regs.values():
            string += register.qasm() + "\n"
        for instruction in self.data:
            string += instruction.qasm() + "\n"
        return string

    def measure(self, qubit, cbit):
        """Measure quantum bit into classical bit (tuples).

        Returns:
            Gate: the attached measure gate.

        Raises:
            QISKitError: if qubit is not in this circuit or bad format;
                if cbit is not in this circuit or not creg.
        """
        if isinstance(qubit, QuantumRegister) and \
           isinstance(cbit, ClassicalRegister) and len(qubit) == len(cbit):
            instructions = InstructionSet()
            for i in range(qubit.size):
                instructions.add(self.measure((qubit, i), (cbit, i)))
            return instructions

        self._check_qubit(qubit)
        self._check_creg(cbit[0])
        cbit[0].check_range(cbit[1])
        return self._attach(Measure(qubit, cbit, self))

    def reset(self, quantum_register):
        """Reset q."""
        if isinstance(quantum_register, QuantumRegister):
            instructions = InstructionSet()
            for sizes in range(quantum_register.size):
                instructions.add(self.reset((quantum_register, sizes)))
            return instructions
        self._check_qubit(quantum_register)
        return self._attach(Reset(quantum_register, self))
