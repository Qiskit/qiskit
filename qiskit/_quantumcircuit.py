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
    instances = 0
    prefix = 'circuit'

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
        """Create a new circuit.

        Args:
            *regs (Registers): registers to include in the circuit.
            name (str or None): the name of the quantum circuit. If
                None, an automatically generated identifier will be
                assigned.

        Raises:
            QISKitError: if the circuit name, if given, is not valid.
        """
        self._increment_instances()
        if name is None:
            name = self.cls_prefix() + str(self.cls_instances())

        if not isinstance(name, str):
            raise QISKitError("The circuit name should be a string "
                              "(or None for autogenerate a name).")

        self.name = name
        # Data contains a list of instructions in the order they were applied.
        self.data = []
        # This is a map of registers bound to this circuit, by name.
        self.regs = OrderedDict()
        self.add(*regs)

    @classmethod
    def _increment_instances(cls):
        cls.instances += 1

    @classmethod
    def cls_instances(cls):
        """Return the current number of instances of this class,
        useful for auto naming."""
        return cls.instances

    @classmethod
    def cls_prefix(cls):
        """Return the prefix to use for auto naming."""
        return cls.prefix

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
        qregs = OrderedDict()
        for name, register in self.regs.items():
            if isinstance(register, QuantumRegister):
                qregs[name] = register
        return qregs

    def get_cregs(self):
        """Get the cregs from the registers."""
        cregs = OrderedDict()
        for name, register in self.regs.items():
            if isinstance(register, ClassicalRegister):
                cregs[name] = register
        return cregs

    def combine(self, rhs):
        """
        Append rhs to self if self contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Return self + rhs as a new object.
        """
        combined_registers = []
        # Check registers in LHS are compatible with RHS
        for name, register in self.regs.items():
            if name in rhs.regs and register != rhs.regs[name]:
                raise QISKitError("circuits are not compatible")
            else:
                combined_registers.append(register)
        # Add registers in RHS not in LHS
        complement_registers = set(rhs.regs) - set(self.regs)
        for name in complement_registers:
            combined_registers.append(rhs.regs[name])
        # Make new circuit with combined registers
        circuit = QuantumCircuit(*combined_registers)
        for gate in itertools.chain(self.data, rhs.data):
            gate.reapply(circuit)
        return circuit

    def extend(self, rhs):
        """
        Append rhs to self if self if it contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Modify and return self.
        """
        # Check compatibility and add new registers
        for name, register in rhs.regs.items():
            if name not in self.regs:
                self.add(register)
            elif name in self.regs and register != self.regs[name]:
                raise QISKitError("circuits are not compatible")

        # Add new gates
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
        if not isinstance(qubit, tuple):
            raise QISKitError("%s is not a tuple."
                              "A qubit should be formated as a tuple." % str(qubit))
        if not len(qubit) == 2:
            raise QISKitError("%s is not a tuple with two elements, but %i instead" % len(qubit))
        if not isinstance(qubit[1], int):
            raise QISKitError("The second element of a tuple defining a qubit should be an int:"
                              "%s was found instead" % type(qubit[1]).__name__)
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
            out += "\n{\n" + self.definitions[name]["body"].qasm() + "}\n"
        return out

    def qasm(self):
        """Return OPENQASM string."""
        string_temp = self.header + "\n"
        for gate_name in self.definitions:
            if self.definitions[gate_name]["print"]:
                string_temp += self._gate_string(gate_name)
        for register in self.regs.values():
            string_temp += register.qasm() + "\n"
        for instruction in self.data:
            string_temp += instruction.qasm() + "\n"
        return string_temp

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
