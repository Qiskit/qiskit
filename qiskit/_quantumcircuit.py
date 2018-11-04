# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=cyclic-import

"""
Quantum circuit object.
"""
import itertools
from collections import OrderedDict

from qiskit.qasm import _qasm
from qiskit.unrollers import _unroller
from qiskit.unrollers import _circuitbackend
from ._qiskiterror import QISKitError
from ._quantumregister import QuantumRegister
from ._classicalregister import ClassicalRegister


def _circuit_from_qasm(qasm, basis=None):
    default_basis = ["id", "u0", "u1", "u2", "u3", "x", "y", "z", "h", "s",
                     "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cy", "cz",
                     "ch", "crz", "cu1", "cu3", "swap", "ccx", "cswap"]
    if not basis:
        basis = default_basis

    ast = qasm.parse()
    unroll = _unroller.Unroller(
        ast, _circuitbackend.CircuitBackend(basis))
    circuit = unroll.execute()
    return circuit


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

    @staticmethod
    def from_qasm_file(path):
        """Take in a QASM file and generate a QuantumCircuit object.

        Args:
          path (str): Path to the file for a QASM program
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = _qasm.Qasm(filename=path)
        return _circuit_from_qasm(qasm)

    @staticmethod
    def from_qasm_str(qasm_str):
        """Take in a QASM string and generate a QuantumCircuit object.

        Args:
          qasm_str (str): A QASM program string
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = _qasm.Qasm(data=qasm_str)
        return _circuit_from_qasm(qasm)

    def __init__(self, *regs, name=None):
        """Create a new circuit.

        A circuit is a list of instructions bound to some registers.

        Args:
            *regs (Registers): registers to include in the circuit.
            name (str or None): the name of the quantum circuit. If
                None, an automatically generated string will be assigned.

        Raises:
            QISKitError: if the circuit name, if given, is not valid.
        """
        if name is None:
            name = self.cls_prefix() + str(self.cls_instances())
        self._increment_instances()

        if not isinstance(name, str):
            raise QISKitError("The circuit name should be a string "
                              "(or None to auto-generate a name).")

        self.name = name

        # Data contains a list of instructions in the order they were applied.
        self.data = []

        # This is a map of registers bound to this circuit, by name.
        self.qregs = OrderedDict()
        self.cregs = OrderedDict()
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

        Args:
            register (Register): a quantum or classical register.

        Returns:
            bool: True if the register is contained in this circuit.
        """
        has_reg = False
        if (isinstance(register, QuantumRegister) and
                register in self.qregs.values()):
            has_reg = True
        elif (isinstance(register, ClassicalRegister) and
              register in self.cregs.values()):
            has_reg = True
        return has_reg

    def combine(self, rhs):
        """
        Append rhs to self if self contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Return self + rhs as a new object.
        """
        # Check registers in LHS are compatible with RHS
        self._check_compatible_regs(rhs)

        # Make new circuit with combined registers
        combined_qregs = {**self.qregs, **rhs.qregs}.values()
        combined_cregs = {**self.cregs, **rhs.cregs}.values()
        circuit = QuantumCircuit(*combined_qregs, *combined_cregs)
        for gate in itertools.chain(self.data, rhs.data):
            gate.reapply(circuit)
        return circuit

    def extend(self, rhs):
        """
        Append rhs to self if self contains compatible registers.

        Two circuits are compatible if they contain the same registers
        or if they contain different registers with unique names. The
        returned circuit will contain all unique registers between both
        circuits.

        Modify and return self.
        """
        # Check registers in LHS are compatible with RHS
        self._check_compatible_regs(rhs)

        # Add new registers
        self.qregs.update(rhs.qregs)
        self.cregs.update(rhs.cregs)

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

    def _attach(self, instruction):
        """Attach an instruction."""
        self.data.append(instruction)
        return instruction

    def add(self, *regs):
        """Add registers."""
        for register in regs:
            if register.name in self.qregs or register.name in self.cregs:
                raise QISKitError("register name \"%s\" already exists"
                                  % register.name)
            if isinstance(register, QuantumRegister):
                self.qregs[register.name] = register
            elif isinstance(register, ClassicalRegister):
                self.cregs[register.name] = register
            else:
                raise QISKitError("expected a register")

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

    def _check_compatible_regs(self, rhs):
        """Raise exception if the circuits are defined on incompatible registers"""
        lhs_regs = {**self.qregs, **self.cregs}
        rhs_regs = {**rhs.qregs, **rhs.cregs}
        common_registers = lhs_regs.keys() & rhs_regs.keys()
        for name in common_registers:
            if lhs_regs[name] != rhs_regs[name]:
                raise QISKitError("circuits are not compatible")

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
        for register in self.qregs.values():
            string_temp += register.qasm() + "\n"
        for register in self.cregs.values():
            string_temp += register.qasm() + "\n"
        for instruction in self.data:
            string_temp += instruction.qasm() + "\n"
        return string_temp
