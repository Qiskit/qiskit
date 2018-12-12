# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Composite gate, a container for a sequence of unitary gates.
"""
from qiskit.qiskiterror import QiskitError
from .gate import Gate


class CompositeGate(Gate):  # pylint: disable=abstract-method
    """Composite gate, a sequence of unitary gates."""

    def __init__(self, name, param, qargs, circuit=None, inverse_name=None):
        """Create a new composite gate.

        name = instruction name string
        param = list of real parameters
        qarg = list of pairs (QuantumRegister, index)
        circ = QuantumCircuit or CompositeGate containing this gate
        """
        super().__init__(name, param, qargs, circuit)
        self.data = []  # gate sequence defining the composite unitary
        self.inverse_flag = False
        self.inverse_name = inverse_name or (name + 'dg')

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
                lambda x: (x[0].name, x[1]), self.qargs):
            raise QiskitError("qubit '%s[%d]' not argument of gate"
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
            raise QiskitError("duplicate qubit arguments")

    def qasm(self):
        """Return OPENQASM string."""
        return "\n".join([g.qasm() for g in self.data])

    def inverse(self):
        """Invert this gate."""
        self.data = [gate.inverse() for gate in reversed(self.data)]
        self.inverse_flag = not self.inverse_flag
        return self

    def reapply(self, circ):
        """Reapply this gate to corresponding qubits in circ."""
        for gate in self.data:
            gate.reapply(circ)

    def q_if(self, *qregs):
        """Add controls to this gate."""
        self.data = [gate.q_if(qregs) for gate in self.data]
        return self

    def c_if(self, classical, val):
        """Add classical control register."""
        self.data = [gate.c_if(classical, val) for gate in self.data]
        return self
