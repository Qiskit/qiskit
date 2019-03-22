# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Composite gate, a container for a sequence of unitary gates.
"""
import warnings
from qiskit.exceptions import QiskitError
from .gate import Gate


class CompositeGate(Gate):  # pylint: disable=abstract-method
    """Composite gate, a sequence of unitary gates."""

    def __init__(self, name, params, inverse_name=None):
        """Create a new composite gate.

        name = instruction name string
        params = list of real parameters
        """
        warnings.warn('CompositeGate is deprecated and will be removed in v0.9. '
                      'Any Instruction can now be composed of other sub-instructions. '
                      'To build them, you construct a circuit then use '
                      'circuit.to_instruction().', DeprecationWarning)
        super().__init__(name, params)
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

    def append(self, gate):
        """Attach a gate."""
        self.data.append(gate)
        return gate

    def _attach(self, gate):
        """DEPRECATED after 0.8."""
        self.append(gate)

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

    def q_if(self, *qregs):
        """Add controls to this gate."""
        self.data = [gate.q_if(qregs) for gate in self.data]
        return self

    def c_if(self, classical, val):
        """Add classical control register."""
        self.data = [gate.c_if(classical, val) for gate in self.data]
        return self
