# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Simulator command to toggle noise off or on.
"""
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import Instruction
from qiskit.extensions._extensionerror import ExtensionError
from qiskit.extensions.standard import header  # pylint: disable=unused-import


class Noise(Instruction):
    """Simulator noise operation."""

    def __init__(self, switch, qubits, circ):
        """Create new noise instruction."""
        super().__init__("noise", [switch], list(qubits), [], circ)

    def inverse(self):
        """Special case. Return self."""
        return self

    def reapply(self, circ):
        """Reapply this instruction to corresponding qubits in circ."""
        self._modifiers(circ.noise(self.param[0]))


def noise(self, switch):
    """Turn noise on/off in simulator.
    Works on all qubits, and prevents reordering (like barrier).

    Args:
        switch (int): turn noise on (1) or off (0)

    Returns:
        QuantumCircuit: with attached command

    Raises:
        ExtensionError: malformed command
    """
    tuples = []
    if isinstance(self, QuantumCircuit):
        for register in self.qregs:
            tuples.append(register)
    if not tuples:
        raise ExtensionError("no qubits for noise")
    if switch is None:
        raise ExtensionError("no noise switch passed")
    qubits = []
    for tuple_element in tuples:
        if isinstance(tuple_element, QuantumRegister):
            for j in range(tuple_element.size):
                self._check_qubit((tuple_element, j))
                qubits.append((tuple_element, j))
        else:
            self._check_qubit(tuple_element)
            qubits.append(tuple_element)
    self._check_dups(qubits)
    return self._attach(Noise(switch, qubits, self))


# Add to QuantumCircuit class
QuantumCircuit.noise = noise
