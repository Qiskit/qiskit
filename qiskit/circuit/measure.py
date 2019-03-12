# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Quantum measurement in the computational basis.
"""
from qiskit.circuit.decorators import _op_expand
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit


class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    def __init__(self, circuit=None):
        """Create new measurement instruction."""
        super().__init__("measure", 1, 1, [], circuit, is_reversible=False)


@_op_expand(2, broadcastable=[True, False])
def measure(self, qubit, cbit):
    """Measure quantum bit into classical bit (tuples).

    Args:
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.Instruction: the attached measure instruction.

    Raises:
        QiskitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    return self.append(Measure(self), [qubit], [cbit])


QuantumCircuit.measure = measure
