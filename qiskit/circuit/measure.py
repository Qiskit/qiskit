# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum measurement in the computational basis.
"""
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class Measure(Instruction):
    """Quantum measurement in the computational basis."""

    def __init__(self):
        """Create new measurement instruction."""
        super().__init__("measure", 1, 1, [])

    def broadcast_arguments(self, qargs, cargs):
        qarg = qargs[0]
        carg = cargs[0]

        if len(carg) == len(qarg):
            for qarg, carg in zip(qarg, carg):
                yield [qarg], [carg]
        elif len(qarg) == 1 and carg:
            for each_carg in carg:
                yield qarg, [each_carg]
        else:
            raise CircuitError('register size error')


def measure(self, qubit, cbit):
    """Measure quantum bit into classical bit (tuples).

    Args:
        qubit (QuantumRegister|list|tuple): quantum register
        cbit (ClassicalRegister|list|tuple): classical register

    Returns:
        qiskit.Instruction: the attached measure instruction.

    Raises:
        CircuitError: if qubit is not in this circuit or bad format;
            if cbit is not in this circuit or not creg.
    """
    return self.append(Measure(), [qubit], [cbit])


QuantumCircuit.measure = measure


class PauliMeasure(Measure):
    """Perform a measurement with preceding basis change operations."""

    def __init__(self, basis):
        """Create a new basis transformation measurement.

        Args:
            basis (str): The target measurement basis, can be 'X', 'Y' or 'Z'.

        Raises:
            ValueError: If an unsupported basis is specified.
        """
        super().__init__()

        pre_rotation, post_rotation = None, None

        if basis.lower() == 'x':
            self.name = 'x_measure'
            from qiskit.extensions.standard import HGate
            pre_rotation = [HGate()]
        elif basis.lower() == 'y':
            self.name = 'y_measure'
            from qiskit.extensions.standard import HGate, SdgGate
            # since measure and S commute, S and Sdg cancel each other
            pre_rotation = [SdgGate(), HGate()]
            post_rotation = [HGate(), SdgGate()]
        elif basis.lower() == 'z':
            pre_rotation = []
        else:
            raise ValueError('Unsupported measurement basis choose either of X, Y or Z.')

        # default post rotation is the inverse of the pre rotations
        if post_rotation is None:
            post_rotation = [gate.inverse() for gate in reversed(pre_rotation)]

        self.basis = basis
        self.pre_rotation = pre_rotation
        self.post_rotation = post_rotation

    def _define(self):
        definition = []
        q = QuantumRegister(1, 'q')
        c = ClassicalRegister(1, 'c')

        # switch to the measurement basis
        for gate in self.pre_rotation:
            definition.append((gate, [q[0]], []))

        # measure
        definition.append((Measure(), [q[0]], [c[0]]))

        # apply inverse basis transformation for correct post-measurement state
        for gate in self.post_rotation:
            definition.append(gate, [q[0]], [])

        self.definition = definition


def pauli_measure(self, basis, qubit, cbit):
    """Measure in the Pauli-X basis."""
    return self.append(PauliMeasure(basis=basis), [qubit], [cbit])


QuantumCircuit.pauli_measure = pauli_measure
