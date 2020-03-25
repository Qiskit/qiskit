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
from numpy import pi
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


def measure_z(self, qubit, cbit):
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


class BasisTransformationMeasurement(Instruction):
    """Perform a measurement with preceding basis change operations."""

    def __init__(self, measurement_basis=None, basis_transformation=None):
        """Create a new basis transformation measurement.

        Args:
            measurement_basis (str): The target measurement basis, can be 'x', 'y' or 'z'.
            basis_transformation (Instruction | list): An Instruction (or list of) to be applied
                before the measurement.

        Raises:
            ValueError: If either both or none of measurement_basis and basis_transformation are
                specified.
            ValueError: If an unsupported measurement_basis is specified.
        """
        super().__init__('generalized_measure', 1, 1, [])
        if (measurement_basis and basis_transformation) \
                or (not measurement_basis and not basis_transformation):
            raise ValueError('Provide either a target measurement basis or a transformation.')

        if measurement_basis:
            if measurement_basis.lower() == 'x':
                from qiskit.extensions.standard import HGate
                basis_transformation = [HGate()]
            elif measurement_basis.lower() == 'y':
                from qiskit.extensions.standard import RXGate
                basis_transformation = [RXGate(pi / 2)]
            elif measurement_basis.lower() == 'z':
                basis_transformation = []
            else:
                raise ValueError('Unsupported measurement basis choose either of x, y or z.')

        self.basis_transformation = basis_transformation

    def broadcast_arguments(self, qargs, cargs):
        """Broadcast the qubit and classical arguments."""
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

    def _define(self):
        definition = []
        q = QuantumRegister(1, 'q')
        c = ClassicalRegister(1, 'c')
        for gate in self.basis_transformation:
            definition.append((gate, [q[0]], []))
        definition.append((Measure(), [q[0]], [c[0]]))

        self.definition = definition


def measure_x(self, qubit, cbit):
    """Measure in the Pauli-X basis."""
    return self.append(BasisTransformationMeasurement(measurement_basis='x'), [qubit], [cbit])


def measure_y(self, qubit, cbit):
    """Measure in the Pauli-Y basis."""
    return self.append(BasisTransformationMeasurement(measurement_basis='y'), [qubit], [cbit])


QuantumCircuit.measure_x = measure_x
QuantumCircuit.measure_y = measure_y
QuantumCircuit.measure_z = measure_z
QuantumCircuit.measure = measure_z  # keep Z measure as default
