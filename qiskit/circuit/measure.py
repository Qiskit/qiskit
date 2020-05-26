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

    def __init__(self, name='measure'):
        """Create new measurement instruction."""
        super().__init__(name, 1, 1, [])

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


class MeasurePauli(Measure):
    """Perform a measurement with preceding basis change operations."""

    def __init__(self, basis):
        """Create a new basis transformation measurement.

        Args:
            basis (str): The target measurement basis, can be 'X', 'Y' or 'Z'.

        Raises:
            ValueError: If an unsupported basis is specified.
        """
        super().__init__(name='measure_' + basis)

        transformations = []

        from .library.standard_gates import HGate, SGate, SdgGate

        for qubit_basis in basis:
            if qubit_basis.lower() == 'x':
                pre_rotation = post_rotation = [HGate()]
            elif qubit_basis.lower() == 'y':
                # since measure and S commute, S and Sdg cancel each other
                pre_rotation = [SdgGate(), HGate()]
                post_rotation = [HGate(), SGate()]
            elif qubit_basis.lower() == 'z':
                pre_rotation = post_rotation = []
            else:
                raise ValueError('Unsupported measurement basis choose either of X, Y or Z.')

            transformations += [(pre_rotation, post_rotation)]

        self.basis = basis
        self.transformations = transformations

    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits, 'q')
        c = ClassicalRegister(self.num_clbits, 'c')

        for i, transformation in enumerate(self.transformations):
            pre_rotation, post_rotation = transformation

            # switch to the measurement basis
            for gate in pre_rotation:
                definition += [(gate, [q[i]], [])]

            # measure
            definition += [(Measure(), [q[i]], [c[i]])]

            # apply inverse basis transformation for correct post-measurement state
            for gate in post_rotation:
                definition += [(gate, [q[i]], [])]

        self.definition = definition


def measure_pauli(self, basis, qubit, cbit):
    """Measure in the Pauli-X basis."""
    # transform to list if they are not already
    # qubits = qubits if hasattr(qubits, '__len__') else [qubits]
    # cbits = cbits if hasattr(cbits, '__len__') else [cbits]

    # # if only one Pauli basis is specified, broadcast to all qubits
    # if len(basis) == 1:
    #     basis *= len(qubits)

    # if len(basis) != len(qubits):
    #     raise ValueError('Number of qubits does not match basis arguments.')

    return self.append(MeasurePauli(basis=basis), [qubit], [cbit])


QuantumCircuit.measure_pauli = measure_pauli
