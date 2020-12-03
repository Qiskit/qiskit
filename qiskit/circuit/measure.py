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
Quantum measurement
"""
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class Measure(Instruction):
    """Quantum measurement"""

    # pylint: disable=dangerous-default-value
    def __init__(self, name='measure', num_qubits=1, num_clbits=1, params=[]):
        """Create new measurement instruction."""
        super().__init__(name, num_qubits, num_clbits, params=params)

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

    def __init__(self, basis, num_qubits, num_clbits):
        """Create a new basis transformation measurement.

        Args:
            basis (str): The target measurement basis,
                         consists of the characters 'X', 'Y', and 'Z'.
            num_qubits (integer): number of qubits to measure.
            num_clbits (integer): number of classical bits.

        Raises:
            ValueError: If an unsupported basis is specified,
                        or in case of mismatch between the number of qubits,
                        classical bits, and basis length.
        """

        params = []
        for qubit_basis in basis.upper():
            if qubit_basis not in ['X', 'Y', 'Z']:
                raise ValueError('Unsupported measurement basis, choose either of X, Y, or Z.')
            params.append(qubit_basis)

        if num_qubits != num_clbits or num_qubits != len(basis):
            raise ValueError('Mismatch between the number of qubits, \
            classical bits, and basis length.')

        super().__init__('measure_pauli', num_qubits, num_clbits, params)

    def broadcast_arguments(self, qargs, cargs):
        yield [qarg[0] for qarg in qargs], \
              [carg[0] for carg in cargs]

    def _define(self):
        definition = []
        q = QuantumRegister(self.num_qubits, 'q')
        c = ClassicalRegister(self.num_clbits, 'c')

        # pylint: disable=cyclic-import
        from .library import HGate, SGate, SdgGate

        for i, qubit_basis in enumerate(self.params):
            if qubit_basis == 'X':
                pre_rotation = post_rotation = [HGate()]
            elif qubit_basis == 'Y':
                # since measure and S commute, S and Sdg cancel each other
                pre_rotation = [SdgGate(), HGate()]
                post_rotation = [HGate(), SGate()]
            else:  # Z
                pre_rotation = post_rotation = []

            # switch to the measurement basis
            for gate in pre_rotation:
                definition += [(gate, [q[i]], [])]

            # measure
            definition += [(Measure(), [q[i]], [c[i]])]

            # apply inverse basis transformation for correct post-measurement state
            for gate in post_rotation:
                definition += [(gate, [q[i]], [])]

        qc = QuantumCircuit(q, c)
        qc._data = definition
        self.definition = qc


def measure_pauli(self, basis, qubits, clbits):
    """Measure in the the Pauli X, Y, or Z basis."""
    return self.append(MeasurePauli(basis, len(qubits), len(clbits)),
                       qubits, clbits)


QuantumCircuit.measure_pauli = measure_pauli
