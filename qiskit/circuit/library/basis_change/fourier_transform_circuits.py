# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum Fourier Transform Circuit.
"""

import numpy as np

from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit  # pylint: disable=unused-import

from qiskit.aqua import AquaError


class FourierTransformCircuits:
    """
    Quantum Fourier Transform Circuit.
    """
    @staticmethod
    def _do_swaps(circuit, qubits):
        num_qubits = len(qubits)
        for i in range(num_qubits // 2):
            circuit.cx(qubits[i], qubits[num_qubits - i - 1])
            circuit.cx(qubits[num_qubits - i - 1], qubits[i])
            circuit.cx(qubits[i], qubits[num_qubits - i - 1])

    @staticmethod
    def construct_circuit(
            circuit=None,
            qubits=None,
            inverse=False,
            approximation_degree=0,
            do_swaps=True
    ):
        """
        Construct the circuit representing the desired state vector.

        Args:
            circuit (QuantumCircuit): The optional circuit to extend from.
            qubits (Union(QuantumRegister, list[Qubit])): The optional qubits to construct
                the circuit with.
            approximation_degree (int): degree of approximation for the desired circuit
            inverse (bool): Boolean flag to indicate Inverse Quantum Fourier Transform
            do_swaps (bool): Boolean flag to specify if swaps should be included to align
                the qubit order of
                input and output. The output qubits would be in reversed order without the swaps.

        Returns:
            QuantumCircuit: quantum circuit
        Raises:
            AquaError: invalid input
        """

        if circuit is None:
            raise AquaError('Missing input QuantumCircuit.')

        if qubits is None:
            raise AquaError('Missing input qubits.')

        if isinstance(qubits, QuantumRegister):
            if not circuit.has_register(qubits):
                circuit.add_register(qubits)
        elif isinstance(qubits, list):
            for qubit in qubits:
                if isinstance(qubit, Qubit):
                    if not circuit.has_register(qubit.register):
                        circuit.add_register(qubit.register)
                else:
                    raise AquaError('A QuantumRegister or a list of qubits '
                                    'is expected for the input qubits.')
        else:
            raise AquaError('A QuantumRegister or a list of qubits '
                            'is expected for the input qubits.')

        if do_swaps and not inverse:
            FourierTransformCircuits._do_swaps(circuit, qubits)

        qubit_range = reversed(range(len(qubits))) if inverse else range(len(qubits))
        for j in qubit_range:
            neighbor_range = range(np.max([0, j - len(qubits) + approximation_degree + 1]), j)
            if inverse:
                neighbor_range = reversed(neighbor_range)
                circuit.u2(0, np.pi, qubits[j])
            for k in neighbor_range:
                lam = 1.0 * np.pi / float(2 ** (j - k))
                if inverse:
                    lam *= -1
                circuit.u1(lam / 2, qubits[j])
                circuit.cx(qubits[j], qubits[k])
                circuit.u1(-lam / 2, qubits[k])
                circuit.cx(qubits[j], qubits[k])
                circuit.u1(lam / 2, qubits[k])
            if not inverse:
                circuit.u2(0, np.pi, qubits[j])

        if do_swaps and inverse:
            FourierTransformCircuits._do_swaps(circuit, qubits)

        return circuit
