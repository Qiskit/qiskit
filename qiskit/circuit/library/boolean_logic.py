# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Implementations of boolean logic quantum circuits.
"""

from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


def permutation(n_qubits: int,
                pattern: Optional[List[int]] = None,
                seed: Optional[int] = None) -> QuantumCircuit:
    """Return an n_qubit circuit that permutes qubits using SWAPs.

    Args:
        n_qubits: circuit width.
        pattern: permutation pattern. If None, permute randomly.
        seed: random seed in case a random permutation is requested.

    Returns:
        A permutation circuit.

    Raises:
        CircuitError: if permutation pattern is malformed.
    """
    circuit = QuantumCircuit(n_qubits, name="permutation")

    if pattern is not None:
        if sorted(pattern) != list(range(n_qubits)):
            raise CircuitError("Permutation pattern must be some "
                               "ordering of 0..n_qubits-1 in a list.")
        pattern = np.array(pattern)
    else:
        rng = np.random.RandomState(seed)
        pattern = np.arange(n_qubits)
        rng.shuffle(pattern)

    for i in range(n_qubits):
        if (pattern[i] != -1) and (pattern[i] != i):
            circuit.swap(i, int(pattern[i]))
            pattern[pattern[i]] = -1

    return circuit


def shift(n_qubits: int, shift: int) -> QuantumCircuit:
    """Return a circuit implementing bitwise xor (shift over Z_2).

    Args:
        n_qubits: the width of circuit.
        shift: shift amount in decimal form.

    Returns:
        A boolean shift circuit.

    Raises:
        CircuitError: if the shift amount exceeds available qubits.
    """
    if len(bin(shift)[2:]) > n_qubits:
        raise Exception("Bits in 'shift' exceed circuit width")

    circuit = QuantumCircuit(n_qubits, name="shift")

    for i in range(n_qubits):
        bit = shift & 1
        shift = shift >> 1
        if bit == 1:
            circuit.x(i)

    return circuit


def inner_product(n_qubits: int) -> QuantumCircuit:
    """Return a circuit to compute the inner product of 2 n-qubit registers.

    Args:
        n_qubits: width of top and bottom registers (half total circuit width)

    Returns:
        A circuit computing inner product of two registers.
    """
    qr_a = QuantumRegister(n_qubits)
    qr_b = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qr_a, qr_b, name="inner_product")

    for i in range(n_qubits):
        circuit.cz(i, i + n_qubits)

    return circuit
