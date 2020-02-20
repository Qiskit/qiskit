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

# pylint: disable=no-member

"""Implementations of boolean logic quantum circuits."""

from typing import List, Optional

import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError


class Permutation(QuantumCircuit):
    """An n_qubit circuit that permutes qubits."""

    def __init__(self,
                 n_qubits: int,
                 pattern: Optional[List[int]] = None,
                 seed: Optional[int] = None) -> QuantumCircuit:
        """Return an n_qubit permutation circuit implemented using SWAPs.

        Args:
            n_qubits: circuit width.
            pattern: permutation pattern. If None, permute randomly.
            seed: random seed in case a random permutation is requested.

        Returns:
            A permutation circuit.

        Raises:
            CircuitError: if permutation pattern is malformed.
        """
        super().__init__(n_qubits, name="permutation")

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
                self.swap(i, int(pattern[i]))
                pattern[pattern[i]] = -1


class XOR(QuantumCircuit):
    """An n_qubit circuit for bitwise xor-ing the input with some integer ``amount``.

    The ``amount`` is xor-ed in bitstring form with the input.

    This circuit can also represent addition by ``amount`` over the finite field GF(2).
    """

    def __init__(self,
                 n_qubits: int,
                 amount: Optional[int] = None,
                 seed: Optional[int] = None) -> QuantumCircuit:
        """Return a circuit implementing bitwise xor.

        Args:
            n_qubits: the width of circuit.
            amount: the xor amount in decimal form.
            seed: random seed in case a random xor is requested.

        Returns:
            A circuit for bitwise XOR.

        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.
        """
        super().__init__(n_qubits, name="xor")

        if amount is not None:
            if len(bin(amount)[2:]) > n_qubits:
                raise CircuitError("Bits in 'amount' exceed circuit width")
        else:
            rng = np.random.RandomState(seed)
            amount = rng.randint(0, 2**n_qubits)

        for i in range(n_qubits):
            bit = amount & 1
            amount = amount >> 1
            if bit == 1:
                self.x(i)


class InnerProduct(QuantumCircuit):
    """An n_qubit circuit that computes the inner product of two registers."""

    def __init__(self, n_qubits: int) -> QuantumCircuit:
        """Return a circuit to compute the inner product of 2 n-qubit registers.

        This implementation uses CZ gates.

        Args:
            n_qubits: width of top and bottom registers (half total circuit width)

        Returns:
            A circuit computing inner product of two registers.
        """
        qr_a = QuantumRegister(n_qubits)
        qr_b = QuantumRegister(n_qubits)
        super().__init__(qr_a, qr_b, name="inner_product")

        for i in range(n_qubits):
            self.cz(qr_a[i], qr_b[i])
