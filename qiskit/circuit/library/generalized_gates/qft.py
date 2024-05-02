# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QFTGate: gate class for natively reasoning about Quantum Fourier Transforms."""

from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import Gate


class QFTGate(Gate):
    r"""Quantum Fourier Transform Circuit.

    The Quantum Fourier Transform (QFT) on :math:`n` qubits is the operation

    .. math::

        |j\rangle \mapsto \frac{1}{2^{n/2}} \sum_{k=0}^{2^n - 1} e^{2\pi ijk / 2^n} |k\rangle

    """

    def __init__(
        self,
        num_qubits: int,
    ):
        """Construct a new QFT gate.

        Args:
            num_qubits: The number of qubits on which the QFT acts.
        """
        super().__init__(name="qft", num_qubits=num_qubits, params=[])

    def __array__(self, dtype=complex):
        """Return a numpy array for the QFTGate."""
        num_qubits = self.num_qubits
        mat = np.empty((2**num_qubits, 2**num_qubits), dtype=dtype)
        for i in range(2**num_qubits):
            i_index = int(bin(i)[2:].zfill(num_qubits), 2)
            for j in range(i, 2**num_qubits):
                entry = np.exp(2 * np.pi * 1j * i * j / 2**num_qubits) / 2 ** (num_qubits / 2)
                j_index = int(bin(j)[2:].zfill(num_qubits), 2)
                mat[i_index, j_index] = entry
                if i != j:
                    mat[j_index, i_index] = entry
        return mat

    def _basic_decomposition(self):
        """Provide a specific decomposition of the QFT gate into a quantum circuit.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        from qiskit.synthesis.qft import synth_qft_full

        decomposition = synth_qft_full(num_qubits=self.num_qubits)
        return decomposition

    def _define(self):
        """Populate self.definition with a specific decomposition of the gate.
        This is used for constructing Operator from QFTGate, creating qasm
        representations and more.
        """
        self.definition = self._basic_decomposition()
