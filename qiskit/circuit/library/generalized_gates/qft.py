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
        n = self.num_qubits
        nums = np.arange(2**n)
        outer = np.outer(nums, nums)
        return np.exp(2j * np.pi * outer * (0.5**n), dtype=dtype) * (0.5 ** (n / 2))

    def _define(self):
        """Provide a specific decomposition of the QFTgate into a quantum circuit."""
        from qiskit.synthesis.qft import synth_qft_full

        self.definition = synth_qft_full(num_qubits=self.num_qubits)
