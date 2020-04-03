# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Fourier Transform Circuit."""

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister


class QFT(QuantumCircuit):
    """Quantum Fourier Transform Circuit."""

    def __init__(self, *regs,
                 approximation_degree: int = 0,
                 do_swaps: bool = True,
                 name: str = 'qft') -> None:
        """Construct a new QFT circuit.

        Args:
            *regs: The number of qubits or qubit registers on which the QFT acts.
            approximation_degree: The degree of approximation (0 for no approximation).
            do_swaps: Whether to include the final swaps in the QFT.
            name: The name of the circuit.
        """
        super().__init__(*regs, name=name)

        self._approximation_degree = approximation_degree
        self._do_swaps = do_swaps
        self._data = None

    @QuantumCircuit.num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits.

        Args:
            num_qubits: The new number of qubits.
        """
        # pad with new qubits if the circuit is too small
        if num_qubits > super().num_qubits:
            self.add_register(QuantumRegister(num_qubits - super().num_qubits))
            self._data = None
        # reset if the circuit is being shrunk
        elif num_qubits < super().num_qubits:
            self.qregs = [QuantumRegister(num_qubits)]
            self._data = None

    @property
    def approximation_degree(self) -> int:
        """The approximation degree of the QFT.

        Returns:
            The currently set approximation degree.
        """
        return self._approximation_degree

    @approximation_degree.setter
    def approximation_degree(self, approximation_degree: int) -> None:
        """Set the approximation degree of the QFT.

        Args:
            approximation_degree: The new approximation degree.

        Raises:
            ValueError: If the approximation degree is smaller than 0.
        """
        if approximation_degree < 0:
            raise ValueError('Approximation degree cannot be smaller than 0.')

        if approximation_degree != self._approximation_degree:
            self._data = []
            self._approximation_degree = approximation_degree

    @property
    def do_swaps(self) -> bool:
        """Whether the final swaps of the QFT are applied or not.

        Returns:
            True, if the final swaps are applied, False if not.
        """
        return self._do_swaps

    @do_swaps.setter
    def do_swaps(self, do_swaps: bool) -> None:
        """Specifiy whether to do the final swaps of the QFT circuit or not.

        Args:
            do_swaps: If True, the final swaps are applied, if False not.
        """
        if do_swaps != self._do_swaps:
            self._data = None
            self._do_swaps = do_swaps

    def _swap_qubits(self):
        num_qubits = self.num_qubits
        for i in range(num_qubits // 2):
            self.swap(i, num_qubits - i - 1)

    def inverse(self) -> QuantumCircuit:
        """Return the inverse QFT."""
        iqft = self.copy(name=self.name + '_dg')
        iqft._data = None
        iqft._build(inverse=True)
        return iqft

    def _build(self, inverse: bool = False) -> None:
        """Construct the circuit representing the desired state vector.

        Args:
            inverse: Boolean flag to indicate Inverse Quantum Fourier Transform.
        """
        if self._data:
            return

        self._data = []

        if self._do_swaps and not inverse:
            self._swap_qubits()

        qubit_range = reversed(range(self.num_qubits)) if inverse else range(self.num_qubits)
        for j in qubit_range:
            neighbor_range = range(max(0, j - self.num_qubits + self._approximation_degree + 1), j)
            if inverse:
                neighbor_range = reversed(neighbor_range)
                self.h(j)
            for k in neighbor_range:
                lam = 1.0 * np.pi / float(2 ** (j - k))
                if inverse:
                    lam *= -1
                self.cu1(lam, j, k)
            if not inverse:
                self.h(j)

        if self._do_swaps and inverse:
            self._swap_qubits()

    @property
    def data(self):
        if self._data is None:
            self._build()
        return super().data
