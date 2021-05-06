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

"""Compute the product of two qubit registers using QFT."""

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate
from qiskit.circuit.library.basis_change import QFT


class QFTMultiplier(QuantumCircuit):
    r"""A QFT multiplication circuit to store product of two input registers out-of-place."""

    def __init__(self,
                 num_state_qubits: int,
                 name: str = 'QFTMultiplier') -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is smaller than 1.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')

        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        qr_out = QuantumRegister(2 * num_state_qubits, name='out')

        # initialize quantum circuit with register list
        super().__init__(qr_a, qr_b, qr_out, name=name)

        # build multiplication circuit
        self.append(QFT(2 * num_state_qubits, do_swaps=False).to_gate(), qr_out[:])

        for j in range(1, num_state_qubits + 1):
            for i in range(1, num_state_qubits + 1):
                for k in range(1, 2 * num_state_qubits + 1):
                    lam = (2 * np.pi) / (2 ** (i + j + k - 2 * num_state_qubits))
                    self.append(
                        RZGate(lam).control(2),
                        [qr_a[num_state_qubits - j], qr_b[num_state_qubits - i], qr_out[k - 1]]
                    )

        self.append(QFT(2 * num_state_qubits, do_swaps=False).inverse().to_gate(), qr_out[:])
