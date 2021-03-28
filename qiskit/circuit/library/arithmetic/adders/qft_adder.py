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

"""Compute the sum of two qubit registers using QFT."""

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT


class QFTAdder(QuantumCircuit):
    r"""A circuit that uses QFT to perform in-place addition on two qubit registers.

    **References:**

    [1] T. G. Draper, Addition on a Quantum Computer, 2000.
    `arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>`_

    [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_
    """

    def __init__(self,
                 num_state_qubits: int,
                 modular: bool = True,
                 name: str = 'QFTAdder'
                 ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            modular: Whether addition is modular with mod :math:`2^n`.
                Additional qubit is attached in case of non-modular addition
                to carry the most significant qubit of the sum.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if num_state_qubits < 1:
            raise ValueError('The number of qubits must be at least 1.')

        qr_a = QuantumRegister(num_state_qubits, name='input_a')
        qr_b = QuantumRegister(num_state_qubits, name='input_b')
        qr_list = [qr_a, qr_b]

        if not modular:
            qr_z = QuantumRegister(1, name='carry_out')
            qr_list.append(qr_z)

        # initialize quantum circuit with register list
        super().__init__(*qr_list, name=name)

        # define register containing the sum and number of qubits for QFT circuit
        qr_sum = qr_b[:] if modular else qr_b[:] + qr_z[:]
        num_qubits_qft = num_state_qubits if modular else num_state_qubits + 1

        # build QFT adder circuit
        self.append(QFT(num_qubits_qft, do_swaps=False).to_instruction(), qr_sum[:])
        for j in range(num_state_qubits):
            for k in range(num_state_qubits - j):
                lam = np.pi / (2 ** k)
                self.cp(lam, qr_a[j], qr_b[j + k])
        if not modular:
            for j in range(num_state_qubits):
                lam = np.pi / (2 ** (j + 1))
                self.cp(lam, qr_a[j], qr_z[0])
        self.append(QFT(num_qubits_qft, do_swaps=False, inverse=True).to_instruction(), qr_sum[:])
