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

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.library.basis_change import QFT

from .adder import Adder


class DraperQFTAdder(Adder):
    r"""A circuit that uses QFT to perform in-place addition on two qubit registers.

    For registers with :math:`n` qubits, the QFT adder can perform addition modulo
    :math:`2^n` (with `fixed_point=True`) or ordinary addition by adding a carry qubits (with
    `fixed_point=False`).

    As an example, a non-fixed_point QFT adder circuit that performs addition on two 2-qubit sized
    registers is as follows:

    .. parsed-literal::

         a_0:   ─────────■──────■────────────────────────■────────────────
                         │      │                        │
         a_1:   ─────────┼──────┼────────■──────■────────┼────────────────
                ┌──────┐ │P(π)  │        │      │        │       ┌───────┐
         b_0:   ┤0     ├─■──────┼────────┼──────┼────────┼───────┤0      ├
                │      │        │P(π/2)  │P(π)  │        │       │       │
         b_1:   ┤1 qft ├────────■────────■──────┼────────┼───────┤1 iqft ├
                │      │                        │P(π/2)  │P(π/4) │       │
        cout_0: ┤2     ├────────────────────────■────────■───────┤2      ├
                └──────┘                                         └───────┘

    **References:**

    [1] T. G. Draper, Addition on a Quantum Computer, 2000.
    `arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>`_

    [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    [3] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """

    def __init__(
        self, num_state_qubits: int, fixed_point: bool = False, name: str = "DraperQFTAdder"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            fixed_point: Whether addition is fixed_point with mod :math:`2^n`.
                Additional qubit is attached in case of non-fixed_point addition
                to carry the most significant qubit of the sum.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        """
        if num_state_qubits < 1:
            raise ValueError("The number of qubits must be at least 1.")

        super().__init__(num_state_qubits, name=name)

        qr_a = QuantumRegister(num_state_qubits, name="a")
        qr_b = QuantumRegister(num_state_qubits, name="b")
        qr_list = [qr_a, qr_b]

        if not fixed_point:
            qr_z = QuantumRegister(1, name="cout")
            qr_list.append(qr_z)

        # add registers
        self.add_register(*qr_list)

        # define register containing the sum and number of qubits for QFT circuit
        qr_sum = qr_b[:] if fixed_point else qr_b[:] + qr_z[:]
        num_qubits_qft = num_state_qubits if fixed_point else num_state_qubits + 1

        # build QFT adder circuit
        self.append(QFT(num_qubits_qft, do_swaps=False).to_gate(), qr_sum[:])

        for j in range(num_state_qubits):
            for k in range(num_state_qubits - j):
                lam = np.pi / (2 ** k)
                self.cp(lam, qr_a[j], qr_b[j + k])

        if not fixed_point:
            for j in range(num_state_qubits):
                lam = np.pi / (2 ** (j + 1))
                self.cp(lam, qr_a[num_state_qubits - j - 1], qr_z[0])

        self.append(QFT(num_qubits_qft, do_swaps=False).inverse().to_gate(), qr_sum[:])
