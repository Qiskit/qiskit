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

    Circuit to compute the sum of two qubit registers using modular QFT approach from [1]
    or non-modular QFT approach from [2].
    Given two equally sized input registers that store quantum states
    :math:`|a\rangle` and :math:`|b\rangle`, performs addition of numbers that
    can be represented by the states, storing the resulting state in-place in the second register:

    .. math::

        |a\rangle |b\rangle \mapsto |a\rangle |(a+b)\ (\text{mod } 2^n)\rangle

    Here :math:`|a\rangle` (and correspondingly :math:`|b\rangle`) stands for the direct product
    :math:`|a_n\rangle \otimes |a_{n-1}\rangle \ldots |a_{1}\rangle \otimes |a_{0}\rangle`
    which denotes a quantum register prepared with the value :math:`a = 2^{0}a_{0} + 2^{1}a_{1} +
    \ldots 2^{n}a_{n}` [3]. :math:`|(a+b)\ (\text{mod } 2^n)\rangle` is the addition result with
    :math:`(\text{mod } 2^n)` indicating that modulo :math:`2^n` addition can be optionally
    performed, where *n* is the number of qubits in either of the equally sized input registers.
    In case of non-modular addition, an additional qubit is added at the end of the circuit to
    store the addition result of most significant qubits.

    As an example, a non-modular QFT adder circuit that performs addition on two 2-qubit sized
    registers is as follows:

    .. parsed-literal::

        input_a_0:   ─────────■──────■────────────────────────■───────────────
                              │      │                        │
        input_a_1:   ─────────┼──────┼────────■──────■────────┼───────────────
                     ┌──────┐ │P(π)  │        │      │        │       ┌──────┐
        input_b_0:   ┤0     ├─■──────┼────────┼──────┼────────┼───────┤0     ├
                     │      │        │P(π/2)  │P(π)  │        │       │      │
        input_b_1:   ┤1 qft ├────────■────────■──────┼────────┼───────┤1 qft ├
                     │      │                        │P(π/2)  │P(π/4) │      │
        carry_out_0: ┤2     ├────────────────────────■────────■───────┤2     ├
                     └──────┘                                         └──────┘

    **References:**

    [1] T. G. Draper, Addition on a Quantum Computer, 2000.
    `arXiv:quant-ph/0008033 <https://arxiv.org/pdf/quant-ph/0008033.pdf>`_

    [2] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    [3] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_
    """

    def __init__(self,
                 num_state_qubits: int,
                 modular: bool = False,
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
                self.cp(lam, qr_a[num_state_qubits - j - 1], qr_z[0])
        self.append(QFT(num_qubits_qft, do_swaps=False, inverse=True).to_instruction(), qr_sum[:])
