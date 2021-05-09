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

from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import RZGate
from qiskit.circuit.library.basis_change import QFT

from .multiplier import Multiplier


class QFTMultiplier(Multiplier):
    r"""A QFT multiplication circuit to store product of two input registers out-of-place.

    Multiplication in this circuit is implemented using the procedure of Fig. 3 in [1], where
    weighted sum rotations are implemented as given in Fig. 5 in [1]. QFT is used on the output
    register and is followed by rotations controlled by input registers. The rotations
    transform the state into the product of two input registers in QFT base, which is
    reverted from QFT base using inverse QFT.

    **References:**

    [1] Ruiz-Perez et al., Quantum arithmetic with the Quantum Fourier Transform, 2017.
    `arXiv:1411.5949 <https://arxiv.org/pdf/1411.5949.pdf>`_

    """

    def __init__(self,
                 num_state_qubits: int,
                 name: str = 'QFTMultiplier') -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            name: The name of the circuit object.

        """
        super().__init__(num_state_qubits, name=name)

        # define the registers
        qr_a = QuantumRegister(num_state_qubits, name='a')
        qr_b = QuantumRegister(num_state_qubits, name='b')
        qr_out = QuantumRegister(2 * num_state_qubits, name='out')
        self.add_register(qr_a, qr_b, qr_out)

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
