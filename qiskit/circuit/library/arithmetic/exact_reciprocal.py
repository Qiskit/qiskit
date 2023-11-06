# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Exact reciprocal rotation."""

from math import isclose
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.generalized_gates import UCRYGate


class ExactReciprocal(QuantumCircuit):
    r"""Exact reciprocal

    .. math::

        |x\rangle |0\rangle \mapsto \cos(1/x)|x\rangle|0\rangle + \sin(1/x)|x\rangle |1\rangle
    """

    def __init__(
        self, num_state_qubits: int, scaling: float, neg_vals: bool = False, name: str = "1/x"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits representing the value to invert.
            scaling: Scaling factor :math:`s` of the reciprocal function, i.e. to compute
                :math:`s / x`.
            neg_vals: Whether :math:`x` might represent negative values. In this case the first
                qubit is the sign, with :math:`|1\rangle` for negative and :math:`|0\rangle` for
                positive.  For the negative case it is assumed that the remaining string represents
                :math:`1 - x`. This is because :math:`e^{-2 \pi i x} = e^{2 \pi i (1 - x)}` for
                :math:`x \in [0,1)`.
            name: The name of the object.

        .. note::

            It is assumed that the binary string :math:`x` represents a number < 1.
        """
        qr_state = QuantumRegister(num_state_qubits, "state")
        qr_flag = QuantumRegister(1, "flag")
        circuit = QuantumCircuit(qr_state, qr_flag, name=name)

        angles = [0.0]
        nl = 2 ** (num_state_qubits - 1) if neg_vals else 2**num_state_qubits

        # Angles to rotate by scaling / x, where x = i / nl
        for i in range(1, nl):
            if isclose(scaling * nl / i, 1, abs_tol=1e-5):
                angles.append(np.pi)
            elif scaling * nl / i < 1:
                angles.append(2 * np.arcsin(scaling * nl / i))
            else:
                angles.append(0.0)

        circuit.compose(
            UCRYGate(angles), [qr_flag[0]] + qr_state[: len(qr_state) - neg_vals], inplace=True
        )

        if neg_vals:
            circuit.compose(
                UCRYGate([-theta for theta in angles]).control(),
                [qr_state[-1]] + [qr_flag[0]] + qr_state[:-1],
                inplace=True,
            )
            angles_neg = [0.0]
            for i in range(1, nl):
                if isclose(scaling * (-1) / (1 - i / nl), -1, abs_tol=1e-5):
                    angles_neg.append(-np.pi)
                elif np.abs(scaling * (-1) / (1 - i / nl)) < 1:
                    angles_neg.append(2 * np.arcsin(scaling * (-1) / (1 - i / nl)))
                else:
                    angles_neg.append(0.0)
            circuit.compose(
                UCRYGate(angles_neg).control(),
                [qr_state[-1]] + [qr_flag[0]] + qr_state[:-1],
                inplace=True,
            )

        super().__init__(*circuit.qregs, name=name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)
