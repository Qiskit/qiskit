# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two qubit registers using ripple-carry approach."""

from qiskit.synthesis.arithmetic import adder_ripple_rv25
from .adder import Adder


class RVRippleCarryAdder(Adder):
    r"""A ripple-carry circuit to perform in-place addition on two qubit registers of depth
    :math:`O(\log^2 n)` without ancilla qubits as described in [1].

    **References:**

    [1] Remaud and Vandaele, Ancilla-free Quantum Adder with Sublinear Depth, 2025.
    `arXiv:2501.16802 <https://arxiv.org/abs/2501.16802>`__

    """

    def __init__(
        self, num_state_qubits: int, kind: str = "half", name: str = "RVRippleCarryAdder"
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            kind: The kind of adder. We only support ``'half'``. The argument is a placeholder.
            name: The name of the circuit object.
        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
            ValueError: If ``kind`` is not ``'half'``.
        """
        if kind != "half":
            raise ValueError("Only 'half' kind is supported.")

        super().__init__(num_state_qubits, name=name)
        circuit = adder_ripple_rv25(num_state_qubits)

        self.add_register(*circuit.qregs)
        self.append(circuit.to_gate(), self.qubits)
