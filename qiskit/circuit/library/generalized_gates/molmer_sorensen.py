# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Global Mølmer–Sørensen gate.
"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.extensions.standard.rxx import RXXGate


class MSGate(QuantumCircuit):
    """Global Mølmer–Sørensen gate.

    Circuit symbol:

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1    MS    ├
             │           │
        q_2: ┤2          ├
             └───────────┘

    The Mølmer–Sørensen gate is native to ion-trap systems. The global MS can be
    applied to multiple ions to entangle multiple qubits simultaneously.

    In the two-qubit case, this is equivalent to an XX(theta) interaction,
    and is thus reduced to the RXXGate.
    """

    def __init__(self, num_qubits: int, theta: float) -> None:
        """Create a new MS circuit.

        Args:
            num_qubits: list of the 2^k diagonal entries (for a diagonal gate on k qubits).
            theta: rotation angle.
        """
        super().__init__(num_qubits, name="ms")
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                self.append(RXXGate(theta), [i, j])
