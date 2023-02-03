# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""sqrt(iSWAP) gate."""

import numpy as np
from qiskit.circuit.gate import Gate


class SQiSWGate(Gate):
    r"""sqrt(iSWAP) gate.

    A 2-qubit symmetric gate from the iSWAP (or XY) family.
    It has Weyl chamber coordinates (π/8, π/8, 0).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.sqisw` method.

    .. parsed-literal::

                 ┌────────────┐┌────────────┐
            q_0: ┤0           ├┤0           ├
                 │  Rxx(-π/4) ││  Ryy(-π/4) │
            q_1: ┤1           ├┤1           ├
                 └────────────┘└────────────┘

    .. math::
        B\ q_0, q_1 =
            \begin{pmatrix}
                1       & 0                 & 0                     & 0     \\
                0       & \frac{1}{\sqrt(2)} & \frac{i}{\sqrt(2)}     & 0     \\
                0       & \frac{i}{\sqrt(2)} & \frac{1}{\sqrt(2)}     & 0     \\
                0       & 0                 & 0                     & 1
            \end{pmatrix}
    """

    def __init__(self):
        """Create new SQiSW gate."""
        super().__init__("sqisw", 2, [])

    def _define(self):
        """
        gate SQiSW a, b { rxx(-pi/4) a, b; ryy(-pi/4) a, b; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumRegister, QuantumCircuit
        from .rxx import RXXGate
        from .ryy import RYYGate

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RXXGate(-np.pi / 4), [q[0], q[1]], []), (RYYGate(-np.pi / 4), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the DCX gate."""
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )
