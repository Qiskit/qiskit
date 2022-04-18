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

"""Two-qubit YY-rotation gate."""

from typing import Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType


class RYYGate(Gate):
    r"""A parametric 2-qubit :math:`Y \otimes Y` interaction (rotation about YY).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤1        ├
             │  Ryy(ϴ) │
        q_1: ┤0        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{YY}(\theta) = \exp\left(-i \th Y{\otimes}Y\right) =
            \begin{pmatrix}
                \cos\left(\th\right)   & 0           & 0           & i\sin\left(\th\right) \\
                0           & \cos\left(\th\right)   & -i\sin\left(\th\right) & 0 \\
                0           & -i\sin\left(\th\right) & \cos\left(\th\right)   & 0 \\
                i\sin\left(\th\right)  & 0           & 0           & \cos\left(\th\right)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{YY}(\theta = 0) = I

        .. math::

            R_{YY}(\theta = \pi) = i Y \otimes Y

        .. math::

            R_{YY}\left(\theta = \frac{\pi}{2}\right) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1 & 0 & 0 & i \\
                                        0 & 1 & -i & 0 \\
                                        0 & -i & 1 & 0 \\
                                        i & 0 & 0 & 1
                                    \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new RYY gate."""
        super().__init__("ryy", 2, [theta], label=label)

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .rx import RXGate
        from .rz import RZGate

        #      ┌─────────┐                   ┌──────────┐
        # q_0: ┤ Rx(π/2) ├──■─────────────■──┤ Rx(-π/2) ├
        #      ├─────────┤┌─┴─┐┌───────┐┌─┴─┐├──────────┤
        # q_1: ┤ Rx(π/2) ├┤ X ├┤ Rz(0) ├┤ X ├┤ Rx(-π/2) ├
        #      └─────────┘└───┘└───────┘└───┘└──────────┘
        q = QuantumRegister(2, "q")
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (RXGate(np.pi / 2), [q[0]], []),
            (RXGate(np.pi / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RXGate(-np.pi / 2), [q[0]], []),
            (RXGate(-np.pi / 2), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse RYY gate (i.e. with the negative rotation angle)."""
        return RYYGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RYY gate."""
        theta = float(self.params[0])
        cos = np.cos(theta / 2)
        isin = 1j * np.sin(theta / 2)
        return np.array(
            [[cos, 0, 0, isin], [0, cos, -isin, 0], [0, -isin, cos, 0], [isin, 0, 0, cos]],
            dtype=dtype,
        )
