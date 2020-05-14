# -*- coding: utf-8 -*-

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

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RYYGate(Gate):
    r"""A parameteric 2-qubit :math:`Y \otimes Y` interaction (rotation about YY).

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

        R_{YY}(\theta) = exp(-i \th Y{\otimes}Y) =
            \begin{pmatrix}
                \cos(\th)   & 0           & 0           & i\sin(\th) \\
                0           & \cos(\th)   & -i\sin(\th) & 0 \\
                0           & -i\sin(\th) & \cos(\th)   & 0 \\
                i\sin(\th)  & 0           & 0           & \cos(\th)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{YY}(\theta = 0) = I

        .. math::

            R_{YY}(\theta = \pi) = i Y \otimes Y

        .. math::

            R_{YY}(\theta = \frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1 & 0 & 0 & i \\
                                        0 & 1 & -i & 0 \\
                                        0 & -i & 1 & 0 \\
                                        i & 0 & 0 & 1
                                    \end{pmatrix}
    """

    def __init__(self, theta):
        """Create new RYY gate."""
        super().__init__('ryy', 2, [theta])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        from .x import CXGate
        from .rx import RXGate
        from .rz import RZGate

        definition = []
        q = QuantumRegister(2, 'q')
        theta = self.params[0]
        rule = [
            (RXGate(np.pi / 2), [q[0]], []),
            (RXGate(np.pi / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RXGate(-np.pi / 2), [q[0]], []),
            (RXGate(-np.pi / 2), [q[1]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverse RYY gate (i.e. with the negative rotation angle)."""
        return RYYGate(-self.params[0])

    # TODO: this is the correct matrix and is equal to the definition above,
    # however the control mechanism cannot distinguish U1 and RZ yet.
    # def to_matrix(self):
    #     """Return a numpy.array for the RYY gate."""
    #     theta = self.params[0]
    #     return np.exp(0.5j * theta) * np.array([
    #         [np.cos(theta / 2), 0, 0, 1j * np.sin(theta / 2)],
    #         [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
    #         [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
    #         [1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)]
    #     ], dtype=complex)
