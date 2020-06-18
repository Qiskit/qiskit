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
        circ = self.decompositions[0]
        gp = circ.phase / len(circ.qregs[0])
        if circ.phase:
            circ.u3(np.pi, gp, gp - np.pi, circ.qregs[0])
            circ.x(circ.qregs[0])
        self.definition = circ.to_gate().definition

    def inverse(self):
        """Return inverse RYY gate (i.e. with the negative rotation angle)."""
        return RYYGate(-self.params[0])

    # TODO: this is the correct matrix and is equal to the definition above,
    # however the control mechanism cannot distinguish U1 and RZ yet.
    def to_matrix_hide(self):
        """Return a numpy.array for the RYY gate."""
        theta = self.params[0]
        halfcos = np.cos(theta / 2)
        halfsin = np.sin(theta / 2)
        return np.exp(0.5j * theta) * np.array([
            [halfcos, 0, 0, 1j * halfsin],
            [0, halfcos, -1j * halfsin, 0],
            [0, -1j * halfsin, halfcos, 0],
            [1j * halfsin, 0, 0, halfcos]
        ], dtype=complex)
