# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit ZZ-rotation gate."""

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RZZGate(Gate):
    r"""A parameteric 2-qubit :math:`Z \otimes Z` interaction (rotation about ZZ).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    **Circuit Symbol:**

    .. parsed-literal::

        q_0: ───■────
                │zz(θ)
        q_1: ───■────

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{ZZ}(\theta) = exp(-i \th Z{\otimes}Z) =
            \begin{pmatrix}
                e^{-i \th} & 0 & 0 & 0 \\
                0 & e^{i \th} & 0 & 0 \\
                0 & 0 & e^{i \th} & 0 \\
                0 & 0 & 0 & e^{-i \th}
            \end{pmatrix}

    This is a direct sum of RZ rotations, so this gate is equivalent to a
    uniformly controlled (multiplexed) RZ gate:

    .. math::

        R_{ZZ}(\theta) =
            \begin{pmatrix}
                RZ(\theta) & 0 \\
                0 & RZ(-\theta)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{ZZ}(\theta = 0) = I

        .. math::

            R_{ZZ}(\theta = 2\pi) = -I

        .. math::

            R_{ZZ}(\theta = \pi) = - Z \otimes Z

        .. math::

            R_{ZZ}(\theta = \frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1-i & 0 & 0 & 0 \\
                                        0 & 1+i & 0 & 0 \\
                                        0 & 0 & 1+i & 0 \\
                                        0 & 0 & 0 & 1-i
                                    \end{pmatrix}
    """

    def __init__(self, theta):
        """Create new RZZ gate."""
        super().__init__('rzz', 2, [theta])

    def _define(self):
        """
        gate rzz(theta) a, b { cx a, b; u1(theta) b; cx a, b; }
        """
        from .u1 import U1Gate
        from .x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0]), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return inverse RZZ gate (i.e. with the negative rotation angle)."""
        return RZZGate(-self.params[0])

    # TODO: this is the correct matrix and is equal to the definition above,
    # however the control mechanism cannot distinguish U1 and RZ yet.
    # def to_matrix(self):
    #    """Return a numpy.array for the RZZ gate."""
    #    theta = float(self.params[0])
    #    return np.array([[np.exp(-1j*theta/2), 0, 0, 0],
    #                     [0, np.exp(1j*theta/2), 0, 0],
    #                     [0, 0, np.exp(1j*theta/2), 0],
    #                     [0, 0, 0, np.exp(-1j*theta/2)]], dtype=complex)
