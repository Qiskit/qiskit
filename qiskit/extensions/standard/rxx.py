# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Two-qubit XX-rotation gate.
"""
import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class RXXGate(Gate):
    r"""Two-qubit XX-rotation gate.

    This gate corresponds to the rotation U(θ) = exp(-1j * θ * X⊗X / 2).

    ** Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{RZ}}(\theta)
            = \exp\left(-i \frac{\theta}{2}
                        (\sigma_X\otimes\sigma_X) \right)
            = \begin{bmatrix}
                \cos(\theta / 2) & 0 & 0 & -i \sin(\theta / 2) \\
                0 & \cos(\theta / 2) & -i \sin(\theta / 2) & 0 \\
                0 & -i \sin(\theta / 2) & \cos(\theta / 2) & 0 \\
                -i \sin(\theta / 2) & 0 & 0 & \cos(\theta / 2)
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new rxx gate."""
        super().__init__('rxx', 2, [theta],
                         phase=phase, label=label)

    def _define(self):
        """
        gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
        """
        from qiskit.extensions.standard.x import CXGate
        from qiskit.extensions.standard.rz import RZGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (CXGate(), [q[0], q[1]], []),
            (RZGate(self.params[0], phase=self.phase),
             [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return RXXGate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the RXX gate."""
        theta = float(self.params[0])
        return np.array([
            [np.cos(theta / 2), 0, 0, -1j * np.sin(theta / 2)],
            [0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0],
            [0, -1j * np.sin(theta / 2), np.cos(theta / 2), 0],
            [-1j * np.sin(theta / 2), 0, 0, np.cos(theta / 2)]], dtype=complex)


def rxx(self, theta, qubit1, qubit2):
    """Apply RXX to circuit."""
    return self.append(RXXGate(theta), [qubit1, qubit2], [])


QuantumCircuit.rxx = rxx
