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

"""
Two-qubit ZZ-rotation gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class RZZGate(Gate):
    r"""The two-qubit ZZ-rotation gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{RZ}}(\theta)
            = \exp\left(-i \frac{\theta}{2}
                         (\sigma_Z\otimes\sigma_Z) \right)
            = \begin{bmatrix}
                e^{-i\theta/2} & 0 & 0 & 0 \\
                0 & 0& e^{\theta/2} & 0 \\
                0 & 0 & e^{i\theta/2} & 0 \\
                0 & 0 & 0 & e^{-i\theta/2}
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new rzz gate."""
        super().__init__('rzz', 2, [theta],
                         phase=phase, label=label)

    def _define(self):
        """
        gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }
        """
        from qiskit.extensions.standard.rz import RZGate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (CXGate(), [q[0], q[1]], []),
            (RZGate(self.params[0], phase=self.phase),
             [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return RZZGate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the RZZ gate."""
        exp_p = numpy.exp(1j * self.params[0] / 2)
        exp_m = numpy.exp(-1j * self.params[0] / 2)
        return numpy.diag([exp_m, exp_p, exp_p, exp_m])


def rzz(self, theta, qubit1, qubit2):
    """Apply RZZ to circuit."""
    return self.append(RZZGate(theta), [qubit1, qubit2], [])


# Add to QuantumCircuit class
QuantumCircuit.rzz = rzz
