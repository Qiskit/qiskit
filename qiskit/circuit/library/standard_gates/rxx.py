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

"""Two-qubit XX-rotation gate."""

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RXXGate(Gate):
    r"""A parameteric 2-qubit :math:`X \otimes X` interaction (rotation about XX).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤1        ├
             │  Rxx(ϴ) │
        q_1: ┤0        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XX}(\theta) = exp(-i \th X{\otimes}X) =
            \begin{pmatrix}
                \cos(\th)   & 0           & 0           & -i\sin(\th) \\
                0           & \cos(\th)   & -i\sin(\th) & 0 \\
                0           & -i\sin(\th) & \cos(\th)   & 0 \\
                -i\sin(\th) & 0           & 0           & \cos(\th)
            \end{pmatrix}

    **Examples:**

        .. math::

            R_{XX}(\theta = 0) = I

        .. math::

            R_{XX}(\theta = \pi) = i X \otimes X

        .. math::

            R_{XX}(\theta = \frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1  & 0  & 0  & -i \\
                                        0  & 1  & -i & 0 \\
                                        0  & -i & 1  & 0 \\
                                        -i & 0  & 0  & 1
                                    \end{pmatrix}
    """

    def __init__(self, theta):
        """Create new RXX gate."""
        super().__init__('rxx', 2, [theta])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        from .h import HGate
        from .rz import RZGate
        theta = self.params[0]
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[0]], []),
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], []),
            (HGate(), [q[0]], []),
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        """Return inverse RXX gate (i.e. with the negative rotation angle)."""
        return RXXGate(-self.params[0])

    def to_matrix(self):
        """Return a Numpy.array for the RXX gate."""
        import numpy
        theta2 = float(self.params[0]) / 2
        cos = numpy.cos(theta2)
        isin = 1j * numpy.sin(theta2)
        return numpy.array([
            [cos, 0, 0, -isin],
            [0, cos, -isin, 0],
            [0, -isin, cos, 0],
            [-isin, 0, 0, cos]], dtype=complex)
