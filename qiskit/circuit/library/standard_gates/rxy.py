# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit XY-rotation gate."""

from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType


class RXYGate(Gate):
    r"""A parametric 2-qubit :math:`X \otimes Y` interaction (rotation about XY).

    **Circuit Symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤0        ├
             │  Rxy(θ) │
        q_1: ┤1        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XY}(\theta)\ q_0, q_1 = exp(-i \frac{\theta}{2} Y{\otimes}X) =
            \begin{pmatrix}
                \cos(\th) & 0         & 0          & -\sin(\th)  \\
                0         & \cos(\th) & -\sin(\th) & 0           \\
                0         & \sin(\th) & \cos(\th)  & 0           \\
                \sin(\th) & 0         & 0          & \cos(\th)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in the :math:`Y \otimes X` tensor order.
        Instead, if we apply it on (q_1, q_0), the matrix will
        be :math:`Y \otimes X`:

        .. parsed-literal::

                 ┌─────────┐
            q_0: ┤1        ├
                 │  Rxy(θ) │
            q_1: ┤0        ├
                 └─────────┘

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            R_{XY}(\theta)\ q_1, q_0 = exp(-i \frac{\theta}{2} X{\otimes}Y) =
                \begin{pmatrix}
                    \cos(\th) & 0          & 0         & -\sin(\th)  \\
                    0         & \cos(\th)  & \sin(\th) & 0           \\
                    0         & -\sin(\th) & \cos(\th) & 0           \\
                    \sin(\th) & 0          & 0         & \cos(\th)
                \end{pmatrix}

    **Examples:**

        .. math::

            R_{XY}(\theta = 0) = I

        .. math::

            R_{XY}(\theta = 2\pi) = -I

        .. math::

            R_{XY}(\theta = \pi) = -i Y \otimes X

        .. math::

            R_{XY}(\theta = \frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1 & 0 & 0  & -1 \\
                                        0 & 1 & -1 & 0  \\
                                        0 & 1 & 1  & 0  \\
                                        1 & 0 & 0  & 1
                                    \end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str] = None):
        """Create new RXY gate."""
        super().__init__("rxy", 2, [theta], label=label)

    def _define(self):
        """
        gate rxy(theta) a, b { h a; cx a, b; ry(theta) b; cx a, b; h a;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CXGate
        from .ry import RYGate

        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (RYGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[0]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Return inverse RXY gate (i.e. with the negative rotation angle)."""
        return RXYGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RXY gate."""
        import numpy

        half_theta = float(self.params[0]) / 2
        cos = numpy.cos(half_theta)
        sin = numpy.sin(half_theta)
        return numpy.array(
            [[cos, 0, 0, -sin], [0, cos, -sin, 0], [0, sin, cos, 0], [sin, 0, 0, cos]],
            dtype=dtype,
        )
