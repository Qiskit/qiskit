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

"""Two-qubit ZX-rotation gate."""

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister


class RZXGate(Gate):
    r"""A parameteric 2-qubit :math:`Z \otimes X` interaction (rotation about ZX).

    This gate is maximally entangling at :math:`\theta = \pi/2`.

    The cross-resonance gate (CR) for superconducting qubits implements
    a ZX interaction (however other terms are also present in an experiment).

    **Circuit Symbol:**

    .. parsed-literal::

             ┌─────────┐
        q_0: ┤0        ├
             │  Rzx(θ) │
        q_1: ┤1        ├
             └─────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{ZX}(\theta)\ q_0, q_1 = exp(-i \frac{\theta}{2} X{\otimes}Z) =
            \begin{pmatrix}
                \cos(\th)   & 0          & -i\sin(\th)  & 0          \\
                0           & \cos(\th)  & 0            & i\sin(\th) \\
                -i\sin(\th) & 0          & \cos(\th)    & 0          \\
                0           & i\sin(\th) & 0            & \cos(\th)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in the :math:`X \otimes Z` tensor order.
        Instead, if we apply it on (q_1, q_0), the matrix will
        be :math:`Z \otimes X`:

        .. parsed-literal::

                 ┌─────────┐
            q_0: ┤1        ├
                 │  Rzx(θ) │
            q_1: ┤0        ├
                 └─────────┘

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            R_{ZX}(\theta)\ q_1, q_0 = exp(-i \frac{\theta}{2} Z{\otimes}X) =
                \begin{pmatrix}
                    \cos(\th)   & -i\sin(\th) & 0           & 0          \\
                    -i\sin(\th) & \cos(\th)   & 0           & 0          \\
                    0           & 0           & \cos(\th)   & i\sin(\th) \\
                    0           & 0           & i\sin(\th)  & \cos(\th)
                \end{pmatrix}

        This is a direct sum of RX rotations, so this gate is equivalent to a
        uniformly controlled (multiplexed) RX gate:

        .. math::

            R_{ZX}(\theta)\ q_1, q_0 =
                \begin{pmatrix}
                    RX(\theta) & 0 \\
                    0 & RX(-\theta)
                \end{pmatrix}

    **Examples:**

        .. math::

            R_{ZX}(\theta = 0) = I

        .. math::

            R_{ZX}(\theta = 2\pi) = -I

        .. math::

            R_{ZX}(\theta = \pi) = -i Z \otimes X

        .. math::

            RZX(\theta = \frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                                    \begin{pmatrix}
                                        1  & 0 & -i & 0 \\
                                        0  & 1 & 0  & i \\
                                        -i & 0 & 1  & 0 \\
                                        0  & i & 0  & 1
                                    \end{pmatrix}
    """

    def __init__(self, theta):
        """Create new RZX gate."""
        super().__init__('rzx', 2, [theta])

    def _define(self):
        """
        gate rzx(theta) a, b { h b; cx a, b; u1(theta) b; cx a, b; h b;}
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CXGate
        from .rz import RZGate
        theta = self.params[0]
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (RZGate(theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        """Return inverse RZX gate (i.e. with the negative rotation angle)."""
        return RZXGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the RZX gate."""
        import numpy
        half_theta = self.params[0] / 2
        cos = numpy.cos(half_theta)
        isin = 1j * numpy.sin(half_theta)
        return numpy.array([[cos, 0, -isin, 0],
                            [0, cos, 0, isin],
                            [-isin, 0, cos, 0],
                            [0, isin, 0, cos]],
                           dtype=complex)
