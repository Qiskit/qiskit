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

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from .rzx import RZXGate
from .x import XGate


class ECRGate(Gate):
    r"""An echoed RZX(pi/2) gate implemented using RZX(pi/4) and RZX(-pi/4).

    This gate is maximally entangling and is equivalent to a CNOT up to
    single-qubit pre-rotations. The echoing procedure mitigates some
    unwanted terms (terms other than ZX) to cancel in an experiment.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌─────────┐            ┌────────────┐┌────────┐┌─────────────┐
        q_0: ┤0        ├       q_0: ┤0           ├┤ RX(pi) ├┤0            ├
             │   ECR   │   =        │  RZX(pi/4) │└────────┘│  RZX(-pi/4) │
        q_1: ┤1        ├       q_1: ┤1           ├──────────┤1            ├
             └─────────┘            └────────────┘          └─────────────┘

    **Matrix Representation:**

    .. math::

        ECR\ q_0, q_1 = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                0   & 1   &  0  & i \\
                1   & 0   &  -i & 0 \\
                0   & i   &  0  & 1 \\
                -i  & 0   &  1  & 0
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
                 │   ECR   │
            q_1: ┤0        ├
                 └─────────┘

        .. math::

            ECR\ q_0, q_1 = \frac{1}{\sqrt{2}}
                \begin{pmatrix}
                    0   & 0   &  1  & i \\
                    0   & 0   &  i  & 1 \\
                    1   & -i  &  0  & 0 \\
                    -i  & 1   &  0  & 0
                \end{pmatrix}
    """

    def __init__(self):
        """Create new ECR gate."""
        super().__init__('ecr', 2, [])

    def _define(self):
        """
        gate ecr a, b { rzx(pi/4) a, b; x a; rzx(-pi/4) a, b;}
        """
        q = QuantumRegister(2, 'q')
        self.definition = [
            (RZXGate(np.pi/4), [q[0], q[1]], []),
            (XGate(), [q[0]], []),
            (RZXGate(-np.pi/4), [q[0], q[1]], [])
        ]

    def to_matrix(self):
       """Return a numpy.array for the ECR gate."""
       return 1/np.sqrt(2) * \
               np.array([[0   , 1   , 0    , 1.j],
                         [1   , 0   , -1.j , 0  ],
                         [0   , 1.j , 0    , 1  ],
                         [-1.j, 0   , 1    , 0  ]],
                        dtype=complex)
