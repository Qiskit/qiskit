# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
iSWAP gate.
"""

import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class iSwapGate(Gate):
    r"""iSWAP gate.

    A 2-qubit XY interaction that is equivalent to a SWAP up to a diagonal.

    .. math::

        iSWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & i & 0 \\
                0 & i & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
          = \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
         .  \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & i & 0 & 0 \\
                0 & 0 & i & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    This gate is symmetric.
    """

    def __init__(self):
        super().__init__('iswap', 2, [])

    def _define(self):
        """
        gate iswap a,b {
            u2(pi/2) a;
            u2(-pi/2) b;
            cx b, a;
            u1(-pi/2) a;
            u2(-pi/2, pi/2) b;
            cx b, a;
            u2(pi, pi/2) a;
            u2(pi, 3*pi/2) b;
            }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.u2 import U2Gate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (U2Gate(np.pi/2, 0), [q[0]], []),
            (U2Gate(-np.pi/2, 0), [q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (U1Gate(-np.pi/2), [q[0]], []),
            (U2Gate(-np.pi/2, np.pi/2), [q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (U2Gate(np.pi, np.pi/2), [q[0]], []),
            (U2Gate(np.pi, -np.pi/2), [q[1]], [])
        ]

    # TODO: after migration to StandardEquivalenceLibrary and adding
    # `phase` to circuits, the matrix can be defined. The decomposition
    # would then be in terms of circuit above, with a phase e^(i*pi/4)
    # def to_matrix(self):
    #    """Return a numpy.array for the iSWAP gate."""
    #    return np.array([[1, 0, 0, 0],
    #                     [0, 0, 1j, 0],
    #                     [0, 1j, 0, 0],
    #                     [0, 0, 0, 1]], dtype=complex)


def iswap(self, qubit1, qubit2):
    """Apply iSWAP gate to a pair specified qubits (qubit1, qubit2).

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.iswap(0,1)
            circuit.draw()

        .. jupyter-execute::

            from qiskit.extensions.standard.iswap import iSwapGate
            from qiskit.quantum_info import Operator
            Operator(iSwapGate()).data
    """
    return self.append(iSwapGate(), [qubit1, qubit2], [])


QuantumCircuit.iswap = iswap
