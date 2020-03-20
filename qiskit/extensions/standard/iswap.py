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

    A symmetric 2-qubit XY interaction that is
    equivalent to a SWAP up to a diagonal.

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
    """

    def __init__(self):
        super().__init__('iswap', 2, [])

    def _define(self):
        """
        gate iswap a,b {
            s q[0];
            s q[1];
            h q[0];
            cx q[0],q[1];
            cx q[1],q[0];
            h q[1];
        }
        """
        from qiskit.extensions.standard.h import HGate
        from qiskit.extensions.standard.s import SGate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (SGate(), [q[0]], []),
            (SGate(), [q[1]], []),
            (HGate(), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], []),
            (HGate(), [q[1]], [])
        ]

    def to_matrix(self):
        """Return a numpy.array for the iSWAP gate."""
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1j, 0],
                         [0, 1j, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)


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
