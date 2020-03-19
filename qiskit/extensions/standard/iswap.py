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
iSWAP gate.
"""

import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class iSwapGate(Gate):
    """iSWAP gate.

    A 2-qubit XY interaction. Symmetric and self-inverse.
    """

    def __init__(self):
        """Create new iSWAP gate."""
        super().__init__('iswap', 2, [])

    def _define(self):
        """
        gate iswap a,b { 
            ry(pi/2) b; rz(-pi/2) b;
            ry(pi/2) a; rz(pi/2) a;
            cx b, a;
            rx(pi/2) b;
            rz(-pi/2) a;
            cx b, a;
            rz(3*pi/2) b; ry(pi/2) b; rz(pi) b;
            rz(5*pi/2) a; ry(pi/2) a; rz(pi) a;
            }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.u2 import U2Gate
        from qiskit.extensions.standard.x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        self.definition = [
            (U2Gate(-np.pi/2, 0), [q[1]], []),
            (U2Gate(np.pi/2, 0), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (U2Gate(-np.pi/2, np.pi/2), [q[1]], []),
            (U1Gate(-np.pi/2), [q[0]], []),
            (CXGate(), [q[1], q[0]], []),
            (U2Gate(np.pi, 5*np.pi/2), [q[0]], []),
            (U2Gate(np.pi, 3*np.pi/2), [q[1]], []),
        ]

    def inverse(self):
        """Invert this gate."""
        return iSwapGate()

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

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.iswap import iSwapGate
            iSwapGate().to_matrix()
    """
    return self.append(iSwapGate(), [qubit1, qubit2], [])


QuantumCircuit.iswap = iswap
