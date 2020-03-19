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
Double-CNOT gate.
"""

import numpy as np
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister


class DCXGate(Gate):
    r"""Double-CNOT gate.

    A 2-qubit interaction consisting of two back-to-back
    CNOTs with alternate controls.

    .. parsed-literal::
                  ┌───┐
        q_0: ──■──┤ X ├
             ┌─┴─┐└─┬─┘
        q_1: ┤ X ├──■──
             └───┘

    This is a classical logic gate, equivalent to a CNOT-SWAP (CNS) sequence,
    and locally equivalent to an iSWAP.

    .. math::

        DCX =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0
            \end{pmatrix}
    """

    def __init__(self):
        super().__init__('dcx', 2, [])

    def _define(self):
        """
        gate dcx a, b { cx a, b; cx a, b; }
        """
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (CXGate(), [q[0], q[1]], []),
            (CXGate(), [q[1], q[0]], [])
        ]
        # alternate decomposition in terms of iswap
        # gate dcx a, b { h a; rz(-pi/2) a; rz(-pi/2) b; iswap(a, b), h b;}

    def to_matrix(self):
        """Return a numpy.array for the DCX gate."""
        return np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0]], dtype=complex)


def dcx(self, qubit1, qubit2):
    """Apply DCX gate to a pair specified qubits (qubit1, qubit2).

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.dcx(0, 1)
            print(circuit)
            print(circuit.decompose())

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.dcx import DCXGate
            DCXGate().to_matrix()
    """
    return self.append(DCXGate(), [qubit1, qubit2], [])


QuantumCircuit.dcx = dcx
