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

# pylint: disable=invalid-name

"""
Diagonal single qubit gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
import qiskit.extensions.standard.u3 as u3
import qiskit.extensions.standard.cu1 as cu1
from qiskit.exceptions import QiskitError


class U1Gate(Gate):
    """Diagonal single-qubit gate."""

    def __init__(self, theta, label=None):
        """Create new diagonal single-qubit gate."""
        super().__init__("u1", 1, [theta], label=label)

    def _define(self):
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (u3.U3Gate(0, 0, self.params[0]), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return U1Gate(-self.params[0])

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
        lam = self.params[0]
        lam = float(lam)
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=complex)

    def q_if(self, num_ctrl_qubits=1, label=None):
        """Return controlled version of gate.

        Args:
            num_ctrl_qubits (int): number of control qubits to add. Default 1.
            label (str): optional label for returned gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return cu1.Cu1Gate(*self.params)
        else:
            return super().q_if(num_ctrl_qubits=num_ctrl_qubits, label=label)

            
def u1(self, theta, q):
    """Apply u1 with angle theta to q."""
    return self.append(U1Gate(theta), [q], [])


QuantumCircuit.u1 = u1
