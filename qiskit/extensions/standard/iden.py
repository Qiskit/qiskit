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
Identity gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
import qiskit.circuit.controlledgate as controlledgate


class IdGate(Gate):
    """Identity gate.

    Identity gate corresponds to a single-qubit gate wait cycle,
    and should not be optimized or unrolled (it is an opaque gate).
    """

    def __init__(self, label=None):
        """Create new Identity gate."""
        super().__init__("id", 1, [], label=label)

    def inverse(self):
        """Invert this gate."""
        return IdGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Id gate."""
        return numpy.array([[1, 0],
                            [0, 1]], dtype=complex)

    def q_if(self, num_ctrl_qubits=1, label=None):
        """Return controlled version of gate.

        Args:
            num_ctrl_qubits (int): number of control qubits to add. Default 1.
            label (str): optional label for returned gate.

        Raise:
            QiskitError: unallowed num_ctrl_qubits specified.
        """
        width = num_ctrl_qubits + 1
        definition = []
        q = QuantumRegister(width, "q")
        rule = [
            (IdGate(label=label), [q[i]], []) for i in range(width)
        ]
        for inst in rule:
            definition.append(inst)
        return controlledgate.ControlledGate('c{0:d}{1}'.format(num_ctrl_qubits, self.name),
                                             num_ctrl_qubits+1, self.params,
                                             num_ctrl_qubits=num_ctrl_qubits, label=label,
                                             definition=definition)
        

def iden(self, q):
    """Apply Identity to q.

    Identity gate corresponds to a single-qubit gate wait cycle,
    and should not be optimized or unrolled (it is an opaque gate).
    """
    return self.append(IdGate(), [q], [])


QuantumCircuit.iden = iden
