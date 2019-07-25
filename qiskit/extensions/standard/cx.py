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
controlled-NOT gate.
"""

import numpy

from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
import qiskit.extensions.standard.ccx as ccx


class CnotGate(ControlledGate):
    """controlled-NOT gate."""

    def __init__(self):
        """Create new CNOT gate."""
        super().__init__("cx", 2, [], num_ctrl_qubits=1)

    def inverse(self):
        """Invert this gate."""
        return CnotGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Cx gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]], dtype=complex)

    def q_if(self, num_ctrl_qubits=1, label=None):
        """return controlled CNOT"""
        if num_ctrl_qubits == 1:
            return ccx.ToffoliGate()
        elif isinstance(num_ctrl_qubits, int) and num_ctrl_qubits>=1:
            return ControlledGate('c{0:d}{1}'.format(num_ctrl_qubits+1, 'x'),
                                  num_ctrl_qubits+self.num_qubits, self.params,
                                  num_ctrl_qubits=num_ctrl_qubits+1, label=label)
        else:
            raise QiskitError('Number of control qubits must be >=1')
            

def cx(self, ctl, tgt):
    """Apply CX from ctl to tgt."""
    return self.append(CnotGate(), [ctl, tgt], [])


QuantumCircuit.cx = cx
