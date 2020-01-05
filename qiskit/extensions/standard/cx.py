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
controlled-NOT gate.
"""
import numpy

from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.extensions.standard.x import XGate


class CnotGate(ControlledGate):
    """controlled-NOT gate."""

    def __init__(self):
        """Create new CNOT gate."""
        super().__init__("cx", 2, [], num_ctrl_qubits=1)
        self.base_gate = XGate
        self.base_gate_name = "x"

    def inverse(self):
        """Invert this gate."""
        return CnotGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Cx gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]], dtype=complex)


def cx(self, ctl, tgt):  # pylint: disable=invalid-name
        """Apply CX gate from a specified control (ctl) to target (tgt) qubit.
    The CX gate applies an X gate on the target qubit when the control qubit is in state |1>.

    Example:
    circuit = QuantumCircuit(2)
    circuit.x(0) # This brings the quantum state from |0> to |1>
    circuit.cx(0,1) 
    circuit.draw()
            ┌───┐     
    q_0: |0>┤ X ├──■──
            └───┘┌─┴─┐
    q_1: |0>─────┤ X ├
                 └───┘
    Resulting Statevector:
    [ 0+0j, 0+0j, 0+0j, 1+0j ]
    """
    return self.append(CnotGate(), [ctl, tgt], [])


QuantumCircuit.cx = cx
QuantumCircuit.cnot = cx
