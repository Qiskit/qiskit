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
controlled-Phase gate.
"""
import numpy

from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.z import ZGate


class CzGate(ControlledGate):
    """controlled-Z gate."""

    def __init__(self, label=None):
        """Create new CZ gate."""
        super().__init__("cz", 2, [], label=label, num_ctrl_qubits=1)
        self.base_gate = ZGate
        self.base_gate_name = "z"

    def _define(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (HGate(), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CzGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Cz gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], dtype=complex)


def cz(self, ctl, tgt):  # pylint: disable=invalid-name
    """Apply cZ gate from a specified control (ctl) to target (tgt) qubit.
    The cZ gate applies a Z gate (flips |+> to |-> and vice versa) on the target qubit
    when the control qubit is in state |1>.

    Examples:

        Construct a circuit with cZ gate.

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.x(0) # This brings the quantum state from |0> to |1>
            circuit.ry(numpy.pi/2,1) # This brings the quantum state from |0> to |+>
            circuit.cz(0,1)
            circuit.draw()

        Resulting Statevector:
        [ 0+0j, 0.707+0j, 0+0j, -0.707+0j ]
    """
    return self.append(CzGate(), [ctl, tgt], [])


QuantumCircuit.cz = cz
