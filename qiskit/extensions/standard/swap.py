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
SWAP gate.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


class SwapGate(Gate):
    """SWAP gate."""

    def __init__(self):
        """Create new SWAP gate."""
        super().__init__("swap", 2, [])

    def _define(self):
        """
        gate swap a,b { cx a,b; cx b,a; cx a,b; }
        """
        from qiskit.extensions.standard.x import CnotGate
        definition = []
        q = QuantumRegister(2, "q")
        rule = [
            (CnotGate(), [q[0], q[1]], []),
            (CnotGate(), [q[1], q[0]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return FredkinGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate."""
        return SwapGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Swap gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=complex)


def swap(self, qubit1, qubit2):
    """Apply SWAP gate to a pair specified qubits (qubit1, qubit2).
    The SWAP gate canonically swaps the states of two qubits.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.swap(0,1)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.swap import SwapGate
            SwapGate().to_matrix()
    """
    return self.append(SwapGate(), [qubit1, qubit2], [])


QuantumCircuit.swap = swap


class FredkinGate(ControlledGate):
    """Fredkin gate."""

    def __init__(self):
        """Create new Fredkin gate."""
        super().__init__("cswap", 3, [], num_ctrl_qubits=1)
        self.base_gate = SwapGate()

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          cx c,b;
        }
        """
        from qiskit.extensions.standard.x import CnotGate, ToffoliGate
        definition = []
        q = QuantumRegister(3, "q")
        rule = [
            (CnotGate(), [q[2], q[1]], []),
            (ToffoliGate(), [q[0], q[1], q[2]], []),
            (CnotGate(), [q[2], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return FredkinGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Fredkin (CSWAP) gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt1': 'target_qubit1',
                      'tgt2': 'target_qubit2'})
def cswap(self, control_qubit, target_qubit1, target_qubit2,
          *, ctl=None, tgt1=None, tgt2=None):  # pylint: disable=unused-argument
    """Apply Fredkin (CSWAP) gate from a specified control (control_qubit) to target1
    (target_qubit1) and target2 (target_qubit2) qubits. The CSWAP gate is canonically
    used to swap the qubit states of target1 and target2 when the control qubit is in
    state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(3)
            circuit.cswap(0,1,2)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.swap import FredkinGate
            FredkinGate().to_matrix()
    """
    return self.append(FredkinGate(),
                       [control_qubit, target_qubit1, target_qubit2],
                       [])


QuantumCircuit.cswap = cswap
QuantumCircuit.fredkin = cswap
