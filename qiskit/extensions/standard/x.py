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
Pauli X (bit-flip) gate.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi


class XGate(Gate):
    """Pauli X (bit-flip) gate."""

    def __init__(self, label=None):
        """Create new X gate."""
        super().__init__("x", 1, [], label=label)

    def _define(self):
        """
        gate x a {
        u3(pi,0,pi) a;
        }
        """
        from qiskit.extensions.standard.u3 import U3Gate
        definition = []
        q = QuantumRegister(1, "q")
        rule = [
            (U3Gate(pi, 0, pi), [q[0]], [])
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
            return CnotGate()
        elif num_ctrl_qubits == 2:
            return ToffoliGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate."""
        return XGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the X gate."""
        return numpy.array([[0, 1],
                            [1, 0]], dtype=complex)


def x(self, q):
    """Apply X gate to a specified qubit (q).
    An X gate implements a pi rotation of the qubit state vector about the
    x axis of the Bloch sphere.
    This gate is canonically used to implement a bit flip on the qubit state from |0⟩ to |1⟩,
    or vice versa.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.x(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.x import XGate
            XGate().to_matrix()
    """
    return self.append(XGate(), [q], [])


QuantumCircuit.x = x


class CnotGate(ControlledGate):
    """controlled-NOT gate."""

    def __init__(self):
        """Create new CNOT gate."""
        super().__init__("cx", 2, [], num_ctrl_qubits=1)
        self.base_gate = XGate
        self.base_gate_name = "x"

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            return ToffoliGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

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
    A CX gate implements a pi rotation of the qubit state vector about the x axis
    of the Bloch sphere when the control qubit is in state |1>.
    This gate is canonically used to implement a bit flip on the qubit state from |0⟩ to |1⟩,
    or vice versa when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.cx(0,1)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.cx import CnotGate
            CnotGate().to_matrix()
    """
    return self.append(CnotGate(), [ctl, tgt], [])


QuantumCircuit.cx = cx
QuantumCircuit.cnot = cx


class ToffoliGate(ControlledGate):
    """Toffoli gate."""

    def __init__(self):
        """Create new Toffoli gate."""
        super().__init__("ccx", 3, [], num_ctrl_qubits=2)
        self.base_gate = XGate
        self.base_gate_name = "x"

    def _define(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        from qiskit.extensions.standard.h import HGate
        from qiskit.extensions.standard.t import TGate
        from qiskit.extensions.standard.t import TdgGate
        definition = []
        q = QuantumRegister(3, "q")
        rule = [
            (HGate(), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CnotGate(), [q[0], q[2]], []),
            (TGate(), [q[2]], []),
            (CnotGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CnotGate(), [q[0], q[2]], []),
            (TGate(), [q[1]], []),
            (TGate(), [q[2]], []),
            (HGate(), [q[2]], []),
            (CnotGate(), [q[0], q[1]], []),
            (TGate(), [q[0]], []),
            (TdgGate(), [q[1]], []),
            (CnotGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return ToffoliGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Toffoli gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0]], dtype=complex)


def ccx(self, ctl1, ctl2, tgt):
    """Apply Toffoli (ccX) gate from two specified controls (ctl1 and ctl2) to target (tgt) qubit.
    This gate is canonically used to rotate the qubit state from |0⟩ to |1⟩, or vice versa when
    both the control qubits are in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(3)
            circuit.ccx(0,1,2)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.ccx import ToffoliGate
            ToffoliGate().to_matrix()
    """
    return self.append(ToffoliGate(), [ctl1, ctl2, tgt], [])


QuantumCircuit.ccx = ccx
QuantumCircuit.toffoli = ccx
