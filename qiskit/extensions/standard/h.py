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
Hadamard gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.extensions.standard.t import TGate, TdgGate
from qiskit.extensions.standard.s import SGate, SdgGate
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


# pylint: disable=cyclic-import
class HGate(Gate):
    r"""Hadamard gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{H}} = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                1 & 1 \\
                1 & -1
            \end{bmatrix}
    """

    def __init__(self, phase=0, label=None):
        """Create new Hadamard gate."""
        super().__init__('h', 1, [], phase=phase, label=label)

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        from qiskit.extensions.standard.u2 import U2Gate
        definition = []
        q = QuantumRegister(1, 'q')
        self.definition = [
            (U2Gate(0, pi, phase=self.phase), [q[0]], [])
        ]

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            if num_ctrl_qubits == 1:
                return CHGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return HGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a Numpy.array for the H gate."""
        return numpy.array([[1, 1],
                            [1, -1]], dtype=complex) / numpy.sqrt(2)


@deprecate_arguments({'q': 'qubit'})
def h(self, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply Hadamard (H) gate to a specified qubit (qubit).
    An H gate implements a rotation of pi about the axis (x + z)/sqrt(2) on the Bloch sphere.
    This gate is canonically used to rotate the qubit state from |0⟩ to |+⟩ or |1⟩ to |-⟩.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.h(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.h import HGate
            HGate().to_matrix()
    """
    return self.append(HGate(), [qubit], [])


QuantumCircuit.h = h


class CHGate(ControlledGate):
    r"""The controlled-H gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{CH}} =
            I \otimes |0 \rangle\!\langle 0| +
            U_{\text{H}} \otimes |1 \rangle\!\langle 1|
            = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & \frac{1}{\sqrt{2}} \\
                0 & 0 & 1 & 0 \\
                0 & \frac{1}{\sqrt{2}} & 0 & -\frac{1}{\sqrt{2}}
            \end{bmatrix}
    """


    def __init__(self, phase=0, label=None):
        """Create new CH gate."""
        super().__init__('ch', 2, [], phase=0, label=None,
                         num_ctrl_qubits=1)
        self.base_gate = HGate()

    def _define(self):
        """
        gate ch a,b {
            s b;
            h b;
            t b;
            cx a, b;
            tdg b;
            h b;
            sdg b;
        }
        """
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (SGate(phase=self.phase), [q[1]], []),
            (HGate(), [q[1]], []),
            (TGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (TdgGate(), [q[1]], []),
            (HGate(), [q[1]], []),
            (SdgGate(), [q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CHGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the CH gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 1 / numpy.sqrt(2), 0, 1 / numpy.sqrt(2)],
                            [0, 0, 1, 0],
                            [0, 1 / numpy.sqrt(2), 0, -1 / numpy.sqrt(2)]],
                           dtype=complex)


@deprecate_arguments({'ctl': 'control_qubit', 'tgt': 'target_qubit'})
def ch(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
       *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cH gate from a specified control (control_qubit) to target (target_qubit) qubit.
    This gate is canonically used to rotate the qubit state from |0⟩ to |+⟩ and and |1⟩ to |−⟩
    when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.ch(0,1)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.h import CHGate
            CHGate().to_matrix()
    """
    return self.append(CHGate(), [control_qubit, target_qubit], [])


QuantumCircuit.ch = ch
