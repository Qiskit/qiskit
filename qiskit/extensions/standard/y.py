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
Pauli Y (bit-phase-flip) gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumRegister
from qiskit.circuit import QuantumCircuit
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class YGate(Gate):
    r"""Pauli Y (bit-phase-flip) gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{Z}} =
            \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}
    """

    def __init__(self, phase=0, label=None):
        """Create new Y gate."""
        super().__init__('y', 1, [], phase=phase, label=label)

    def _define(self):
        from qiskit.extensions.standard.u3 import U3Gate
        q = QuantumRegister(1, 'q')
        self.definition = [
            (U3Gate(pi, pi/2, pi/2, phase=self.phase), [q[0]], [])
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
                return CYGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return YGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the Y gate."""
        return numpy.array([[0, -1j],
                            [1j, 0]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def y(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply Y gate to a specified qubit (qubit).
    A Y gate implements a pi rotation of the qubit state vector about the
    y axis of the Bloch sphere.
    This gate is canonically used to implement a bit flip and phase flip on the qubit state
    from |0⟩ to i|1⟩, or from |1> to -i|0>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.y(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.y import YGate
            YGate().to_matrix()
    """
    return self.append(YGate(), [qubit], [])


QuantumCircuit.y = y


class CYMeta(type):
    """A metaclass to ensure that CyGate and CYGate are of the same type.

    Can be removed when CyGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CYGate, CyGate}  # pylint: disable=unidiomatic-typecheck


class CYGate(ControlledGate, metaclass=CYMeta):
    r"""The controlled-Y gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{CT}} =
            I \otimes |0 \rangle\!\langle 0| +
            U_{\text{Y}} \otimes |1 \rangle\!\langle 1|
            =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & 1 & 0 \\
                0 & i & 0 & 0
            \end{bmatrix}
    """

    def __init__(self, phase=0, label=None):
        """Create new CY gate."""
        super().__init__('cy', 2, [],  phase=0, label=None,
                         num_ctrl_qubits=1)
        self.base_gate = YGate()

    def _define(self):
        """
        gate cy a,b { sdg b; cx a,b; s b; }
        """
        from qiskit.extensions.standard.s import SGate
        from qiskit.extensions.standard.s import SdgGate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (SdgGate(phase=self.phase), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (SGate(), [q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CYGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the CY gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, -1j],
                            [0, 0, 1, 0],
                            [0, 1j, 0, 0]], dtype=complex)


class CyGate(CYGate, metaclass=CYMeta):
    """A deprecated CYGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class CyGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CYGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cy(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
       *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cY gate from a specified control (control_qubit) to target (target_qubit) qubit.
    A cY gate implements a pi rotation of the qubit state vector about the y axis
    of the Bloch sphere when the control qubit is in state |1>.
    This gate is canonically used to implement a bit flip and phase flip on the qubit state
    from |0⟩ to i|1⟩, or from |1> to -i|0> when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.cy(0,1)
            circuit.draw()
    """
    return self.append(CYGate(), [control_qubit, target_qubit], [])


QuantumCircuit.cy = cy
