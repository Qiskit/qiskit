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
Pauli Z (phase-flip) gate.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class ZGate(Gate):
    r"""Pauli Z (phase-flip) gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{Z}} =
            \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}
    """

    def __init__(self, phase=0, label=None):
        """Create new Z gate."""
        super().__init__('z', 1, [], phase=phase, label=label)

    def _define(self):
        from qiskit.extensions.standard.u1 import U1Gate
        q = QuantumRegister(1, 'q')
        self.definition = [
            (U1Gate(pi, phase=self.phase), [q[0]], [])
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
                return CZGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return ZGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the Z gate."""
        return numpy.array([[1, 0],
                            [0, -1]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def z(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply Z gate to a specified qubit (qubit).
    A Z gate implements a pi rotation of the qubit state vector about the
    z axis of the Bloch sphere.
    This gate is canonically used to implement a phase flip on the qubit state from |+⟩ to |-⟩,
    or vice versa.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.z(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.z import ZGate
            ZGate().to_matrix()
    """
    return self.append(ZGate(), [qubit], [])


QuantumCircuit.z = z


class CZMeta(type):
    """A metaclass to ensure that CzGate and CZGate are of the same type.

    Can be removed when CzGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CZGate, CzGate}  # pylint: disable=unidiomatic-typecheck


class CZGate(ControlledGate, metaclass=CZMeta):
    r"""The controlled-Z gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{CZ}} =
            I \otimes |0 \rangle\!\langle 0| +
            U_{\text{Z}} \otimes |1 \rangle\!\langle 1|
            =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
            \end{bmatrix}
    """

    def __init__(self, phase=0, label=None):
        """Create new CZ gate."""
        super().__init__('cz', 2, [], phase=phase, label=label,
                         num_ctrl_qubits=1)
        self.base_gate = ZGate()

    def _define(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        from qiskit.extensions.standard.h import HGate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (HGate(phase=self.phase), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CZGate(phase=-self.phase)  # self-inverse

    def _matrix_definition(self):
        """Return a numpy.array for the CZ gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, -1]], dtype=complex)


class CzGate(CZGate, metaclass=CZMeta):
    """The deprecated CZGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class CzGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CZGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cz(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
       *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cZ gate from a specified control (control_qubit) to target (target_qubit) qubit.
    A cZ gate implements a pi rotation of the qubit state vector about the z axis
    of the Bloch sphere when the control qubit is in state |1>.
    This gate is canonically used to implement a phase flip on the qubit state from |+⟩ to |-⟩,
    or vice versa when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit
            import numpy

            circuit = QuantumCircuit(2)
            circuit.cz(0,1)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.z import CZGate
            CZGate().to_matrix()
    """
    return self.append(CZGate(), [control_qubit, target_qubit], [])


QuantumCircuit.cz = cz
