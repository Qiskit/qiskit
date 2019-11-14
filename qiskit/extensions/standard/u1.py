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
Diagonal single qubit gate.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


# pylint: disable=cyclic-import
class U1Gate(Gate):
    r"""Diagonal single-qubit gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_1(\lambda) = \begin{bmatrix}
            1 & 0 \\
            0 &  e^{i \lambda}
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new diagonal single-qubit gate."""
        super().__init__("u1", 1, [theta],
                         phase=phase, label=label)

    def _define(self):
        from qiskit.extensions.standard.u3 import U3Gate
        q = QuantumRegister(1, "q")
        self.definition = [
            (U3Gate(0, 0, self.params[0], phase=self.phase),
             [q[0]], [])
        ]

    def control(self, num_ctrl_qubits=1, label=None):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1 and not self.phase:
            return Cu1Gate(*self.params, label=label)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label)

    def inverse(self):
        """Invert this gate."""
        return U1Gate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the U3 gate."""
        lam = float(self.params[0])
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def u1(self, theta, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply U1 gate with angle theta to a specified qubit (qubit).
    u1(λ) := diag(1, eiλ) ∼ U(0, 0, λ) = Rz(λ) where ~ is equivalence up to a global phase.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(1)
            circuit.u1(theta,0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            import numpy
            from qiskit.extensions.standard.u1 import U1Gate
            U1Gate(numpy.pi/2).to_matrix()
    """
    return self.append(U1Gate(theta), [qubit], [])


QuantumCircuit.u1 = u1


class Cu1Gate(ControlledGate):
    """controlled-u1 gate."""r"""Controlled-Z gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{Cu1}}(\lambda) =
            I \otimes |0 \rangle\!\langle 0| +
            U_{1}(\lambda) \otimes |1 \rangle\!\langle 1|
            =
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i \lambda}
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new cu1 gate."""
        super().__init__("cu1", 2, [theta], phase=0, label=None,
                         num_ctrl_qubits=1)
        self.base_gate = U1Gate(theta)

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        from qiskit.extensions.standard.x import CnotGate
        q = QuantumRegister(2, "q")
        self.definition = [
            (U1Gate(self.params[0] / 2, phase=self.phase), [q[0]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CnotGate(), [q[0], q[1]], []),
            (U1Gate(self.params[0] / 2), [q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return Cu1Gate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the Cu1 gate."""
        lam = float(self.params[0])
        return numpy.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, numpy.exp(1j * lam)]], dtype=complex)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cu1(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cU1 gate from a specified control (control_qubit) to target (target_qubit) qubit
    with angle theta. A cU1 gate implements a theta radian rotation of the qubit state vector
    about the z axis of the Bloch sphere when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(2)
            circuit.cu1(theta,0,1)
            circuit.draw()
    """
    return self.append(Cu1Gate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.cu1 = cu1
