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
Rotation around the z-axis.
"""
import numpy
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


class RZGate(Gate):
    r"""The rotation around the z-axis.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{RZ}}(\theta)
            = \exp\left(-i \frac{\theta}{2} \sigma_Z \right)
            = \begin{bmatrix}
                e^{-i \theta/2} & 0 \\
                0 & e^{i \theta/2}
            \end{bmatrix}
    """

    def __init__(self, phi, phase=0, label=None):
        """Create new RZ single qubit gate."""
        super().__init__('rz', 1, [phi],
                         phase=phase, label=label)

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        self.definition = [
            (U1Gate(self.params[0], phase=self.phase), [q[0]], [])
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
                return CRZGate(self.params[0])
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate.

        rz(phi)^dagger = rz(-phi)
        """
        return RZGate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a Numpy.array for the RZ gate."""
        return numpy.array([[numpy.exp(-1j * self.params[0] / 2), 0],
                            [0, numpy.exp(1j * self.params[0] / 2)]],
                           dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def rz(self, phi, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply Rz gate with angle phi to a specified qubit (qubit).
    An Rz gate implemements a phi radian rotation of the qubit state vector about the
    z axis of the Bloch sphere.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            phi = Parameter('φ')
            circuit = QuantumCircuit(1)
            circuit.rz(phi,0)
            circuit.draw()
    """
    return self.append(RZGate(phi), [qubit], [])


QuantumCircuit.rz = rz


class CRZMeta(type):
    """A metaclass to ensure that CrzGate and CRZGate are of the same type.

    Can be removed when CrzGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CRZGate, CrzGate}  # pylint: disable=unidiomatic-typecheck


class CRZGate(ControlledGate, metaclass=CRZMeta):
    r"""The controlled-rz gate.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{CRZ}}(\theta) =
            I \otimes |0 \rangle\!\langle 0| +
            U_{\text{RZ}}(\theta) \otimes |1 \rangle\!\langle 1|
            = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{-i \theta/2} & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i \theta/2}
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        """Create new crz gate."""
        super().__init__('crz', 2, [theta], phase=0, label=None,
                         num_ctrl_qubits=1)
        self.base_gate = RZGate(theta)

    def _define(self):
        """
        gate crz(lambda) a,b
        { u1(lambda/2) b; cx a,b;
          u1(-lambda/2) b; cx a,b;
        }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.x import CXGate
        q = QuantumRegister(2, 'q')
        self.definition = [
            (U1Gate(self.params[0] / 2, phase=self.phase), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]

    def inverse(self):
        """Invert this gate."""
        return CRZGate(-self.params[0])


class CrzGate(CRZGate, metaclass=CRZMeta):
    """The deprecated CRZGate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class CrzGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CRZGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)


@deprecate_arguments({'ctl': 'control_qubit', 'tgt': 'target_qubit'})
def crz(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cRz gate from a specified control (control_qubit) to target (target_qubit) qubit
    with angle theta. A cRz gate implements a theta radian rotation of the qubit state vector
    about the z axis of the Bloch sphere when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(2)
            circuit.crz(theta,0,1)
            circuit.draw()
    """
    return self.append(CRZGate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.crz = crz
