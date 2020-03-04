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
Rotation around the y-axis.
"""
import math
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class RYGate(Gate):
    r"""The rotation around the y-axis.

    **Matrix Definition**

    The matrix for this gate is given by:

    .. math::

        U_{\text{RY}}(\theta)
            = \exp\left(-i \frac{\theta}{2} \sigma_Y \right)
            = \begin{bmatrix}
                \cos(\theta / 2) & -\sin(\theta / 2) \\
                \sin(\theta / 2) &  \cos(\theta / 2)
            \end{bmatrix}
    """

    def __init__(self, theta, phase=0, label=None):
        super().__init__('ry', 1, [theta],
                         phase=phase, label=label)

    def _define(self):
        """
        gate ry(theta) a { r(theta, pi/2) a; }
        """
        from qiskit.extensions.standard.r import RGate
        q = QuantumRegister(1, 'q')
        self.definition = [
            (RGate(self.params[0], pi/2, phase=self.phase),
             [q[0]], [])
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
                return CRYGate(self.params[0])
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate.

        ry(theta)^dagger = ry(-theta)
        """
        return RYGate(-self.params[0], phase=-self.phase)

    def _matrix_definition(self):
        """Return a numpy.array for the RY gate."""
        cos = math.cos(self.params[0] / 2)
        sin = math.sin(self.params[0] / 2)
        return numpy.array([[cos, -sin],
                            [sin, cos]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def ry(self, theta, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply Ry gate with angle theta to a specified qubit (qubit).
    An Ry gate implements a theta radian rotation of the qubit state vector about the
    y axis of the Bloch sphere.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(1)
            circuit.ry(theta,0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            import numpy
            from qiskit.extensions.standard.ry import RYGate
            RYGate(numpy.pi/2).to_matrix()
    """
    return self.append(RYGate(theta), [qubit], [])


QuantumCircuit.ry = ry


class CRYMeta(type):
    """A metaclass to ensure that CryGate and CRYGate are of the same type.

    Can be removed when CryGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CRYGate, CryGate}  # pylint: disable=unidiomatic-typecheck


class CRYGate(ControlledGate, metaclass=CRYMeta):
    """The controlled-ry gate."""

    def __init__(self, theta):
        """Create new cry gate."""
        super().__init__('cry', 2, [theta], num_ctrl_qubits=1)
        self.base_gate = RYGate(theta)

    def _define(self):
        """
        gate cry(lambda) a,b
        { u3(lambda/2,0,0) b; cx a,b;
          u3(-lambda/2,0,0) b; cx a,b;
        }

        """
        from qiskit.extensions.standard.u3 import U3Gate
        from qiskit.extensions.standard.x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (U3Gate(self.params[0] / 2, 0, 0), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return CRYGate(-self.params[0])


class CryGate(CRYGate, metaclass=CRYMeta):
    """The deprecated CRYGate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class CryGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CRYGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cry(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply cry from ctl to tgt with angle theta."""
    return self.append(CRYGate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.cry = cry
