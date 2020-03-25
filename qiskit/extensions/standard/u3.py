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
U3 Gate, three-parameter single-qubit gate.
"""

import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


# pylint: disable=cyclic-import
class U3Gate(Gate):
    r"""Generic single-qubit rotation gate with 3 Euler angles.

    Implemented using two X90 pulses on IBM Quantum systems:

    .. math::
        U2(\phi, \lambda) = RZ(\phi+\pi/2).RX(\frac{\pi}{2}).RZ(\lambda-\pi/2)

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        U3(\theta, \phi, \lambda) =
            \begin{pmatrix}
                \cos(\th)          & e^{-i\lambda}\sin(\th) \\
                e^{i\phi}\sin(\th) & e^{i(\phi+\lambda)\cos(\th)}
            \end{pmatrix}

    **Examples:**

    .. math::

        U3(\theta, -\frac{\pi}{2}, \frac{pi}{2}) = RX(\theta)

    .. math::

        U3(\theta, 0, 0) = RY(\theta)
    """

    def __init__(self, theta, phi, lam, label=None):
        """Create new U3 gate."""
        super().__init__('u3', 1, [theta, phi, lam], label=label)

    def inverse(self):
        r"""Return inverted U3 gate.

        :math:`U3(\theta,\phi,\lambda)^{\dagger} =U3(-\theta,-\phi,-\lambda)`)
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-U3 gate.

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
                return CU3Gate(*self.params)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def to_matrix(self):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        return numpy.array([
            [
                numpy.cos(theta / 2),
                -numpy.exp(1j * lam) * numpy.sin(theta / 2)
            ],
            [
                numpy.exp(1j * phi) * numpy.sin(theta / 2),
                numpy.exp(1j * (phi + lam)) * numpy.cos(theta / 2)
            ]
        ], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def u3(self, theta, phi, lam, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply :class:`~qiskit.extensions.standard.U3Gate`."""
    return self.append(U3Gate(theta, phi, lam), [qubit], [])


QuantumCircuit.u3 = u3


class CU3Meta(type):
    """A metaclass to ensure that Cu3Gate and CU3Gate are of the same type.

    Can be removed when Cu3Gate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CU3Gate, Cu3Gate}  # pylint: disable=unidiomatic-typecheck


class CU3Gate(ControlledGate, metaclass=CU3Meta):
    r"""Controlled-U3 gate (3-parameter two-qubit gate).

    This is a controlled version of the U3 gate (generic single qubit rotation).
    It is restricted to 3 parameters, and so cannot cover generic two-qubit
    controlled gates).

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────┐
        q_0: ┤ U3(ϴ,φ,λ) ├
             └─────┬─────┘
        q_1: ──────■──────


    **Matrix representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}
        
        CU3(\theta, \phi, \lambda)\ q_1, q_0=
            |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U3(\theta,\phi,\lambda) =
            \begin{pmatrix}
                1 & 0   & 0                  & 0 \\
                0 & 1   & 0                  & 0 \\
                0 & 0   & \cos(\th)          & e^{-i\lambda}\sin(\th) \\
                0 & 0   & e^{i\phi}\sin(\th) & e^{i(\phi+\lambda)\cos(\th)}
            \end{pmatrix}


    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which is how we present the gate above as well, resulting in textbook
        matrices. Instead, if we use q_0 as control, the matrix will be:

        .. math::

            CU3(\theta, \phi, \lambda)\ q_0, q_1 =
                I \otimes |0\rangle\langle 0| +
                U3(\theta,\phi,\lambda) \otimes |1\rangle\langle 1| =
                \begin{pmatrix}
                    1 & 0                   & 0 & 0 \\
                    0 & \cos(\th)           & 0 & e^{-i\lambda}\sin(\th) \\
                    0 & 0                   & 1 & 0 \\
                    0 & e^{i\phi}\sin(\th)  & 0 & e^{i(\phi+\lambda)\cos(\th)}
                \end{pmatrix}
    """

    def __init__(self, theta, phi, lam):
        """Create new CU3 gate."""
        super().__init__('cu3', 2, [theta, phi, lam], num_ctrl_qubits=1)
        self.base_gate = U3Gate(theta, phi, lam)

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1((lambda+phi)/2) c;
          u1((lambda-phi)/2) t;
          cx c,t;
          u3(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u3(theta/2,phi,0) t;
        }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.x import CXGate
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (U1Gate((self.params[2] + self.params[1]) / 2), [q[0]], []),
            (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        r"""Return inverted CU3 gate.

        :math:`CU3(\theta,\phi,\lambda)^{\dagger} =CU3(-\theta,-\phi,-\lambda)`)
        """
        return CU3Gate(-self.params[0], -self.params[2], -self.params[1])


class Cu3Gate(CU3Gate, metaclass=CU3Meta):
    """The deprecated CU3Gate class."""

    def __init__(self, theta, phi, lam):
        import warnings
        warnings.warn('The class Cu3Gate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CU3Gate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta, phi, lam)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cu3(self, theta, phi, lam, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.U3Gate`."""
    return self.append(CU3Gate(theta, phi, lam), [control_qubit, target_qubit], [])


QuantumCircuit.cu3 = cu3
