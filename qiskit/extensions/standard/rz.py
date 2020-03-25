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
from qiskit.circuit import Gate
from qiskit.circuit import ControlledGate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.util import deprecate_arguments


class RZGate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Rz(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        RZ(\lambda) =
            \begin{pmatrix}
                e^{-i\frac{\lambda}{2}} & 0 \\
                0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.extensions.standard.U1Gate`
        This gate is equivalent to U1 up to a phase factor.

            .. math::

                U1(\lambda) = e^{i.{\lambda}/2}.RZ(\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """
    def __init__(self, phi):
        super().__init__('rz', 1, [phi])

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        from qiskit.extensions.standard.u1 import U1Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U1Gate(self.params[0]), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-RZ gate.

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
        r"""Return inverted RZ gate (:math:`RZ(\lambda){\dagger} = RZ(-\lambda)`)"""
        return RZGate(-self.params[0])


@deprecate_arguments({'q': 'qubit'})
def rz(self, phi, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply :class:`~qiskit.extensions.standard.RZGate`."""
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
    r"""Controlled-RZ gate.

    This is a diagonal but non-symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    **Circuit symbol:**

    .. parsed-literal::

                ┌───────┐
        q_0: |0>┤ Rz(λ) ├
                └───┬───┘
        q_1: |0>────■────
                         

    **Matrix representation:**

    .. math::

        CRZ(\lambda)\ q_1, q_0 =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{-i\frac{\lambda}{2}} & 0 \\
                0 & 0 & 0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which is how we present the gate above as well, resulting in textbook
        matrices. Instead, if we use q_0 as control, the matrix will be:

        .. math::

            CRZ(\lambda)\ q_0, q_1 =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & e^{-i\frac{\lambda}{2}} & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\frac{\lambda}{2}}
                \end{pmatrix}

    .. seealso::

        :class:`~qiskit.extensions.standard.CU1Gate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """
    def __init__(self, theta):
        super().__init__('crz', 2, [theta], num_ctrl_qubits=1)
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
        definition = []
        q = QuantumRegister(2, 'q')
        rule = [
            (U1Gate(self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

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
    """Apply cRz gate
    """
    return self.append(CRZGate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.crz = crz
