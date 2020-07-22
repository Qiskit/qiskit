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

"""Rotation around the Z axis."""

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumregister import QuantumRegister


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

        RZ(\lambda) = exp(-i\frac{\lambda}{2}Z) =
            \begin{pmatrix}
                e^{-i\frac{\lambda}{2}} & 0 \\
                0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.U1Gate`
        This gate is equivalent to U1 up to a phase factor.

            .. math::

                U1(\lambda) = e^{i{\lambda}/2}RZ(\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, phi, label=None):
        """Create new RZ gate."""
        super().__init__('rz', 1, [phi], label=label)

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name, global_phase=-theta / 2)
        rules = [
            (U1Gate(theta), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

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
        if num_ctrl_qubits == 1:
            gate = CRZGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted RZ gate

        :math:`RZ(\lambda){\dagger} = RZ(-\lambda)`
        """
        return RZGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the RZ gate."""
        import numpy as np
        ilam2 = 0.5j * float(self.params[0])
        return np.array([[np.exp(-ilam2), 0],
                         [0, np.exp(ilam2)]], dtype=complex)


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

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Rz(λ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        CRZ(\lambda)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| + RZ(\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{-i\frac{\lambda}{2}} & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\frac{\lambda}{2}}
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Rz(λ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            CRZ(\lambda)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes RZ(\lambda) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & e^{-i\frac{\lambda}{2}} & 0 \\
                    0 & 0 & 0 & e^{i\frac{\lambda}{2}}
                \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta, label=None, ctrl_state=None):
        """Create new CRZ gate."""
        super().__init__('crz', 2, [theta], num_ctrl_qubits=1, label=label,
                         ctrl_state=ctrl_state)
        self.base_gate = RZGate(theta)

    def _define(self):
        """
        gate crz(lambda) a,b
        { u1(lambda/2) b; cx a,b;
          u1(-lambda/2) b; cx a,b;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U1Gate(self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-self.params[0] / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        qc._data = rules
        self.definition = qc

    def inverse(self):
        """Return inverse RZ gate (i.e. with the negative rotation angle)."""
        return CRZGate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the CRZ gate."""
        import numpy
        arg = 1j * self.params[0] / 2
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0],
                                [0, numpy.exp(-arg), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, numpy.exp(arg)]],
                               dtype=complex)
        else:
            return numpy.array([[numpy.exp(-arg), 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, numpy.exp(arg), 0],
                                [0, 0, 0, 1]],
                               dtype=complex)


class CrzGate(CRZGate, metaclass=CRZMeta):
    """The deprecated CRZGate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class CrzGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CRZGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)
