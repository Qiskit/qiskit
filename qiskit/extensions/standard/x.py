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
X, CX and CCX gates.
"""
import numpy
from qiskit.circuit import ControlledGate
from qiskit.circuit import Gate
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.t import TGate
from qiskit.extensions.standard.t import TdgGate
from qiskit.qasm import pi
from qiskit.util import deprecate_arguments


class XGate(Gate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`).

    **Matrix Representation:**

    .. math::

        X = \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ X ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the X axis.

    .. note::
    
        A global phase difference exists between the definitions of
        :math:`RX(\pi)` and :math:`X`.

        .. math::

            RX(\pi) = \begin{pmatrix}
                        0 & -i \\
                        -i & 0
                      \end{pmatrix}
                    = -i.X

    The gate is equivalent to a classical bit flip.

    .. math::

        |0\rangle \rightarrow |1\rangle \\
        |1\rangle \rightarrow |0\rangle
    """

    def __init__(self, label=None):
        super().__init__('x', 1, [], label=label)

    def _define(self):
        """
        gate x a { u3(pi,0,pi) a; }
        """
        from qiskit.extensions.standard.u3 import U3Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U3Gate(pi, 0, pi), [q[0]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a (mutli-)controlled-X gate.

        One control returns a CX gate. Two controls returns a CCX gate.

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
                return CXGate()
            elif num_ctrl_qubits == 2:
                return CCXGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        r"""Return inverted X gate (:math:`X{\dagger} = X`)"""
        return XGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the X gate."""
        return numpy.array([[0, 1],
                            [1, 0]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def x(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.XGate`.
    """
    return self.append(XGate(), [qubit], [])


QuantumCircuit.x = x


class CXMeta(type):
    """A metaclass to ensure that CnotGate and CXGate are of the same type.

    Can be removed when CnotGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CnotGate, CXGate}  # pylint: disable=unidiomatic-typecheck


class CXGate(ControlledGate, metaclass=CXMeta):
    r"""Controlled-X gate.

    **Circuit symbol:**

    .. parsed-literal::

                ┌───┐
        q_0: |0>┤ X ├
                └─┬─┘
        q_1: |0>──■──


    **Matrix representation:**

    .. math::

        CX\ q_1, q_0 =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which is how we present the gate above as well, resulting in textbook
        matrices. Instead, if we use q_0 as control, the matrix will be:

        .. math::

            CX\ q_0, q_1 =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0
                \end{pmatrix}

    In the computational basis, this gate flips the target qubit
    if the control qubit is in the :math:`|1\rangle` state.
    In this sense it is similar to a classical XOR gate.

    .. math::
        `|a, b\rangle \rightarrow |a, a \oplus b\rangle`
    """

    def __init__(self):
        super().__init__('cx', 2, [], num_ctrl_qubits=1)
        self.base_gate = XGate()

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a controlled-X gate with more control lines.

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
                return CCXGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Return inverted CX gate (:math:`CX^{\dagger} = CX`)"""
        return CXGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CX gate."""
        return numpy.array([[1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0]], dtype=complex)


class CnotGate(CXGate, metaclass=CXMeta):
    """The deprecated CXGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class CnotGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CXGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cx(self, control_qubit, target_qubit,  # pylint: disable=invalid-name
       *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.CXGate`.
    """
    return self.append(CXGate(), [control_qubit, target_qubit], [])


# support both cx and cnot in QuantumCircuits
QuantumCircuit.cx = cx
QuantumCircuit.cnot = cx


class CCXMeta(type):
    """A metaclass to ensure that CCXGate and ToffoliGate are of the same type.

    Can be removed when ToffoliGate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CCXGate, ToffoliGate}  # pylint: disable=unidiomatic-typecheck


class CCXGate(ControlledGate, metaclass=CCXMeta):
    r"""CCX gate, also known as Toffoli gate.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
               │
        q_1: ──■──
             ┌─┴─┐
        q_2: ┤ X ├
             └───┘

    **Matrix representation:**

    .. math::

        CX 0, 1, 2 =
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, lower qubit indices are less significant
        (little endian). Textbook representations of CCX matrix implicitly
        assume the more significant qubits to be the controls. So in order to
        get such a matrix in Qiskit, the higher-index qubits must be the controls.

        .. math::

            CCX 2, 1, 0 =
               \begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{pmatrix}
    """

    def __init__(self):
        super().__init__('ccx', 3, [], num_ctrl_qubits=2)
        self.base_gate = XGate()

    def _define(self):
        """
        gate ccx a,b,c
        {
        h c; cx b,c; tdg c; cx a,c;
        t c; cx b,c; tdg c; cx a,c;
        t b; t c; h c; cx a,b;
        t a; tdg b; cx a,b;}
        """
        definition = []
        q = QuantumRegister(3, 'q')
        rule = [
            (HGate(), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CXGate(), [q[0], q[2]], []),
            (TGate(), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (TdgGate(), [q[2]], []),
            (CXGate(), [q[0], q[2]], []),
            (TGate(), [q[1]], []),
            (TGate(), [q[2]], []),
            (HGate(), [q[2]], []),
            (CXGate(), [q[0], q[1]], []),
            (TGate(), [q[0]], []),
            (TdgGate(), [q[1]], []),
            (CXGate(), [q[0], q[1]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Return an inverted CCX gate (also a CCX)."""
        return CCXGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CCX gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0]], dtype=complex)


class ToffoliGate(CCXGate, metaclass=CCXMeta):
    """The deprecated CCXGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class ToffoliGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CCXGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


@deprecate_arguments({'ctl1': 'control_qubit1',
                      'ctl2': 'control_qubit2',
                      'tgt': 'target_qubit'})
def ccx(self, control_qubit1, control_qubit2, target_qubit,
        *, ctl1=None, ctl2=None, tgt=None):  # pylint: disable=unused-argument
    """Apply :class:`~qiskit.extensions.standard.CCXGate` (Toffoli).
    """

    return self.append(CCXGate(),
                       [control_qubit1, control_qubit2, target_qubit], [])


# support both ccx and toffoli as methods of QuantumCircuit
QuantumCircuit.ccx = ccx
QuantumCircuit.toffoli = ccx
