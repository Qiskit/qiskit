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

"""X, CX, CCX and multi-controlled X gates."""

from math import ceil
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.qasm import pi
from .h import HGate
from .t import TGate, TdgGate
from .u1 import U1Gate
from .u2 import U2Gate


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
                    = -i X

    The gate is equivalent to a classical bit flip.

    .. math::

        |0\rangle \rightarrow |1\rangle \\
        |1\rangle \rightarrow |0\rangle
    """

    def __init__(self, label=None):
        """Create new X gate."""
        super().__init__('x', 1, [], label=label)

    def _define(self):
        """
        gate x a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U3Gate(pi, 0, pi), [q[0]], [])
        ]
        qc._data = rules
        self.definition = qc

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
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        r"""Return inverted X gate (itself)."""
        return XGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the X gate."""
        return numpy.array([[0, 1],
                            [1, 0]], dtype=complex)


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

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ X ├
             └───┘

    **Matrix representation:**

    .. math::

        CX\ q_0, q_1 =
            I \otimes |0\rangle\langle0| + X \otimes |1\rangle\langle1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CX\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes X =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{pmatrix}


    In the computational basis, this gate flips the target qubit
    if the control qubit is in the :math:`|1\rangle` state.
    In this sense it is similar to a classical XOR gate.

    .. math::
        `|a, b\rangle \rightarrow |a, a \oplus b\rangle`
    """

    def __init__(self, ctrl_state=None, label=None):
        """Create new CX gate."""
        super().__init__('cx', 2, [], num_ctrl_qubits=1, label=label,
                         ctrl_state=ctrl_state)
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
            ctrl_state = 2**num_ctrl_qubits - 1
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 1, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Return inverted CX gate (itself)."""
        return CXGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CX gate."""
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0],
                                [0, 1, 0, 0]], dtype=complex)
        else:
            return numpy.array([[0, 0, 1, 0],
                                [0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]], dtype=complex)


class CnotGate(CXGate, metaclass=CXMeta):
    """The deprecated CXGate class."""

    def __init__(self, label=None, ctrl_state=None):
        import warnings
        warnings.warn('The class CnotGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CXGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(label=label, ctrl_state=ctrl_state)


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

        CCX q_0, q_1, q_2 =
            |0 \rangle \langle 0| \otimes I \otimes I + |1 \rangle \langle 1| \otimes CX =
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

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2 and q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──
                   │
            q_2: ──■──

        .. math::

            CCX\ q_2, q_1, q_0 =
                I \otimes I \otimes |0 \rangle \langle 0| + CX \otimes |1 \rangle \langle 1| =
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

    def __init__(self, label=None, ctrl_state=None):
        """Create new CCX gate."""
        super().__init__('ccx', 3, [], num_ctrl_qubits=2, label=label,
                         ctrl_state=ctrl_state)
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
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
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
        qc._data = rules
        self.definition = qc

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
            ctrl_state = 2**num_ctrl_qubits - 1
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 2, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Return an inverted CCX gate (also a CCX)."""
        return CCXGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the CCX gate."""
        return _compute_control_matrix(self.base_gate.to_matrix(),
                                       self.num_ctrl_qubits,
                                       ctrl_state=self.ctrl_state)


class ToffoliGate(CCXGate, metaclass=CCXMeta):
    """The deprecated CCXGate class."""

    def __init__(self):
        import warnings
        warnings.warn('The class ToffoliGate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CCXGate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__()


class RCCXGate(Gate):
    """The simplified Toffoli gate, also referred to as Margolus gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    This implementation requires three CX gates which is the minimal amount possible,
    as shown in https://arxiv.org/abs/quant-ph/0312225.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the dashed box
    of Fig. 3.
    """

    def __init__(self, label=None):
        """Create a new simplified CCX gate."""
        super().__init__('rccx', 3, [], label=label)

    def _define(self):
        """
        gate rccx a,b,c
        { u2(0,pi) c;
          u1(pi/4) c;
          cx b, c;
          u1(-pi/4) c;
          cx a, c;
          u1(pi/4) c;
          cx b, c;
          u1(-pi/4) c;
          u2(0,pi) c;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U2Gate(0, pi), [q[2]], []),  # H gate
            (U1Gate(pi / 4), [q[2]], []),  # T gate
            (CXGate(), [q[1], q[2]], []),
            (U1Gate(-pi / 4), [q[2]], []),  # inverse T gate
            (CXGate(), [q[0], q[2]], []),
            (U1Gate(pi / 4), [q[2]], []),
            (CXGate(), [q[1], q[2]], []),
            (U1Gate(-pi / 4), [q[2]], []),  # inverse T gate
            (U2Gate(0, pi), [q[2]], []),  # H gate
        ]
        qc._data = rules
        self.definition = qc

    def to_matrix(self):
        """Return a numpy.array for the simplified CCX gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1j],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, -1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1j, 0, 0, 0, 0]], dtype=complex)


class C3XGate(ControlledGate):
    """The 3-qubit controlled X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(self, angle=numpy.pi/4, label=None, ctrl_state=None):
        """Create a new 3-qubit controlled X gate.

        Args:
            angle (float): The angle used in the controlled-U1 gates. An angle of π/4 yields the
                3-qubit controlled X gate, an angle of π/8 the 3-qubit controlled sqrt(X) gate.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.
        """
        super().__init__('mcx', 4, [], num_ctrl_qubits=3, label=label, ctrl_state=ctrl_state)
        self.base_gate = XGate()
        self._angle = angle

    def _define(self):
        """
        gate c3x a,b,c,d
        {
            h d; cu1(-pi/4) a,d; h d;
            cx a,b;
            h d; cu1(pi/4) b,d; h d;
            cx a,b;
            h d; cu1(-pi/4) b,d; h d;
            cx b,c;
            h d; cu1(pi/4) c,d; h d;
            cx a,c;
            h d; cu1(-pi/4) c,d; h d;
            cx b,c;
            h d; cu1(pi/4) c,d; h d;
            cx a,c;
            h d; cu1(-pi/4) c,d; h d;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate
        q = QuantumRegister(4, name='q')
        rules = [
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[0], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[1], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[1], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], [])
        ]
        qc = QuantumCircuit(q)
        qc._data = rules
        self.definition = qc

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
            ctrl_state = 2**num_ctrl_qubits - 1
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 3, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Invert this gate. The C3X is its own inverse."""
        return C3XGate(angle=self._angle)

    # This matrix is only correct if the angle is pi/4
    # def to_matrix(self):
    #     """Return a numpy.array for the C3X gate."""
    #     return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)


class RC3XGate(Gate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.
    """

    def __init__(self, label=None):
        """Create a new RC3X gate."""
        super().__init__('rcccx', 4, [], label=label)

    def _define(self):
        """
        gate rc3x a,b,c,d
        { u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          cx a,d;
          u1(pi/4) d;
          cx b,d;
          u1(-pi/4) d;
          u2(0,pi) d;
          u1(pi/4) d;
          cx c,d;
          u1(-pi/4) d;
          u2(0,pi) d;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(4, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U2Gate(0, pi), [q[3]], []),  # H gate
            (U1Gate(pi / 4), [q[3]], []),  # T gate
            (CXGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),  # inverse T gate
            (U2Gate(0, pi), [q[3]], []),
            (CXGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (CXGate(), [q[0], q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[1], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
            (U1Gate(pi / 4), [q[3]], []),
            (CXGate(), [q[2], q[3]], []),
            (U1Gate(-pi / 4), [q[3]], []),
            (U2Gate(0, pi), [q[3]], []),
        ]
        qc._data = rules
        self.definition = qc

    def to_matrix(self):
        """Return a numpy.array for the RC3X gate."""
        return numpy.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=complex)


class C4XGate(ControlledGate):
    """The 4-qubit controlled X gate.

    This implementation is based on Page 21, Lemma 7.5, of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(self, label=None, ctrl_state=None):
        """Create a new 4-qubit controlled X gate."""
        super().__init__('mcx', 5, [], num_ctrl_qubits=4, label=label, ctrl_state=ctrl_state)
        self.base_gate = XGate()

    # seems like open controls not hapening?
    def _define(self):
        """
        gate c3sqrtx a,b,c,d
        {
            h d; cu1(-pi/8) a,d; h d;
            cx a,b;
            h d; cu1(pi/8) b,d; h d;
            cx a,b;
            h d; cu1(-pi/8) b,d; h d;
            cx b,c;
            h d; cu1(pi/8) c,d; h d;
            cx a,c;
            h d; cu1(-pi/8) c,d; h d;
            cx b,c;
            h d; cu1(pi/8) c,d; h d;
            cx a,c;
            h d; cu1(-pi/8) c,d; h d;
        }
        gate c4x a,b,c,d,e
        {
            h e; cu1(-pi/2) d,e; h e;
            c3x a,b,c,d;
            h d; cu1(pi/4) d,e; h d;
            c3x a,b,c,d;
            c3sqrtx a,b,c,e;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate
        q = QuantumRegister(5, name='q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (HGate(), [q[4]], []),
            (CU1Gate(-numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (C3XGate(), [q[0], q[1], q[2], q[3]], []),
            (HGate(), [q[4]], []),
            (CU1Gate(numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (C3XGate(), [q[0], q[1], q[2], q[3]], []),
            (C3XGate(numpy.pi / 8), [q[0], q[1], q[2], q[4]], []),
        ]
        qc._data = rules
        self.definition = qc

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
            ctrl_state = 2**num_ctrl_qubits - 1
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 4, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Invert this gate. The C4X is its own inverse."""
        return C4XGate()

    def to_matrix(self):
        """Return a numpy.array for the C4X gate."""
        return _compute_control_matrix(self.base_gate.to_matrix(),
                                       self.num_ctrl_qubits,
                                       ctrl_state=self.ctrl_state)


class MCXGate(ControlledGate):
    """The general, multi-controlled X gate."""

    def __new__(cls, num_ctrl_qubits=None, label=None, ctrl_state=None):
        """Create a new MCX instance.

        Depending on the number of controls, this creates an explicit CX, CCX, C3X or C4X
        instance or a generic MCX gate.
        """
        # these gates will always be implemented for all modes of the MCX if the number of control
        # qubits matches this
        explicit = {
            1: CXGate,
            2: CCXGate
        }
        if num_ctrl_qubits in explicit.keys():
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class.__new__(gate_class, label=label, ctrl_state=ctrl_state)
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(label=label, ctrl_state=ctrl_state)
            return gate
        return super().__new__(cls)

    def __init__(self, num_ctrl_qubits, label=None, ctrl_state=None, _name='mcx'):
        """Create new MCX gate."""
        num_ancilla_qubits = self.__class__.get_num_ancilla_qubits(num_ctrl_qubits)
        super().__init__(_name, num_ctrl_qubits + 1 + num_ancilla_qubits, [],
                         num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        self.base_gate = XGate()

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits, mode='noancilla'):
        """Get the number of required ancilla qubits without instantiating the class.

        This staticmethod might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        if mode == 'noancilla':
            return 0
        if mode in ['recursion', 'advanced']:
            return int(num_ctrl_qubits > 4)
        if mode[:7] == 'v-chain' or mode[:5] == 'basic':
            return max(0, num_ctrl_qubits - 2)
        raise AttributeError('Unsupported mode ({}) specified!'.format(mode))

    def _define(self):
        """The standard definition used the Gray code implementation."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, name='q')
        qc = QuantumCircuit(q)
        qc._append(MCXGrayCode(self.num_ctrl_qubits), q[:], [])
        self.definition = qc

    @property
    def num_ancilla_qubits(self):
        """The number of ancilla qubits."""
        return self.__class__.get_num_ancilla_qubits(self.num_ctrl_qubits)

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return a multi-controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if ctrl_state is None:
            # use __class__ so this works for derived classes
            gate = self.__class__(self.num_ctrl_qubits + num_ctrl_qubits, label=label,
                                  ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)


class MCXGrayCode(MCXGate):
    r"""Implement the multi-controlled X gate using the Gray code.

    This delegates the implementation to the MCU1 gate, since :math:`X = H \cdot U1(\pi) \cdot H`.
    """

    def __init__(self, num_ctrl_qubits, label=None, ctrl_state=None):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name='mcx_gray')

    def _define(self):
        """Define the MCX gate using the Gray code."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import MCU1Gate
        q = QuantumRegister(self.num_qubits, name='q')
        qc = QuantumCircuit(q, name=self.name)
        qc._append(HGate(), [q[-1]], [])
        qc._append(MCU1Gate(numpy.pi, num_ctrl_qubits=self.num_ctrl_qubits), q[:], [])
        qc._append(HGate(), [q[-1]], [])
        self.definition = qc


class MCXRecursive(MCXGate):
    """Implement the multi-controlled X gate using recursion.

    Using a single ancilla qubit, the multi-controlled X gate is recursively split onto
    four sub-registers. This is done until we reach the 3- or 4-controlled X gate since
    for these we have a concrete implementation that do not require ancillas.
    """

    def __init__(self, num_ctrl_qubits, label=None, ctrl_state=None):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name='mcx_recursive')

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits, mode='recursion'):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def _define(self):
        """Define the MCX gate using recursion."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, name='q')
        qc = QuantumCircuit(q, name=self.name)
        if self.num_qubits == 4:
            qc._append(C3XGate(), q[:], [])
            self.definition = qc
        elif self.num_qubits == 5:
            qc._append(C4XGate(), q[:], [])
            self.definition = qc
        else:
            self.definition = qc
            self.definition._data = self._recurse(q[:-1], q_ancilla=q[-1])

    def _recurse(self, q, q_ancilla=None):
        # recursion stop
        if len(q) == 4:
            return [(C3XGate(), q[:], [])]
        if len(q) == 5:
            return [(C4XGate(), q[:], [])]
        if len(q) < 4:
            raise AttributeError('Something went wrong in the recursion, have less than 4 qubits.')

        # recurse
        num_ctrl_qubits = len(q) - 1
        middle = ceil(num_ctrl_qubits / 2)
        first_half = [*q[:middle], q_ancilla]
        second_half = [*q[middle:num_ctrl_qubits], q_ancilla, q[num_ctrl_qubits]]

        rule = []
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])

        return rule


class MCXVChain(MCXGate):
    """Implement the multi-controlled X gate using a V-chain of CX gates."""

    def __new__(cls, num_ctrl_qubits=None, dirty_ancillas=False,  # pylint: disable=unused-argument
                label=None, ctrl_state=None):
        """Create a new MCX instance.

        This must be defined anew to include the additional argument ``dirty_ancillas``.
        """
        return super().__new__(cls, num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def __init__(self, num_ctrl_qubits, dirty_ancillas=False, label=None, ctrl_state=None):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name='mcx_vchain')
        self._dirty_ancillas = dirty_ancillas

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits, mode='v-chain'):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def _define(self):
        """Define the MCX gate using a V-chain of CX gates."""
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, name='q')
        qc = QuantumCircuit(q, name=self.name)
        q_controls = q[:self.num_ctrl_qubits]
        q_target = q[self.num_ctrl_qubits]
        q_ancillas = q[self.num_ctrl_qubits + 1:]

        definition = []

        if self._dirty_ancillas:
            i = self.num_ctrl_qubits - 3
            ancilla_pre_rule = [
                (U2Gate(0, numpy.pi), [q_target], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
            ]
            for inst in ancilla_pre_rule:
                definition.append(inst)

            for j in reversed(range(2, self.num_ctrl_qubits - 1)):
                definition.append(
                    (RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]], []))
                i -= 1

        definition.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[0]], []))
        i = 0
        for j in range(2, self.num_ctrl_qubits - 1):
            definition.append((RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]], []))
            i += 1

        if self._dirty_ancillas:
            ancilla_post_rule = [
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U2Gate(0, numpy.pi), [q_target], []),
            ]
            for inst in ancilla_post_rule:
                definition.append(inst)
        else:
            definition.append((CCXGate(), [q_controls[-1], q_ancillas[i], q_target], []))

        for j in reversed(range(2, self.num_ctrl_qubits - 1)):
            definition.append((RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]], []))
            i -= 1
        definition.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[i]], []))

        if self._dirty_ancillas:
            for i, j in enumerate(list(range(2, self.num_ctrl_qubits - 1))):
                definition.append(
                    (RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]], []))

        qc._data = definition
        self.definition = qc
