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

from typing import Optional, Union
import warnings
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _compute_control_matrix, _ctrl_state_to_int
from qiskit.qasm import pi
from .h import HGate
from .t import TGate, TdgGate
from .u1 import U1Gate
from .u2 import U2Gate
from .sx import SXGate


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

    def __init__(self, label: Optional[str] = None):
        """Create new X gate."""
        super().__init__("x", 1, [], label=label)

    def _define(self):
        """
        gate x a { u3(pi,0,pi) a; }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u3 import U3Gate

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U3Gate(pi, 0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a (multi-)controlled-X gate.

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

    def __array__(self, dtype=None):
        """Return a numpy.array for the X gate."""
        return numpy.array([[0, 1], [1, 0]], dtype=dtype)


class CXGate(ControlledGate):
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

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
        """Create new CX gate."""
        super().__init__(
            "cx", 2, [], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=XGate()
        )

    def _define_qasm3(self):
        from qiskit.qasm3.ast import (
            Constant,
            Identifier,
            Integer,
            QuantumBlock,
            QuantumGateModifier,
            QuantumGateModifierName,
            QuantumGateSignature,
            QuantumGateDefinition,
            QuantumGateCall,
        )

        control, target = Identifier("c"), Identifier("t")
        call = QuantumGateCall(
            Identifier("U"),
            [control, target],
            parameters=[Constant.PI, Integer(0), Constant.PI],
            modifiers=[QuantumGateModifier(QuantumGateModifierName.CTRL)],
        )
        return QuantumGateDefinition(
            QuantumGateSignature(Identifier("cx"), [control, target]),
            QuantumBlock([call]),
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 1, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Return inverted CX gate (itself)."""
        return CXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the CX gate."""
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=dtype
            )
        else:
            return numpy.array(
                [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=dtype
            )


class CCXGate(ControlledGate):
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
            I \otimes I \otimes |0 \rangle \langle 0| + CX \otimes |1 \rangle \langle 1| =
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
                |0 \rangle \langle 0| \otimes I \otimes I + |1 \rangle \langle 1| \otimes CX =
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

    def __init__(self, label: Optional[str] = None, ctrl_state: Optional[Union[str, int]] = None):
        """Create new CCX gate."""
        super().__init__(
            "ccx", 3, [], num_ctrl_qubits=2, label=label, ctrl_state=ctrl_state, base_gate=XGate()
        )

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

        q = QuantumRegister(3, "q")
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
            (CXGate(), [q[0], q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Controlled version of this gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(num_ctrl_qubits=num_ctrl_qubits + 2, label=label, ctrl_state=new_ctrl_state)
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        """Return an inverted CCX gate (also a CCX)."""
        return CCXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __array__(self, dtype=None):
        """Return a numpy.array for the CCX gate."""
        mat = _compute_control_matrix(
            self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )
        if dtype:
            return numpy.asarray(mat, dtype=dtype)
        return mat


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

    def __init__(self, label: Optional[str] = None):
        """Create a new simplified CCX gate."""
        super().__init__("rccx", 3, [], label=label)

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

        q = QuantumRegister(3, "q")
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
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the simplified CCX gate."""
        return numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, -1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1j, 0, 0, 0, 0],
            ],
            dtype=dtype,
        )


class C3SXGate(ControlledGate):
    """The 3-qubit controlled sqrt-X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        *,
        angle: Optional[ParameterValueType] = None,
    ):
        """Create a new 3-qubit controlled sqrt-X gate.

        Args:
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.
            angle (float): DEPRECATED. The angle used in the controlled-U1 gates. An angle of π/8
                yields the sqrt(X) gates, an angle of π/4 the 3-qubit controlled X gate.
        """
        super().__init__(
            "c3sx", 4, [], num_ctrl_qubits=3, label=label, ctrl_state=ctrl_state, base_gate=SXGate()
        )

        if angle is not None:
            warnings.warn(
                "The angle argument is deprecated as of Qiskit Terra 0.17.0 and will "
                "be removed no earlier than 3 months after the release date.",
                DeprecationWarning,
                stacklevel=2,
            )

        if angle is None:
            angle = numpy.pi / 8

        self._angle = angle

    def _define(self):
        """
        gate c3sqrtx a,b,c,d
        {
            h d; cu1(pi/8) a,d; h d;
            cx a,b;
            h d; cu1(-pi/8) b,d; h d;
            cx a,b;
            h d; cu1(pi/8) b,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
            cx b,c;
            h d; cu1(-pi/8) c,d; h d;
            cx a,c;
            h d; cu1(pi/8) c,d; h d;
        }
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import CU1Gate

        q = QuantumRegister(4, name="q")
        # pylint: disable=invalid-unary-operand-type
        rules = [
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[0], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[1], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[1], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[2], q[3]], []),
            (HGate(), [q[3]], []),
        ]
        qc = QuantumCircuit(q)
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        """Invert this gate. The C3X is its own inverse."""
        # pylint: disable=invalid-unary-operand-type
        if self._angle is not None:
            angle = -self._angle
        else:
            angle = None

        return C3SXGate(angle=angle, ctrl_state=self.ctrl_state)


class RC3XGate(Gate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.
    """

    def __init__(self, label: Optional[str] = None):
        """Create a new RC3X gate."""
        super().__init__("rcccx", 4, [], label=label)

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

        q = QuantumRegister(4, "q")
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
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def __array__(self, dtype=None):
        """Return a numpy.array for the RC3X gate."""
        return numpy.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
                [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=dtype,
        )


class MCXGate(ControlledGate):
    """The general, multi-controlled X gate."""

    # pylint: disable=unused-argument
    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        synthesis: Optional["MCXSynthesis"] = None,
        _name="mcx",
    ):
        """Create a new MCX instance.

        Depending on the number of controls and which mode of the MCX, this creates an
        explicit CX, CCX, C3X or C4X instance or a generic MCX gate.

        The synthesis argument refers to the synthesis algorithm for implementing the gate.
        """

        # The CXGate and CCXGate will be implemented for all modes of the MCX, and
        # the C3XGate and C4XGate will be implemented in the MCXGrayCode class.
        explicit = {1: CXGate, 2: CCXGate}

        if num_ctrl_qubits in explicit:
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class(label=label, ctrl_state=ctrl_state)
            return gate

        return super().__new__(cls)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
        synthesis=None,
        _name="mcx",
    ):
        """Create new MCX gate."""
        if synthesis is None:
            # pylint: disable=cyclic-import
            from qiskit.synthesis.mcx_synthesis import MCXSynthesisGrayCode

            synthesis = MCXSynthesisGrayCode()
        self.synthesis = synthesis

        num_ancilla_qubits = self.synthesis.get_num_ancilla_qubits(num_ctrl_qubits)

        super().__init__(
            _name,
            num_ctrl_qubits + 1 + num_ancilla_qubits,
            [],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(),
        )

    def inverse(self):
        """Invert this gate. The MCX is its own inverse."""
        return MCXGate(
            num_ctrl_qubits=self.num_ctrl_qubits,
            ctrl_state=self.ctrl_state,
            synthesis=self.synthesis,
            _name=self.name,
        )

    def _define(self):
        """The standard definition used the Gray code implementation."""
        self.definition = self.synthesis.synthesize(self)

    @property
    def num_ancilla_qubits(self):
        """The number of ancilla qubits."""
        return self.synthesis.get_num_ancilla_qubits(self.num_ctrl_qubits)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a multi-controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(
            num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits,
            label=label,
            ctrl_state=new_ctrl_state,
            synthesis=self.synthesis,
            _name=self.name,
        )
        gate.base_gate.label = self.label
        return gate

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "noancilla") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        This static method might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        # pylint: disable=cyclic-import
        from qiskit.synthesis.mcx_synthesis import mcx_mode_to_num_ancilla_qubits

        return mcx_mode_to_num_ancilla_qubits(num_ctrl_qubits, mode)

    def __array__(self, dtype=None):
        """Return a numpy.array for the MCX gate."""

        # Though, we can only handle the case that there are no ancilla qubits
        # (which includes the previous cases for C3XGate and C4XGate)
        if self.num_ancilla_qubits == 0:
            mat = _compute_control_matrix(
                self.base_gate.to_matrix(), self.num_ctrl_qubits, ctrl_state=self.ctrl_state
            )
            if dtype:
                return numpy.asarray(mat, dtype=dtype)
            return mat
        else:
            return None


# The following explicit classes remain for backward compatibility.
# The __new__ method is modified to return the MCX gate with the matching synthesis algorithm
# (or, as previously, an explicit few-qubit gate).
# These classes still inherit from MCXGate and in particular from ControlledGate.


class C4XGate(ControlledGate):
    """The 3-qubit controlled X gate.
    The default MCXSynthesisGrayCode algorithm implements the optimized method for 4 control qubits.
    """

    # pylint: disable=signature-differs
    def __new__(
        cls,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new MCX instance"""
        # pylint: disable=cyclic-import
        from qiskit.synthesis.mcx_synthesis import MCXSynthesisGrayCode

        gate = MCXGate(
            num_ctrl_qubits=4,
            label=label,
            ctrl_state=ctrl_state,
            synthesis=MCXSynthesisGrayCode(),
            _name="mcx",
        )
        return gate

    # pylint: disable=unused-argument
    def __init__(
        self,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Dummy (required for tests that inspect signature)"""
        pass


class C3XGate(ControlledGate):
    """The 3-qubit controlled X gate.
    The default MCXSynthesisGrayCode algorithm implements the optimized method for 3 control qubits.
    """

    # pylint: disable=signature-differs
    def __new__(
        cls,
        angle: Optional[ParameterValueType] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        if angle is not None:
            return C3SXGate(label, ctrl_state, angle=angle)

        from qiskit.synthesis.mcx_synthesis import MCXSynthesisGrayCode

        gate = MCXGate(
            num_ctrl_qubits=3,
            label=label,
            ctrl_state=ctrl_state,
            synthesis=MCXSynthesisGrayCode(),
            _name="mcx",
        )
        return gate

    # pylint: disable=unused-argument
    def __init__(
        self,
        angle: Optional[ParameterValueType] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Dummy (required for tests that inspect signature)"""
        pass


class MCXGrayCode(MCXGate):
    r"""Implement the multi-controlled X gate using the Gray code.

    This delegates the implementation to the MCU1 gate, since :math:`X = H \cdot U1(\pi) \cdot H`.
    """

    # pylint: disable=signature-differs
    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new MCXGrayCode instance"""
        # if 1 to 4 control qubits, create explicit gates
        # explicit = {1: CXGate, 2: CCXGate, 3: C3XGate, 4: C4XGate}
        explicit = {1: CXGate, 2: CCXGate}
        if num_ctrl_qubits in explicit:
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class(label=label, ctrl_state=ctrl_state)
        else:
            # pylint: disable=cyclic-import
            from qiskit.synthesis.mcx_synthesis import MCXSynthesisGrayCode

            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                synthesis=MCXSynthesisGrayCode(),
                _name="mcx_gray",
            )
        return gate


class MCXRecursive(MCXGate):
    """Implement the multi-controlled X gate using recursion.

    Using a single ancilla qubit, the multi-controlled X gate is recursively split onto
    four sub-registers. This is done until we reach the 3- or 4-controlled X gate since
    for these we have a concrete implementation that do not require ancillas.
    """

    # pylint: disable=signature-differs
    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new MCXRecursive instance"""

        # if 1 to 2 control qubits, create explicit gates
        explicit = {1: CXGate, 2: CCXGate}
        if num_ctrl_qubits in explicit:
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class(label=label, ctrl_state=ctrl_state)
        else:
            # pylint: disable=cyclic-import
            from qiskit.synthesis.mcx_synthesis import MCXSynthesisRecursive

            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                synthesis=MCXSynthesisRecursive(),
                _name="mcx_recursive",
            )
        return gate


class MCXVChain(MCXGate):
    """Implement the multi-controlled X gate using a V-chain of CX gates."""

    # pylint: disable=signature-differs
    def __new__(
        cls,
        num_ctrl_qubits: Optional[int] = None,
        dirty_ancillas: bool = False,  # pylint: disable=unused-argument
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create a new MCX instance with v-chain synthesis algorithm."""

        # if 1 to 2 control qubits, create explicit gates
        explicit = {1: CXGate, 2: CCXGate}
        if num_ctrl_qubits in explicit:
            gate_class = explicit[num_ctrl_qubits]
            gate = gate_class(label=label, ctrl_state=ctrl_state)
        else:
            # pylint: disable=cyclic-import
            from qiskit.synthesis.mcx_synthesis import MCXSynthesisVChain

            gate = MCXGate(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                synthesis=MCXSynthesisVChain(dirty_ancillas),
                _name="mcx_vchain",
            )
        return gate
