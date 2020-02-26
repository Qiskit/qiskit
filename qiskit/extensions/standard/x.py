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
Pauli X (bit-flip) gate.
"""
import numpy
from math import ceil
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
    """Pauli X (bit-flip) gate."""

    def __init__(self, label=None):
        """Create new X gate."""
        super().__init__('x', 1, [], label=label)

    def _define(self):
        """
        gate x a {
        u3(pi,0,pi) a;
        }
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
                return CXGate()
            if num_ctrl_qubits == 2:
                return CCXGate()
            if num_ctrl_qubits == 3:
                return CCCXGate()
            if num_ctrl_qubits == 4:
                return CCCCXGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return XGate()  # self-inverse

    def to_matrix(self):
        """Return a numpy.array for the X gate."""
        return numpy.array([[0, 1],
                            [1, 0]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def x(self, qubit, *, q=None):  # pylint: disable=unused-argument
    """Apply X gate to a specified qubit (qubit).
    An X gate implements a pi rotation of the qubit state vector about the
    x axis of the Bloch sphere.
    This gate is canonically used to implement a bit flip on the qubit state from |0⟩ to |1⟩,
    or vice versa.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(1)
            circuit.x(0)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.x import XGate
            XGate().to_matrix()
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
    """The controlled-X gate."""

    def __init__(self):
        """Create new cx gate."""
        super().__init__('cx', 2, [], num_ctrl_qubits=1)
        self.base_gate = XGate()

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
                return CCXGate()
            if num_ctrl_qubits == 2:
                return CCCXGate()
            if num_ctrl_qubits == 3:
                return CCCCXGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
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
    """Apply CX gate from a specified control (control_qubit) to target (target_qubit) qubit.
    A CX gate implements a pi rotation of the qubit state vector about the x axis
    of the Bloch sphere when the control qubit is in state |1>.
    This gate is canonically used to implement a bit flip on the qubit state from |0⟩ to |1⟩,
    or vice versa when the control qubit is in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(2)
            circuit.cx(0,1)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.x import CXGate
            CXGate().to_matrix()
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
    """The double-controlled-not gate, also called Toffoli gate."""

    def __init__(self):
        """Create new CCX gate."""
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
                return CCCXGate()
            if num_ctrl_qubits == 2:
                return CCCCXGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
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
    """Apply Toffoli (ccX) gate from two specified controls (control_qubit1 and control_qubit2)
    to target (target_qubit) qubit. This gate is canonically used to rotate the qubit state from
    |0⟩ to |1⟩, or vice versa when both the control qubits are in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(3)
            circuit.ccx(0,1,2)
            circuit.draw()

        Matrix Representation:

        .. jupyter-execute::

            from qiskit.extensions.standard.x import CCXGate
            CCXGate().to_matrix()
    """

    return self.append(CCXGate(),
                       [control_qubit1, control_qubit2, target_qubit], [])


# support both ccx and toffoli as methods of QuantumCircuit
QuantumCircuit.ccx = ccx
QuantumCircuit.toffoli = ccx


class CCCXGate(ControlledGate):
    """The 3-qubit controlled X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(self, angle=numpy.pi/4):
        """Create a new 3-qubit controlled X gate.

        Args:
            angle (float): The angle used in the controlled-U1 gates. An angle of π/4 yields the
                3-qubit controlled X gate, an angle of π/8 the 3-qubit controlled sqrt(X) gate.
        """
        super().__init__('cccx', 4, [])
        self.base_gate = XGate()
        self._angle = angle

    def _define(self):
        """
        gate cccx a,b,c,d
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

        gate cccsqrtx a,b,c,d
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
        """
        from qiskit.extensions.standard.u1 import CU1Gate
        definition = []
        q = QuantumRegister(4)
        rule = [
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[0], q[3]]),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[1], q[3]]),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[1]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[1], q[3]]),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[2], q[3]]),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[2], q[3]]),
            (HGate(), [q[3]], []),
            (CXGate(), [q[1], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(self._angle), [q[2], q[3]]),
            (HGate(), [q[3]], []),
            (CXGate(), [q[0], q[2]], []),
            (HGate(), [q[3]], []),
            (CU1Gate(-self._angle), [q[2], q[3]]),
            (HGate(), [q[3]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

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
            if self._angle == numpy.pi / 4 and num_ctrl_qubits == 1:
                return CCCCXGate()
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate. The CCCX is its own inverse."""
        return CCCXGate(angle=self._angle)


def cccx(self, control_qubit1, control_qubit2, control_qubit3, target_qubit):
    """Apply the 3-qubit controlled X (cccX) gate from four specified controls
    (control_qubit1..3) to target (target_qubit) qubit. This gate is canonically used to rotate the
    qubit state from |0⟩ to |1⟩, or vice versa when both the control qubits are in state |1⟩.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(4)
            circuit.cccx(0,1,2,3)
            circuit.draw()
    """

    return self.append(CCCXGate(),
                       [control_qubit1, control_qubit2, control_qubit3, target_qubit],
                       [])


QuantumCircuit.cccx = cccx


class CCCCXGate(ControlledGate):
    """The 4-qubit controlled X gate.

    This implementation is based on Page 21, Lemma 7.5, of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    def __init__(self):
        """Create a new 4-qubit controlled X gate."""
        super().__init__('ccccx', 5, [])
        self.base_gate = XGate()

    def _define(self):
        """
        gate ccccx a,b,c,d,e
        {
            h e; cu1(-pi/2) d,e; h e;
            cccx a,b,c,d;
            h d; cu1(pi/4) d,e; h d;
            cccx a,b,c,d;
            cccsqrtx a,b,c,e;
        }
        """
        from qiskit.extensions.standard.u1 import CU1Gate
        definition = []
        q = QuantumRegister(4)
        rule = [
            (HGate(), [q[4]], []),
            (CU1Gate(-numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (CCCXGate(), [q[0], q[1], q[2], q[3]], []),
            (HGate(), [q[4]], []),
            (CU1Gate(numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (CCCXGate(), [q[0], q[1], q[2], q[3]], []),
            (CCCXGate(numpy.pi / 8), [q[0], q[1], q[2], q[4]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate. The CCCCX is its own inverse."""
        return CCCCXGate()


def ccccx(self, control_qubit1, control_qubit2, control_qubit3, control_qubit4, target_qubit):
    """Apply the 4-qubit controlled X (ccccX) gate from four specified controls
    (control_qubit1..4) to target (target_qubit) qubit. This gate is canonically used to rotate the
    qubit state from |0⟩ to |1⟩, or vice versa when both the control qubits are in state |1⟩.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit import QuantumCircuit

            circuit = QuantumCircuit(5)
            circuit.ccccx(0,1,2,3,4)
            circuit.draw()
    """

    return self.append(CCCCXGate(),
                       [control_qubit1, control_qubit2, control_qubit3, control_qubit4,
                        target_qubit],
                       [])


QuantumCircuit.ccccx = ccccx


class MCXGate(ControlledGate):
    """The multi-controlled X gate."""

    def __init__(self, num_ctrl_qubits, mode='v-chain-clean-ancilla'):
        """Create a new multi-controlled X gate."""
        self._mode = mode
        num_qubits = 1 + num_ctrl_qubits + MCXGate.num_required_ancillas(num_ctrl_qubits, mode)
        super().__init__('mcx', num_qubits, [], num_ctrl_qubits=num_ctrl_qubits)

    @staticmethod
    def num_required_ancillas(num_ctrl_qubits, mode):
        """Return the number of required ancillas."""
        required_ancillas = {'v-chain-clean-ancilla': num_ctrl_qubits - 2,
                             'v-chain-dirty-ancilla': num_ctrl_qubits - 2,
                             'recursion': int(num_ctrl_qubits > 4),
                             'no-ancilla': 0}
        try:
            return required_ancillas[mode]
        except KeyError:
            raise ValueError('The mode "{}" is not supported. '.format(mode)
                             + 'Choose one of {}'.format(','.join(required_ancillas.keys())))

    @property
    def mode(self):
        """Return the strategy used to implement the multi-controlled X gate."""
        return self._mode

    def _define(self):
        """Define the multi-controlled X gate, according to the set strategy."""
        q = QuantumRegister(self.num_qubits)
        if self.num_ctrl_qubits == 0:
            rule = [(XGate(), [q[0]], [])]
        elif self.num_ctrl_qubits == 1:
            rule = [(CXGate(), [q[0], q[1]], [])]
        elif self.num_ctrl_qubits == 2:
            rule = [(CCXGate(), [q[0], q[1], q[2]], [])]
        elif self.num_ctrl_qubits == 3:
            rule = [(CCCXGate(), [q[0], q[1], q[2], q[3]], [])]
        elif self.num_ctrl_qubits == 4:
            rule = [(CCCCXGate(), [q[0], q[1], q[2], q[3], q[3]], [])]
        else:
            if self._mode == 'no-ancilla':
                rule = self._no_ancilla_rule(q)
            elif self._mode == 'recursion':
                q_state, q_ancilla = q[:-1], q[-1]
                rule = self._recursion_rule(q_state, q_ancilla)
            elif self._mode == 'v-chain-clean-ancilla':
                q_state, q_ancillas = q[:self.num_ctrl_qubits - 2], q[self.num_ctrl_qubits - 2:]
                rule = self._v_chain_rule(q_state, q_ancillas, dirty_ancillas=False)
            elif self._mode == 'v-chain-dirty-ancilla':
                q_state, q_ancillas = q[:self.num_ctrl_qubits - 2], q[self.num_ctrl_qubits - 2:]
                rule = self._v_chain_rule(q_state, q_ancillas, dirty_ancillas=True)

        definition = []
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def _no_ancilla_rule(self, q):
        """Get the rule if no ancilla is used."""
        from qiskit.extensions.standard.u1 import MCU1Gate
        rule = [
            (HGate(), [q[-1]], []),
            (MCU1Gate(numpy.pi, self.num_qubits - 1), q, []),
            (HGate(), [q[-1], []])
        ]
        return rule

    def _recursion_rule(self, q, q_ancilla):
        """Get the rule if the recursion strategy is used."""
        # recursion stop
        if len(q) == 4:
            return [(CCCXGate(), q, [])]
        if len(q) == 5:
            return [(CCCCXGate(), q, [])]

        # recurse
        num_controls = len(q) - 1
        middle = ceil(num_controls / 2)
        first_half = [*q[:middle], q_ancilla]
        second_half = [*q[middle:num_controls], q_ancilla, q[middle]]

        rule = []
        rule.append(self._recursion_rule(first_half, q_ancilla=q[middle]))
        rule.append(self._recursion_rule(second_half, q_ancilla=q[middle - 1]))
        rule.append(self._recursion_rule(first_half, q_ancilla=q[middle]))
        rule.append(self._recursion_rule(second_half, q_ancilla=q[middle - 1]))

        return rule

    def _v_chain_rule(self, q, q_ancillas, dirty_ancillas=False):
        """Get the rule for the V chain strategy."""
        if len(q_ancillas) < self.num_ctrl_qubits - 2:
            raise ValueError('At least {} ancillas are required.'.format(self.num_ctrl_qubits - 2))

        from qiskit.extensions.standard.u1 import U1Gate
        from qiskit.extensions.standard.u2 import U2Gate
        from qiskit.extensions.standard.rccx import RCCXGate

        q_controls, q_target = q[:-1], q[-1]
        if dirty_ancillas:
            i = self.num_ctrl_qubits - 3
            rule = [
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
            for j in reversed(range(2, self.num_ctrl_qubits - 1)):
                rule.append((RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]]))
                i -= 1
        else:
            rule = []

        rule.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[0]]))
        # i = 0
        for i, j in enumerate(list(range(2, self.num_ctrl_qubits - 1))):
            rule.append((RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]]))
            # i += 1

        if dirty_ancillas:
            sub_rule = [
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U1Gate(numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_controls[-1], q_ancillas[i]], []),
                (U1Gate(-numpy.pi / 4), [q_ancillas[i]], []),
                (CXGate(), [q_target, q_ancillas[i]], []),
                (U2Gate(0, numpy.pi), [q_target], []),
            ]
            for inst in sub_rule:
                rule.append(inst)
        else:
            rule.append((CCXGate(), [q_controls[-1], q_ancillas[i], q_target], []))

        for j in reversed(range(2, self.num_ctrl_qubits - 1)):
            rule.append((RCCXGate(), [q_controls[j], q_ancillas[i - 1], q_ancillas[i]]))
            i -= 1
        rule.append((RCCXGate(), [q_controls[0], q_controls[1], q_ancillas[i]]))

        if dirty_ancillas:
            for i, j in enumerate(list(range(2, self.num_ctrl_qubits - 1))):
                rule.append((RCCXGate(), [q_controls[j], q_ancillas[i], q_ancillas[i + 1]]))

        return rule
