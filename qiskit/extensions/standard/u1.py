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
    """Diagonal single-qubit gate."""

    def __init__(self, theta, label=None):
        """Create new diagonal single-qubit gate."""
        super().__init__('u1', 1, [theta], label=label)

    def _define(self):
        from qiskit.extensions.standard.u3 import U3Gate
        definition = []
        q = QuantumRegister(1, 'q')
        rule = [
            (U3Gate(0, 0, self.params[0]), [q[0]], [])
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
            return CU1Gate(*self.params, num_ctrl_qubits)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return U1Gate(-self.params[0])

    def to_matrix(self):
        """Return a numpy.array for the U1 gate."""
        lam = self.params[0]
        lam = float(lam)
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=complex)


@deprecate_arguments({'q': 'qubit'})
def u1(self, theta, qubit, *, q=None):  # pylint: disable=invalid-name,unused-argument
    """Apply U1 gate with angle theta

    Applied to a specified qubit ``qubit``.
    :math:`u1(\\lambda) := diag(1, ei\\lambda) ∼ U(0, 0, \\lambda) = Rz(\\lambda)`
    where :math:`~` is equivalence up to a global phase.

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


class CU1Meta(type):
    """A metaclass to ensure that Cu1Gate and CU1Gate are of the same type.

    Can be removed when Cu1Gate gets removed.
    """
    @classmethod
    def __instancecheck__(mcs, inst):
        return type(inst) in {CU1Gate, Cu1Gate}  # pylint: disable=unidiomatic-typecheck


class CU1Gate(ControlledGate, metaclass=CU1Meta):
    """The controlled-u1 gate."""

    def __init__(self, theta, num_ctrl_qubits=1):
        """Create new cu1 gate."""
        super().__init__('cu1', num_ctrl_qubits + 1, [theta], num_ctrl_qubits=num_ctrl_qubits)
        self.base_gate = U1Gate(theta)

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        definition = []
        q = QuantumRegister(self.num_qubits, 'q')
        if self.num_ctrl_qubits == 1:
            from qiskit.extensions.standard.x import CXGate
            rule = [
                (U1Gate(self.params[0] / 2), [q[0]], []),
                (CXGate(), [q[0], q[1]], []),
                (U1Gate(-self.params[0] / 2), [q[1]], []),
                (CXGate(), [q[0], q[1]], []),
                (U1Gate(self.params[0] / 2), [q[1]], [])
            ]
        else:
            rule = _mcu1_rule(q, self.params[0], self.num_ctrl_qubits)

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
            return CU1Gate(*self.params, num_ctrl_qubits=self.num_ctrl_qubits + num_ctrl_qubits)
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label,
                               ctrl_state=ctrl_state)

    def inverse(self):
        """Invert this gate."""
        return CU1Gate(-self.params[0], num_ctrl_qubits=self.num_ctrl_qubits)

    def to_matrix(self):
        """Return a numpy.array for the multi-controlled U1 gate."""
        lam = self.params[0]
        if self.num_ctrl_qubits == 0:
            return U1Gate(lam).to_matrix()

        from qiskit.extensions.unitary import _compute_control_matrix
        base_mat = U1Gate(lam).to_matrix()
        return _compute_control_matrix(base_mat, self.num_ctrl_qubits, ctrl_state=self.ctrl_state)


class Cu1Gate(CU1Gate, metaclass=CU1Meta):
    """The deprecated CU1Gate class."""

    def __init__(self, theta):
        import warnings
        warnings.warn('The class Cu1Gate is deprecated as of 0.14.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the class CU1Gate instead.',
                      DeprecationWarning, stacklevel=2)
        super().__init__(theta)


@deprecate_arguments({'ctl': 'control_qubit',
                      'tgt': 'target_qubit'})
def cu1(self, theta, control_qubit, target_qubit,
        *, ctl=None, tgt=None):  # pylint: disable=unused-argument
    r"""Apply cU1 gate

    Applied from a specified control ``control_qubit`` to target
    ``target_qubit`` qubit  with angle theta. A cU1 gate implements a
    :math:`\theta` radian rotation of the qubit state vector about the z axis
    of the Bloch sphere when the control qubit is in state :math:`|1\rangle`.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            theta = Parameter('θ')
            circuit = QuantumCircuit(2)
            circuit.cu1(theta,0,1)
            circuit.draw()
    """
    return self.append(CU1Gate(theta), [control_qubit, target_qubit], [])


QuantumCircuit.cu1 = cu1


def mcu1(self, lam, control_qubits, target_qubit):
    """Apply multi-cU1 gate from specified controls (control_qubits) to target (target_qubit) qubit
    with angle ``lam``. A multi-cU1 gate implements a ``lam`` radian rotation of the qubit state
    vector about the z axis of the Bloch sphere when the control qubits are all in state |1>.

    Examples:

        Circuit Representation:

        .. jupyter-execute::

            from qiskit.circuit import QuantumCircuit, Parameter

            lam = Parameter('λ')
            circuit = QuantumCircuit(4)
            circuit.mcu1(lam, [0, 1, 2], 3)
            circuit.draw()
    """
    num_ctrl_qubits = len(control_qubits)
    return self.append(CU1Gate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])


QuantumCircuit.mcu1 = mcu1


def _mcu1_rule(q, lam, num_ctrl_qubits):
    """The gate definition of the multi-controlled U1 gate.

    Ported from Aqua (github.com/Qiskit/qiskit-aqua),
    commit 769ca8d, file qiskit/aqua/circuits/gates/multi_control_u1_gate.py.
    """
    rule = []
    q_controls = q[:num_ctrl_qubits]
    q_target = q[num_ctrl_qubits]
    if num_ctrl_qubits == 0:
        rule.append(
            (U1Gate(lam), [q_target], [])
        )
    elif num_ctrl_qubits == 1:
        rule.append(
            (CU1Gate(lam), q_controls + [q_target], [])
        )
    else:
        from qiskit.extensions.standard.x import CXGate
        from sympy.combinatorics.graycode import GrayCode
        gray_code = list(GrayCode(num_ctrl_qubits).generate_gray())
        last_pattern = None

        lam_angle = lam * (1 / (2**(num_ctrl_qubits - 1)))

        for pattern in gray_code:
            if '1' not in pattern:
                continue
            if last_pattern is None:
                last_pattern = pattern
            # find left most set bit
            lm_pos = list(pattern).index('1')

            # find changed bit
            comp = [i != j for i, j in zip(pattern, last_pattern)]
            if True in comp:
                pos = comp.index(True)
            else:
                pos = None
            if pos is not None:
                if pos != lm_pos:
                    rule.append(
                        (CXGate(), [q_controls[pos], q_controls[lm_pos]], [])
                    )
                else:
                    indices = [i for i, x in enumerate(pattern) if x == '1']
                    for idx in indices[1:]:
                        rule.append(
                            (CXGate(), [q_controls[idx], q_controls[lm_pos]], [])
                        )
            # check parity
            if pattern.count('1') % 2 == 0:
                # inverse
                rule.append(
                    (CU1Gate(-lam_angle), [q_controls[lm_pos], q_target], [])
                )
            else:
                rule.append(
                    (CU1Gate(lam_angle), [q_controls[lm_pos], q_target], [])
                )
            last_pattern = pattern

    return rule
