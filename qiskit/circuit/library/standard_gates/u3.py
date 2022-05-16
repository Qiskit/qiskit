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

"""Two-pulse single-qubit gate."""

from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister


class U3Gate(Gate):
    r"""Generic single-qubit rotation gate with 3 Euler angles.

    Implemented using two X90 pulses on IBM Quantum systems:

    .. math::
        U3(\theta, \phi, \lambda) =
            RZ(\phi) RX(-\pi/2) RZ(\theta) RX(\pi/2) RZ(\lambda)

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
                \cos\left(\th\right)          & -e^{i\lambda}\sin\left(\th\right) \\
                e^{i\phi}\sin\left(\th\right) & e^{i(\phi+\lambda)}\cos\left(\th\right)
            \end{pmatrix}

    **Examples:**

    .. math::

        U3\left(\theta, -\frac{\pi}{2}, \frac{\pi}{2}\right) = RX(\theta)

    .. math::

        U3(\theta, 0, 0) = RY(\theta)
    """

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None,
    ):
        """Create new U3 gate."""
        super().__init__("u3", 1, [theta, phi, lam], label=label)

    def inverse(self):
        r"""Return inverted U3 gate.

        :math:`U3(\theta,\phi,\lambda)^{\dagger} =U3(-\theta,-\lambda,-\phi)`)
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Return a (multi-)controlled-U3 gate.

        Args:
            num_ctrl_qubits (int): number of control qubits.
            label (str or None): An optional label for the gate [Default: None]
            ctrl_state (int or str or None): control state expressed as integer,
                string (e.g. '110'), or None. If None, use all 1s.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CU3Gate(*self.params, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
            return gate
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def _define(self):
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.u(self.params[0], self.params[1], self.params[2], 0)
        self.definition = qc

    def __array__(self, dtype=None):
        """Return a Numpy.array for the U3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        cos = numpy.cos(theta / 2)
        sin = numpy.sin(theta / 2)
        return numpy.array(
            [
                [cos, -numpy.exp(1j * lam) * sin],
                [numpy.exp(1j * phi) * sin, numpy.exp(1j * (phi + lam)) * cos],
            ],
            dtype=dtype,
        )


class CU3Gate(ControlledGate):
    r"""Controlled-U3 gate (3-parameter two-qubit gate).

    This is a controlled version of the U3 gate (generic single qubit rotation).
    It is restricted to 3 parameters, and so cannot cover generic two-qubit
    controlled gates).

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──────■──────
             ┌─────┴─────┐
        q_1: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    **Matrix representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        CU3(\theta, \phi, \lambda)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| +
            U3(\theta,\phi,\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0                   & 0 & 0 \\
                0 & \cos(\th)           & 0 & -e^{i\lambda}\sin(\th) \\
                0 & 0                   & 1 & 0 \\
                0 & e^{i\phi}\sin(\th)  & 0 & e^{i(\phi+\lambda)}\cos(\th)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────────┐
            q_0: ┤ U3(ϴ,φ,λ) ├
                 └─────┬─────┘
            q_1: ──────■──────

        .. math::

            CU3(\theta, \phi, \lambda)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I +
                |1\rangle\langle 1| \otimes U3(\theta,\phi,\lambda) =
                \begin{pmatrix}
                    1 & 0   & 0                  & 0 \\
                    0 & 1   & 0                  & 0 \\
                    0 & 0   & \cos(\th)          & -e^{i\lambda}\sin(\th) \\
                    0 & 0   & e^{i\phi}\sin(\th) & e^{i(\phi+\lambda)}\cos(\th)
                \end{pmatrix}
    """

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None,
        ctrl_state: Optional[Union[str, int]] = None,
    ):
        """Create new CU3 gate."""
        super().__init__(
            "cu3",
            2,
            [theta, phi, lam],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U3Gate(theta, phi, lam),
        )

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
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .x import CXGate  # pylint: disable=cyclic-import

        #      ┌───────────────┐
        # q_0: ┤ U1(λ/2 + φ/2) ├──■─────────────────────────────■─────────────────
        #      ├───────────────┤┌─┴─┐┌───────────────────────┐┌─┴─┐┌─────────────┐
        # q_1: ┤ U1(λ/2 - φ/2) ├┤ X ├┤ U3(-0/2,0,-λ/2 - φ/2) ├┤ X ├┤ U3(0/2,φ,0) ├
        #      └───────────────┘└───┘└───────────────────────┘└───┘└─────────────┘
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (U1Gate((self.params[2] + self.params[1]) / 2), [q[0]], []),
            (U1Gate((self.params[2] - self.params[1]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U3Gate(self.params[0] / 2, self.params[1], 0), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def inverse(self):
        r"""Return inverted CU3 gate.

        :math:`CU3(\theta,\phi,\lambda)^{\dagger} =CU3(-\theta,-\phi,-\lambda)`)
        """
        return CU3Gate(
            -self.params[0], -self.params[2], -self.params[1], ctrl_state=self.ctrl_state
        )

    def __array__(self, dtype=None):
        """Return a numpy.array for the CU3 gate."""
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        cos = numpy.cos(theta / 2)
        sin = numpy.sin(theta / 2)
        if self.ctrl_state:
            return numpy.array(
                [
                    [1, 0, 0, 0],
                    [0, cos, 0, -numpy.exp(1j * lam) * sin],
                    [0, 0, 1, 0],
                    [0, numpy.exp(1j * phi) * sin, 0, numpy.exp(1j * (phi + lam)) * cos],
                ],
                dtype=dtype,
            )
        else:
            return numpy.array(
                [
                    [cos, 0, -numpy.exp(1j * lam) * sin, 0],
                    [0, 1, 0, 0],
                    [numpy.exp(1j * phi) * sin, 0, numpy.exp(1j * (phi + lam)) * cos, 0],
                    [0, 0, 0, 1],
                ],
                dtype=dtype,
            )


def _generate_gray_code(num_bits):
    """Generate the gray code for ``num_bits`` bits."""
    if num_bits <= 0:
        raise ValueError("Cannot generate the gray code for less than 1 bit.")
    result = [0]
    for i in range(num_bits):
        result += [x + 2**i for x in reversed(result)]
    return [format(x, "0%sb" % num_bits) for x in result]


def _gray_code_chain(q, num_ctrl_qubits, gate):
    """Apply the gate to the the last qubit in the register ``q``, controlled on all
    preceding qubits. This function uses the gray code to propagate down to the last qubit.

    Ported and adapted from Aqua (github.com/Qiskit/qiskit-aqua),
    commit 769ca8d, file qiskit/aqua/circuits/gates/multi_control_u1_gate.py.
    """
    from .x import CXGate

    rule = []
    q_controls, q_target = q[:num_ctrl_qubits], q[num_ctrl_qubits]
    gray_code = _generate_gray_code(num_ctrl_qubits)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index("1")

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                rule.append((CXGate(), [q_controls[pos], q_controls[lm_pos]], []))
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    rule.append((CXGate(), [q_controls[idx], q_controls[lm_pos]], []))
        # check parity
        if pattern.count("1") % 2 == 0:
            # inverse
            rule.append((gate.inverse(), [q_controls[lm_pos], q_target], []))
        else:
            rule.append((gate, [q_controls[lm_pos], q_target], []))
        last_pattern = pattern

    return rule
