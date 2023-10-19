# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit XX-YY gate."""
import math
from cmath import exp
from math import pi
from typing import Optional

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.ry import RYGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit.circuit.library.standard_gates.s import SdgGate, SGate
from qiskit.circuit.library.standard_gates.sx import SXdgGate, SXGate
from qiskit.circuit.library.standard_gates.x import CXGate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister


class XXMinusYYGate(Gate):
    r"""XX-YY interaction gate.

    A 2-qubit parameterized XX-YY interaction. Its action is to induce
    a coherent rotation by some angle between :math:`|00\rangle` and :math:`|11\rangle`.

    **Circuit Symbol:**

    .. parsed-literal::

             ┌───────────────┐
        q_0: ┤0              ├
             │  (XX-YY)(θ,β) │
        q_1: ┤1              ├
             └───────────────┘

    **Matrix Representation:**

    .. math::

        \newcommand{\th}{\frac{\theta}{2}}

        R_{XX-YY}(\theta, \beta) q_0, q_1 =
          RZ_1(\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX-YY}{2}\right) \cdot RZ_1(-\beta) =
            \begin{pmatrix}
                \cos\left(\th\right)             & 0 & 0 & -i\sin\left(\th\right)e^{-i\beta}  \\
                0                     & 1 & 0 & 0  \\
                0                     & 0 & 1 & 0  \\
                -i\sin\left(\th\right)e^{i\beta} & 0 & 0 & \cos\left(\th\right)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in adding the (optional) phase defined
        by :math:`beta` on q_1. Instead, if we apply it on (q_1, q_0), the
        phase is added on q_0. If :math:`beta` is set to its default value
        of :math:`0`, the gate is equivalent in big and little endian.

        .. parsed-literal::

                 ┌───────────────┐
            q_0: ┤1              ├
                 │  (XX-YY)(θ,β) │
            q_1: ┤0              ├
                 └───────────────┘

        .. math::

            \newcommand{\th}{\frac{\theta}{2}}

            R_{XX-YY}(\theta, \beta) q_1, q_0 =
            RZ_0(\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX-YY}{2}\right) \cdot RZ_0(-\beta) =
                \begin{pmatrix}
                    \cos\left(\th\right)             & 0 & 0 & -i\sin\left(\th\right)e^{i\beta}  \\
                    0                     & 1 & 0 & 0  \\
                    0                     & 0 & 1 & 0  \\
                    -i\sin\left(\th\right)e^{-i\beta} & 0 & 0 & \cos\left(\th\right)
                \end{pmatrix}
    """

    def __init__(
        self,
        theta: ParameterValueType,
        beta: ParameterValueType = 0,
        label: Optional[str] = "(XX-YY)",
        *,
        duration=None,
        unit="dt",
    ):
        """Create new XX-YY gate.

        Args:
            theta: The rotation angle.
            beta: The phase angle.
            label: The label of the gate.
        """
        super().__init__("xx_minus_yy", 2, [theta, beta], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate xx_minus_yy(theta, beta) a, b {
            rz(-beta) b;
            rz(-pi/2) a;
            sx a;
            rz(pi/2) a;
            s b;
            cx a, b;
            ry(theta/2) a;
            ry(-theta/2) b;
            cx a, b;
            sdg b;
            rz(-pi/2) a;
            sxdg a;
            rz(pi/2) a;
            rz(beta) b;
        }
        """
        theta, beta = self.params
        register = QuantumRegister(2, "q")
        circuit = QuantumCircuit(register, name=self.name)
        a, b = register
        rules = [
            (RZGate(-beta), [b], []),
            (RZGate(-pi / 2), [a], []),
            (SXGate(), [a], []),
            (RZGate(pi / 2), [a], []),
            (SGate(), [b], []),
            (CXGate(), [a, b], []),
            (RYGate(theta / 2), [a], []),
            (RYGate(-theta / 2), [b], []),
            (CXGate(), [a, b], []),
            (SdgGate(), [b], []),
            (RZGate(-pi / 2), [a], []),
            (SXdgGate(), [a], []),
            (RZGate(pi / 2), [a], []),
            (RZGate(beta), [b], []),
        ]
        for instr, qargs, cargs in rules:
            circuit._append(instr, qargs, cargs)

        self.definition = circuit

    def inverse(self):
        """Inverse gate."""
        theta, beta = self.params
        return XXMinusYYGate(-theta, beta)

    def __array__(self, dtype=complex):
        """Gate matrix."""
        theta, beta = self.params
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return np.array(
            [
                [cos, 0, 0, -1j * sin * exp(-1j * beta)],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-1j * sin * exp(1j * beta), 0, 0, cos],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float):
        """Raise gate to a power."""
        theta, beta = self.params
        return XXMinusYYGate(exponent * theta, beta)
