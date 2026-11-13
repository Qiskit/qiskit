# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType


def rzx_zz2(theta: ParameterValueType | None = None):
    """RZX-based template for CX - PhaseGate - CX.

    .. code-block:: text

        global phase: π
                                                                                    »
          q_0: ──■────────────■─────────────────────────────────────────────────────»
               ┌─┴─┐┌──────┐┌─┴─┐┌───────┐┌─────────┐┌─────────┐┌─────────┐┌───────┐»
          q_1: ┤ X ├┤ P(ϴ) ├┤ X ├┤ P(-ϴ) ├┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├┤ Rx(ϴ) ├»
               └───┘└──────┘└───┘└───────┘└─────────┘└─────────┘└─────────┘└───────┘»
          «     ┌──────────┐
          «q_0: ┤0         ├─────────────────────────────────
          «     │  Rzx(-ϴ) │┌─────────┐┌─────────┐┌─────────┐
          «q_1: ┤1         ├┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├
          «     └──────────┘└─────────┘└─────────┘└─────────┘
    """
    if theta is None:
        theta = Parameter("ϴ")

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.p(theta, 1)
    qc.cx(0, 1)
    qc.p(-1 * theta, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)

    qc.rx(theta, 1)
    qc.rzx(-1 * theta, 0, 1)
    # Hadamard
    qc.rz(np.pi / 2, 1)
    qc.rx(np.pi / 2, 1)
    qc.rz(np.pi / 2, 1)
    # Gate content has unitary e^{i*pi} * I == -I; global_phase = pi makes Operator(qc) == I exactly.
    qc.global_phase = np.pi

    return qc
