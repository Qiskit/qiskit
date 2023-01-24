# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name, non-ascii-name

"""Synthesis of two-qubit unitaries using at most 3 applications of the sqrt(iSWAP) gate."""

import cmath
from typing import Optional, List
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.circuit.library import (
    SQiSWGate,
    XGate,
    YGate,
    ZGate,
)


_EPS = 1e-10


class SQiSWDecomposer:
    """
    A class for decomposing 2-qubit unitaries into at most 3 uses of the sqrt(iSWAP) gate.

    Args:
        euler_basis (list(str)): single-qubit gates basis in the decomposition

    Reference:
        1. C. Huang et al,
           *Towards ultra-high fidelity quantum operations:
           SQiSW gate as a native two-qubit gate (2021)*
           `arXiv:2105.06074 <https://arxiv.org/abs/2105.06074>`_
    """

    def __init__(self, euler_basis: Optional[List[str]]):
        # decomposer for the local single-qubit gates
        from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
            Optimize1qGatesDecomposition,  # pylint: disable=cyclic-import
        )

        euler_basis = euler_basis or ["u"]
        self._decomposer1q = Optimize1qGatesDecomposition(euler_basis)

    def __call__(self, unitary, basis_fidelity=1.0, approximate=True):
        """Decompose a two-qubit unitary into using the sqrt(iSWAP) gate.

        Args:
            unitary (Operator or ndarray): a 4x4 unitary to synthesize.
            basis_fidelity (float): Fidelity of the B gate.
            approximate (bool): Approximates if basis fidelities are less than 1.0.

        Returns:
            QuantumCircuit: Synthesized circuit.
        """
        u_decomp = TwoQubitWeylDecomposition(Operator(unitary))

        x = u_decomp.a
        y = u_decomp.b
        z = u_decomp.c

        A1 = Operator(u_decomp.K1r)
        A2 = Operator(u_decomp.K1l)
        B1 = Operator(u_decomp.K2r)
        B2 = Operator(u_decomp.K2l)

        # in the case that 2 x SQiSW gates are needed
        if abs(z) <= x - y + _EPS:  # red region
            V = _interleaving_single_qubit_gates(x, y, z)

            v_decomp = TwoQubitWeylDecomposition(Operator(V))

            D1 = Operator(v_decomp.K1r)
            D2 = Operator(v_decomp.K1l)
            E1 = Operator(v_decomp.K2r)
            E2 = Operator(v_decomp.K2l)

            circuit = QuantumCircuit(2)
            circuit.append(B1, [0])
            circuit.append(B2, [1])

            circuit.append(E1.adjoint(), [0])
            circuit.append(E2.adjoint(), [1])
            circuit.compose(V, inplace=True)
            circuit.append(D1.adjoint(), [0])
            circuit.append(D2.adjoint(), [1])

            circuit.append(A1, [0])
            circuit.append(A2, [1])

        # in the case that 3 SQiSW gates are needed
        else:
            if z < 0:  # blue region
                # CAN(x, y, z) ~ CAN(x, y, -z)†
                # so we decompose the adjoint, replace SQiSW with a template in
                # terms of SQiSW†, then invert the whole thing
                inverse_decomposition = self.__call__(Operator(unitary).adjoint())
                inverse_decomposition_with_sqisw_dg = QuantumCircuit(2)
                for instruction in inverse_decomposition:
                    if isinstance(instruction.operation, SQiSWGate):
                        inverse_decomposition_with_sqisw_dg.z(0)
                        inverse_decomposition_with_sqisw_dg.append(SQiSWGate().inverse(), [0, 1])
                        inverse_decomposition_with_sqisw_dg.z(0)
                    else:
                        inverse_decomposition_with_sqisw_dg.append(instruction)

                circuit = inverse_decomposition_with_sqisw_dg.inverse()
            # follow unitary u with a circuit consisting of 1 x SQiSW
            # that takes the coordinate into the red region
            else:
                # first remove the post-rotation to u to be able to
                # play with angles of RXX.RYY.RZZ by appending gates
                nonred = QuantumCircuit(2)
                nonred.append(Operator(unitary), [0, 1])
                nonred.append(A1.adjoint(), [0])
                nonred.append(A2.adjoint(), [1])

                # make a circuit that changes the angles of RXX.RYY.RZZ as desired
                follow = QuantumCircuit(2)

                # starting with a single sqrt(iSWAP) gate: RXX(pi/4).RYY(pi/4).RZZ(0)
                follow = follow.compose(SQiSWGate(), [0, 1])

                # figure out the appropriate conjugations that change RXX/RYY/RZZ angles
                eigenphase_crossing = False
                if x > np.pi / 8:  # green region
                    # RXX(0).RYY(pi/4).RZZ(pi/4)
                    follow = follow.compose(YGate().power(1 / 2), [0], front=True)
                    follow = follow.compose(YGate().power(1 / 2), [1], front=True)
                    follow = follow.compose(YGate().power(-1 / 2), [0])
                    follow = follow.compose(YGate().power(-1 / 2), [1])
                    # RXX(0).RYY(pi/4).RZZ(pi/4)
                    if y + z < np.pi / 4:  # eigenphase crossing condition: a_2 - pi/4 < a_3 + pi/4
                        eigenphase_crossing = True
                else:  # purple region
                    # RXX(-pi/4).RYY(0).RZZ(pi/4)
                    follow = follow.compose(XGate().power(1 / 2), [0], front=True)
                    follow = follow.compose(XGate().power(1 / 2), [1], front=True)
                    follow = follow.compose(XGate().power(-1 / 2), [0])
                    follow = follow.compose(XGate().power(-1 / 2), [1])
                    follow = follow.compose(ZGate(), [0], front=True)
                    follow = follow.compose(ZGate(), [0])
                    # RXX(-pi/4).RYY(pi/4).RZZ(0)
                    if y + z < np.pi / 8:  # eigenphase crossing condition: a_2 - pi/4 < a_3
                        eigenphase_crossing = True

                if eigenphase_crossing:
                    follow = follow.compose(XGate().power(1 / 2), [0], front=True)
                    follow = follow.compose(XGate().power(1 / 2), [1], front=True)
                    follow = follow.compose(XGate().power(-1 / 2), [0])
                    follow = follow.compose(XGate().power(-1 / 2), [1])

                # now we can land in the red region which can be decomposed using 2 x SQiSW
                red = nonred.compose(follow.inverse(), [0, 1], inplace=False)
                red_decomp = self.__call__(Operator(red))

                # now write u in terms of 3 x SQiSW
                circuit = QuantumCircuit(2)
                circuit = circuit.compose(red_decomp, [0, 1])
                circuit = circuit.compose(follow, [0, 1])
                circuit.append(A1, [0])
                circuit.append(A2, [1])

        phase_diff = cmath.phase(Operator(unitary).data[0][0] / Operator(circuit).data[0][0])
        circuit.global_phase += phase_diff

        return self._decomposer1q(circuit)


def _interleaving_single_qubit_gates(x, y, z):
    """
    Find the single-qubit gates given the interaction coefficients
    (x, y, z) ∈ W′ when sandwiched by two SQiSW gates.
    Return the SQiSW sandwich.
    """
    C = np.sin(x + y - z) * np.sin(x - y + z) * np.sin(-x - y - z) * np.sin(-x + y + z)
    C = max(C, 0)

    α = np.arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) + 2 * np.sqrt(C))

    β = np.arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) - 2 * np.sqrt(C))

    s = 4 * (np.cos(x) ** 2) * (np.cos(z) ** 2) * (np.sin(y) ** 2)
    t = np.cos(2 * x) * np.cos(2 * y) * np.cos(2 * z)
    sign_z = 1 if z >= 0 else -1
    γ = np.arccos(sign_z * np.sqrt(s / (s + t)))

    # create V operator
    V = QuantumCircuit(2)
    V.append(SQiSWGate(), [0, 1])
    V.rz(γ, 0)
    V.rx(α, 0)
    V.rz(γ, 0)
    V.rx(β, 1)
    V.append(SQiSWGate(), [0, 1])

    # the returned circuit is the SQiSW sandwich
    return V
