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
from typing import Optional, List, Union
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.circuit.library import SiSwapGate, XGate, YGate, ZGate, SGate, SdgGate
from qiskit.synthesis.su4.utils import find_min_point, average_infidelity


_EPS = 1e-12

# the polytope accessible by two applications of the SiSwapGate
# (red region in paper)
_ID = np.array([0, 0, 0])
_CNOT = np.array([np.pi / 4, 0, 0])
_ISWAP = np.array([np.pi / 4, np.pi / 4, 0])
_MID = np.array([np.pi / 4, np.pi / 8, np.pi / 8])
_MIDDG = np.array([np.pi / 4, np.pi / 8, -np.pi / 8])
_SISWAP = np.array([np.pi / 8, np.pi / 8, 0])
_POLYTOPE = np.array([_ID, _CNOT, _ISWAP, _MID, _MIDDG])


class SiSwapDecomposer:
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

    def __call__(
        self,
        unitary: Union[Operator, np.ndarray],
        basis_fidelity: Optional[float] = 1.0,
        approximate: bool = True,
        *,
        _num_basis_uses: Optional[int] = None,
    ) -> QuantumCircuit:
        """Decompose a two-qubit unitary into using the sqrt(iSWAP) gate.

        Args:
            unitary: a 4x4 unitary to synthesize.
            basis_fidelity: Fidelity of the iSWAP gate.
            approximate: Approximates if basis fidelities are less than 1.0.
            _num_basis_uses: force a particular approximation by passing a number in [0, 3].

        Returns:
            QuantumCircuit: Synthesized circuit.
        """
        if not approximate:
            basis_fidelity = 1.0

        u_decomp = TwoQubitWeylDecomposition(Operator(unitary))

        x = u_decomp.a
        y = u_decomp.b
        z = u_decomp.c

        A1 = Operator(u_decomp.K1r)
        A2 = Operator(u_decomp.K1l)
        B1 = Operator(u_decomp.K2r)
        B2 = Operator(u_decomp.K2l)

        p = np.array([x, y, z])

        if abs(z) <= x - y + _EPS:
            polytope_projection = p
        else:
            polytope_projection = find_min_point([v - p for v in _POLYTOPE]) + p

        candidate_points = [
            _ID,  # 0 applications
            _SISWAP,  # 1 application
            polytope_projection,  # 2 applications
            p,  # 3 applications
        ]

        if _num_basis_uses is None:
            expected_fidelities = [
                (1 - average_infidelity(p, q)) * basis_fidelity**i
                for i, q in enumerate(candidate_points)
            ]
            best_nbasis = int(np.argmax(expected_fidelities))  # tiebreaks with smallest
        else:
            best_nbasis = _num_basis_uses

        p = candidate_points[best_nbasis]

        x = p[0]
        y = p[1]
        z = p[2]

        # in the case that 0 SiSwap gate is needed
        if best_nbasis == 0:
            circuit = QuantumCircuit(2)
            circuit.append(B1, [0])
            circuit.append(B2, [1])
            circuit.append(A1, [0])
            circuit.append(A2, [1])

        # in the case that 1 SiSwap gate is needed
        elif best_nbasis == 1:
            circuit = QuantumCircuit(2)
            circuit.append(B1, [0])
            circuit.append(B2, [1])
            circuit.append(SGate(), [0])
            circuit.append(SGate(), [1])
            circuit.append(SiSwapGate(), [0, 1])
            circuit.append(SdgGate(), [0])
            circuit.append(SdgGate(), [1])
            circuit.append(A1, [0])
            circuit.append(A2, [1])

        # in the case that 2 SiSwap gates are needed
        elif best_nbasis == 2:  # red region
            V = _interleaving_single_qubit_gates(x, y, z)
            v_decomp = TwoQubitWeylDecomposition(Operator(V))

            # Due to the symmetry of Weyl chamber CAN(pi/4, y, z) ~ CAN(pi/4, y, -z)
            # we might get a V operator that implements CAN(pi/4, y, -z) instead
            # we catch this case and fix it by local gates
            deviant = False
            if not np.isclose(v_decomp.c, z):
                deviant = True

            D1 = Operator(v_decomp.K1r)
            D2 = Operator(v_decomp.K1l)
            E1 = Operator(v_decomp.K2r)
            E2 = Operator(v_decomp.K2l)

            circuit = QuantumCircuit(2)
            circuit.append(B1, [0])
            circuit.append(B2, [1])

            if deviant:
                circuit.x(0)
                circuit.z(1)
            circuit.append(E1.adjoint(), [0])
            circuit.append(E2.adjoint(), [1])
            circuit.compose(V, inplace=True)
            circuit.append(D1.adjoint(), [0])
            circuit.append(D2.adjoint(), [1])
            if deviant:
                circuit.y(1)

            circuit.append(A1, [0])
            circuit.append(A2, [1])

        # in the case that 3 SiSwap gates are needed
        else:
            if z < 0:  # blue region
                # CAN(x, y, z) ~ CAN(x, y, -z)†
                # so we decompose the adjoint, replace SiSwap with a template in
                # terms of SiSwap†, then invert the whole thing
                inverse_decomposition = self.__call__(Operator(unitary).adjoint())
                inverse_decomposition_with_siswap_dg = QuantumCircuit(2)
                for instruction in inverse_decomposition:
                    if isinstance(instruction.operation, SiSwapGate):
                        inverse_decomposition_with_siswap_dg.z(0)
                        inverse_decomposition_with_siswap_dg.append(SiSwapGate().inverse(), [0, 1])
                        inverse_decomposition_with_siswap_dg.z(0)
                    else:
                        inverse_decomposition_with_siswap_dg.append(instruction)

                circuit = inverse_decomposition_with_siswap_dg.inverse()
            # follow unitary u with a circuit consisting of 1 x SiSwap
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
                follow = follow.compose(SiSwapGate(), [0, 1])

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

                # now we can land in the red region which can be decomposed using 2 x SiSwap
                red = nonred.compose(follow.inverse(), [0, 1], inplace=False)
                red_decomp = self.__call__(Operator(red))

                # now write u in terms of 3 x SiSwap
                circuit = QuantumCircuit(2)
                circuit = circuit.compose(red_decomp, [0, 1])
                circuit = circuit.compose(follow, [0, 1])
                circuit.append(A1, [0])
                circuit.append(A2, [1])

        # FIXME: there must be a cleaner way to track global phase
        i = np.where(~np.isclose(np.ravel(Operator(circuit).data), 0.0))[0][0]
        phase_diff = cmath.phase(Operator(unitary).data.flat[i] / Operator(circuit).data.flat[i])
        circuit.global_phase += phase_diff

        return self._decomposer1q(circuit)


def _interleaving_single_qubit_gates(x, y, z):
    """
    Find the single-qubit gates given the interaction coefficients
    (x, y, z) ∈ W′ when sandwiched by two SiSwap gates.
    Return the SiSwap sandwich.
    """
    C = np.sin(x + y - z) * np.sin(x - y + z) * np.sin(-x - y - z) * np.sin(-x + y + z)
    if abs(C) < _EPS:
        C = 0.0

    α = np.arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) + 2 * np.sqrt(C))

    β = np.arccos(np.cos(2 * x) - np.cos(2 * y) + np.cos(2 * z) - 2 * np.sqrt(C))

    s = 4 * (np.cos(x) ** 2) * (np.cos(z) ** 2) * (np.sin(y) ** 2)
    t = np.cos(2 * x) * np.cos(2 * y) * np.cos(2 * z)
    sign_z = 1 if z >= 0 else -1
    γ = np.arccos(sign_z * np.sqrt(s / (s + t)))

    # create V (sandwich) operator
    V = QuantumCircuit(2)
    V.append(SiSwapGate(), [0, 1])
    V.rz(γ, 0)
    V.rx(α, 0)
    V.rz(γ, 0)
    V.rx(β, 1)
    V.append(SiSwapGate(), [0, 1])

    # the returned circuit is the SiSwap sandwich
    return V
