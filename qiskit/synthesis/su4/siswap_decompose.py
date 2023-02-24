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
from qiskit.circuit.library import SiSwapGate, XGate, YGate, ZGate
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

    This basis is attractive in that it is shorter/cheaper than iSWAP and generates a large
    volume (~79%) of Haar-random SU(4) unitaries with only two applications, and an even
    larger volume when approximation is enabled.

    Args:
        euler_basis (list(str)): single-qubit gates basis in the decomposition.

    Reference:
        1. C. Huang et al,
           Towards ultra-high fidelity quantum operations:
           SQiSW gate as a native two-qubit gate (2021)
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

        circuit = _can_circuit(*p)
        circuit = circuit.compose(B1.to_instruction(), [0], front=True)
        circuit = circuit.compose(B2.to_instruction(), [1], front=True)
        circuit = circuit.compose(A1.to_instruction(), [0])
        circuit = circuit.compose(A2.to_instruction(), [1])

        # FIXME: cleaner to track global phase during construction
        i = np.where(~np.isclose(np.ravel(Operator(circuit).data), 0.0))[0][0]
        phase_diff = cmath.phase(Operator(unitary).data.flat[i] / Operator(circuit).data.flat[i])
        circuit.global_phase += phase_diff

        return self._decomposer1q(circuit)


def _can_circuit(x, y, z):
    """
    Find a circuit that implements the canonical gate
    CAN(x, y, z) := RXX(x) . RYY(y) . RZZ(z)
    """
    circuit = QuantumCircuit(2)
    if np.allclose([x, y, z], _ID):
        pass
    elif np.allclose([x, y, z], _SISWAP):
        circuit.append(SiSwapGate(), [0, 1])
    elif abs(z) <= x - y + _EPS:  # red region
        V = _interleaving_single_qubit_gates(x, y, z)
        v_decomp = TwoQubitWeylDecomposition(Operator(V))
        can = _remove_pre_post(V, v_decomp)
        circuit = circuit.compose(can, [0, 1])

        # Due to the symmetry of Weyl chamber CAN(pi/4, y, z) ~ CAN(pi/4, y, -z)
        # we might get a V operator that implements CAN(pi/4, y, -z) instead
        # we catch this case and fix it by local gates.
        if not np.isclose(v_decomp.c, z):
            circuit = circuit.compose(XGate(), [0], front=True)
            circuit = circuit.compose(ZGate(), [1], front=True)
            circuit = circuit.compose(YGate(), [1])
    else:
        if z < 0:  # blue region
            # CAN(x, y, z) ~ CAN(x, y, -z)†
            # so we decompose the adjoint, replace SiSwap with a template in
            # terms of SiSwap†, then invert the whole thing.
            inverse_circuit = _can_circuit(x, y, -z)
            inverse_circuit_with_siswap_dg = QuantumCircuit(2)
            for instruction in inverse_circuit:
                if isinstance(instruction.operation, SiSwapGate):
                    inverse_circuit_with_siswap_dg.z(0)
                    inverse_circuit_with_siswap_dg.append(SiSwapGate().inverse(), [0, 1])
                    inverse_circuit_with_siswap_dg.z(0)
                else:
                    inverse_circuit_with_siswap_dg.append(instruction)

            V = inverse_circuit_with_siswap_dg.inverse()
            v_decomp = TwoQubitWeylDecomposition(Operator(V))
            can = _remove_pre_post(V, v_decomp)
            circuit = circuit.compose(can, [0, 1])
        else:
            # make a circuit using 1 SiSwap that is able to bring a red point to here
            follow = QuantumCircuit(2)

            # x -> x + pi/8
            # y -> y + pi/8
            follow = follow.compose(SiSwapGate(), [0, 1])

            eigenphase_crossing = False
            if x > np.pi / 8:  # green region
                # y -> y + pi/8
                # z -> z + pi/8
                follow = follow.compose(YGate().power(1 / 2), [0], front=True)
                follow = follow.compose(YGate().power(1 / 2), [1], front=True)
                follow = follow.compose(YGate().power(-1 / 2), [0])
                follow = follow.compose(YGate().power(-1 / 2), [1])
                if y + z < np.pi / 4:  # eigenphase crossing condition: a_2 - pi/4 < a_3 + pi/4
                    eigenphase_crossing = True
                # corresponding red coordinates
                y -= np.pi / 8
                z -= np.pi / 8
            else:  # purple region
                # x -> x - pi/8
                # z -> z + pi/8
                follow = follow.compose(XGate().power(1 / 2), [0], front=True)
                follow = follow.compose(XGate().power(1 / 2), [1], front=True)
                follow = follow.compose(XGate().power(-1 / 2), [0])
                follow = follow.compose(XGate().power(-1 / 2), [1])
                follow = follow.compose(ZGate(), [0], front=True)
                follow = follow.compose(ZGate(), [0])
                if y + z < np.pi / 8:  # eigenphase crossing condition: a_2 - pi/4 < a_3
                    eigenphase_crossing = True
                # corresponding red coordinates
                x += np.pi / 8
                z -= np.pi / 8

            if eigenphase_crossing:
                y, z = -z, -y

            # final 3xSiSwap circuit: red --> fix crossing --> green or purple
            red_decomp = _can_circuit(x, y, z)
            circuit = circuit.compose(red_decomp, [0, 1])
            if eigenphase_crossing:
                # y, z -> -z, -y
                circuit = circuit.compose(XGate().power(1 / 2), [0], front=True)
                circuit = circuit.compose(XGate().power(1 / 2), [1], front=True)
                circuit = circuit.compose(XGate().power(-1 / 2), [0])
                circuit = circuit.compose(XGate().power(-1 / 2), [1])
                circuit = circuit.compose(XGate(), [0], front=True)
                circuit = circuit.compose(XGate(), [0])
            circuit = circuit.compose(follow, [0, 1])

    return circuit


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


def _remove_pre_post(circuit, decomp):
    """
    Given a circuit and its Weyl decomposition, multiply local gates before and after
    it to get a new circuit which is equivalent to RXX.RYY.RZZ (up to global phase).
    """
    D1 = Operator(decomp.K1r)
    D2 = Operator(decomp.K1l)
    E1 = Operator(decomp.K2r)
    E2 = Operator(decomp.K2l)

    new_circuit = QuantumCircuit(2)
    new_circuit.append(E1.adjoint(), [0])
    new_circuit.append(E2.adjoint(), [1])
    new_circuit.compose(circuit, inplace=True)
    new_circuit.append(D1.adjoint(), [0])
    new_circuit.append(D2.adjoint(), [1])

    return new_circuit
