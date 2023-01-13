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

"""Synthesis of two-qubit unitaries using at most 3 applications of the sqrt(iSWAP) gate."""


import numpy as np
import cmath

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.circuit.library import (SiSwapGate, RXXGate, RYYGate, RZZGate, RZGate, RYGate,
                                    XGate, YGate, ZGate, SGate, SdgGate)


_EPS = 1e-10


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

    def __init__(self, euler_basis: list = ['u']):
        # decomposer for the local single-qubit gates
        from qiskit.transpiler.passes import Optimize1qGatesDecomposition
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

        # in the case that 2 x SiSwap gates are needed
        if abs(z) <= x - y + _EPS:
            V = _interleaving_single_qubit_gates(x, y, z)

            v_decomp = TwoQubitWeylDecomposition(Operator(V))

            D1 = Operator(v_decomp.K1r)
            D2 = Operator(v_decomp.K1l)
            E1 = Operator(v_decomp.K2r)
            E2 = Operator(v_decomp.K2l)

            ret_c = QuantumCircuit(2)
            ret_c.append(B1, [0])
            ret_c.append(B2, [1])

            ret_c.append(E1.adjoint(), [0])
            ret_c.append(E2.adjoint(), [1])
            ret_c.compose(V, inplace=True)
            ret_c.append(D1.adjoint(), [0])
            ret_c.append(D2.adjoint(), [1])

            ret_c.append(A1, [0])
            ret_c.append(A2, [1])

        # in the case that 3 SiSwap gates are needed
        else:
            # CAN(x, y, z) ~ CAN(x, y, -z)†
            # so we decompose the adjoint, replace SiSwap with a template in
            # terms of SiSwap†, then invert the whole thing
            if z < 0:
                inverse_decomposition = self.__call__(Operator(unitary).adjoint())
                inverse_decomposition_with_siswap_dg = QuantumCircuit(2)
                for instruction in inverse_decomposition:
                    if isinstance(instruction.operation, SiSwapGate):
                        inverse_decomposition_with_siswap_dg.z(0)
                        inverse_decomposition_with_siswap_dg.append(SiSwapGate().inverse(), [0, 1])
                        inverse_decomposition_with_siswap_dg.z(0)
                    else:
                        inverse_decomposition_with_siswap_dg.append(instruction)

                ret_c = inverse_decomposition_with_siswap_dg.inverse()
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
                # here we actually make the inverse of the circuit because we want
                # the final result to have SQiSW not SQiSW\dg
                follow = QuantumCircuit(2)
                # canonical gate: (x, y, z) --> (x-pi/8, y-pi/8, z)
                follow = follow.compose(SiSwapGate(), [0, 1])

                eigenphase_crossing = False

                if x > np.pi/8:
                    # (x, y, z) --> (x, y-pi/8, z-pi/8)
                    follow = follow.compose(YGate().power(1/2), [0], front=True)
                    follow = follow.compose(YGate().power(1/2), [1], front=True)
                    follow = follow.compose(YGate().power(-1/2), [0])
                    follow = follow.compose(YGate().power(-1/2), [1])
                    # eigenphase crossing: a_2 - pi/4 < a_3 + pi/4
                    # (x, y, z) --> (x, z-pi/8, y-pi/8)
                    if y + z < np.pi/4:
                        eigenphase_crossing = True
                else:
                    # (x, y, z) --> (x+pi/8, y, z-pi/8)
                    follow = follow.compose(XGate().power(1/2), [0], front=True)
                    follow = follow.compose(XGate().power(1/2), [1], front=True)
                    follow = follow.compose(XGate().power(-1/2), [0])
                    follow = follow.compose(XGate().power(-1/2), [1])
                    follow = follow.compose(ZGate(), [0], front=True)
                    follow = follow.compose(ZGate(), [0])
                    # eigenphase crossing: a_2 - pi/4 < a_3
                    # (x, y, z) --> (x+pi/8, z-pi/8, y)
                    if y + z < np.pi/8:
                        eigenphase_crossing = True

                # eigenphase crossing:
                if eigenphase_crossing:
                    follow = follow.compose(XGate().power(1/2), [0], front=True)
                    follow = follow.compose(XGate().power(1/2), [1], front=True)
                    follow = follow.compose(XGate().power(-1/2), [0])
                    follow = follow.compose(XGate().power(-1/2), [1])

                # now the operator in the red region can be decomposed using 2 x SQiSW
                red = nonred.compose(follow.inverse(), [0, 1], inplace=False)
                c_2_sqisw = self.__call__(Operator(red))

                # now write u in terms of 3 x SQiSW
                ret_c = QuantumCircuit(2)
                ret_c = ret_c.compose(c_2_sqisw, [0, 1])
                ret_c = ret_c.compose(follow, [0, 1])
                ret_c.append(A1, [0])
                ret_c.append(A2, [1])

        phase_diff = cmath.phase(Operator(unitary).data[0][0] / Operator(ret_c).data[0][0])
        ret_c.global_phase += phase_diff

        return self._decomposer1q(ret_c)


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
    V.append(SiSwapGate(), [0, 1])
    V.rz(γ, 0)
    V.rx(α, 0)
    V.rz(γ, 0)
    V.rx(β, 1)
    V.append(SiSwapGate(), [0, 1])

    # the returned circuit is the SQiSW sandwich
    return V


def _canonicalize(a, b, c):
    """
    Decompose an arbitrary gate into one SQiSW and one L(x′, y′, z′)
    where (x′, y′, z′) ∈ W′ and output the coefficients (x′, y′, z′)
    and the interleaving single qubit rotations.
    """
    A1 = Operator(IGate())
    A2 = Operator(IGate())
    B1 = Operator(RYGate(-np.pi/2))
    B2 = Operator(RYGate(np.pi/2))
    C1 = Operator(RYGate(np.pi/2))
    C2 = Operator(RYGate(-np.pi/2))

    s = 1 if c >= 0 else -1

    # a_ corresponds to a' in the paper, and so on
    a_ = a
    b_ = b
    c_ = abs(c)

    if a > np.pi/8:
        b_ -= np.pi/8
        c_ -= np.pi/8
        B1 = Operator(RZGate(np.pi/2)) @ B1
        B2 = Operator(RZGate(-np.pi/2)) @ B2
        C1 = C1 @ Operator(RZGate(-np.pi/2))
        C2 = C2 @ Operator(RZGate(np.pi/2))
    else:
        a_ += np.pi/8
        c_ -= np.pi/8

    if abs(b_) < abs(c_):
        b_, c_ = -c_, -b_
        A1 = Operator(RXGate(np.pi/2))
        A2 = Operator(RXGate(-np.pi/2))
        B1 = Operator(RXGate(-np.pi/2)) @ B1
        B2 = Operator(RXGate(np.pi/2)) @ B2
    if s < 0:
        c_ = -c_
        A1 = Operator(ZGate()) @ A1 @ Operator(ZGate())
        B1 = Operator(ZGate()) @ B1 @ Operator(ZGate())
        C1 = Operator(ZGate()) @ C1 @ Operator(ZGate())

    return a_,b_,c_, A1, A2, B1, B2, C1, C2


