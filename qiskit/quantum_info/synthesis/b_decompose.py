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

"""Synthesis of two-qubit unitaries using at most 2 applications of the B gate."""


import numpy as np
import cmath

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import BGate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.transpiler.passes import Optimize1qGatesDecomposition


class BDecomposer:
    """
    A class for optimal decomposition of 2-qubit unitaries into at most 2 uses of the B gate.

    Args:
        euler_basis (list(str)): single-qubit gates basis in the decomposition

    Reference:
        1. Zhang, Vala, Sastry and Whaley,
           *Minimum construction of two-qubit quantum operations*,
           Physical review letters 93.2 (2004)
           `arXiv:quant-ph/0312193 <https://arxiv.org/abs/quant-ph/0312193>`_
    """

    def __init__(self, euler_basis: list = ['u']):
        # decomposer for the local single-qubit gates
        self._decomposer1q = Optimize1qGatesDecomposition(euler_basis)

    def __call__(self, unitary, basis_fidelity=1.0, approximate=True):
        """Decompose a two-qubit unitary into 2 B gates plus local gates.

        Args:
            unitary (Operator or ndarray): a 4x4 unitary to synthesize.
            basis_fidelity (float): Fidelity of the B gate.
            approximate (bool): Approximates if basis fidelities are less than 1.0.

        Returns:
            QuantumCircuit: Synthesized circuit.
        """
        # get canonical coordinates of the target unitry
        # paper's coordinates are scaled by a factor of 2 from qiskit's coordinates
        weyl_decomposition_u = TwoQubitWeylDecomposition(unitary)
        c1 = weyl_decomposition_u.a * 2
        c2 = weyl_decomposition_u.b * 2
        c3 = weyl_decomposition_u.c * 2

        # implement the B gate sandwich (non-local part of the synthesis)
        r = (np.sin(c2/2) * np.cos(c3/2)) ** 2
        t = np.cos(c2) * np.cos(c3) / (1 - 2*r)
        s = 1 if c3 < 0 else -1   # Clamp out-of-range floating point error.
        r = max(0.0, r)
        t = max(0.0, min(1, t))
        β1 = np.arccos(1 - 4*r)
        β2 = np.arcsin(np.sqrt(t))
        b_sandwich = QuantumCircuit(2)
        b_sandwich.append(BGate(), [0, 1])
        b_sandwich.ry(s * c1, 0)
        b_sandwich.rz(-β2, 1)
        b_sandwich.ry(-β1, 1)
        b_sandwich.rz(-β2, 1)
        b_sandwich.append(BGate(), [0, 1])

        # implement pre-/post-rotations to the B gate sandwich (local part of synthesis)
        weyl_decomposition_b = TwoQubitWeylDecomposition(Operator(b_sandwich))
        circuit = QuantumCircuit(2)
        circuit.append(Operator(weyl_decomposition_u.K2r), [0])
        circuit.append(Operator(weyl_decomposition_u.K2l), [1])
        circuit.append(Operator(weyl_decomposition_b.K2r).adjoint(), [0])
        circuit.append(Operator(weyl_decomposition_b.K2l).adjoint(), [1])
        circuit.compose(b_sandwich, [0, 1], inplace=True)
        circuit.append(Operator(weyl_decomposition_b.K1r).adjoint(), [0])
        circuit.append(Operator(weyl_decomposition_b.K1l).adjoint(), [1])
        circuit.append(Operator(weyl_decomposition_u.K1r), [0])
        circuit.append(Operator(weyl_decomposition_u.K1l), [1])
        circuit = self._decomposer1q(circuit)
        
        # fix global phase
        phase_diff = cmath.phase(Operator(unitary).data[0][0] / Operator(circuit).data[0][0])
        circuit.global_phase += phase_diff
        return circuit

