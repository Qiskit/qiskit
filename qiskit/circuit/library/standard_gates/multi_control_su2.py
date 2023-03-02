# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Multi-controlled SU(2) gate.
"""

from typing import Union, List
from cmath import isclose
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates.x import MCXVChain
from qiskit.circuit import Gate, Qubit


def _check_su2(matrix):
    return isclose(np.linalg.det(matrix), 1.0)


class MCSU2Gate(Gate):
    r"""
    Linear-depth multi-controlled gate for special unitary single-qubit gates.

    This decomposition for SU(2) gates with multiple controls does not use auxiliary qubits.
    An n-qubit gate implemented with this method will have at most 20n - 38 CNOTs if
    the number of qubits is odd, or 20n - 42 CNOTs if the number of qubits is even.
    """

    def __init__(self, su2_matrix, num_ctrl_qubits, ctrl_state: str = None):
        _check_su2(su2_matrix)

        self.su2_matrix = su2_matrix
        self.num_ctrl_qubits = num_ctrl_qubits
        self.ctrl_state = ctrl_state

        super().__init__("mcsu2", self.num_ctrl_qubits + 1, [], "mcsu2")

    def _define(self):
        controls = QuantumRegister(self.num_ctrl_qubits)
        target = QuantumRegister(1)
        self.definition = QuantumCircuit(controls, target)

        is_main_diag_real = isclose(self.su2_matrix[0, 0].imag, 0.0) and isclose(
            self.su2_matrix[1, 1].imag, 0.0
        )
        is_secondary_diag_real = isclose(self.su2_matrix[0, 1].imag, 0.0) and isclose(
            self.su2_matrix[1, 0].imag, 0.0
        )

        if not is_main_diag_real and not is_secondary_diag_real:
            # U = V D V^-1, where the entries of the diagonal D are the eigenvalues
            # `eig_vals` of U and the column vectors of V are the eigenvectors
            # `eig_vecs` of U. These columns are orthonormal and the main diagonal
            # of V is real-valued.
            eig_vals, eig_vecs = np.linalg.eig(self.su2_matrix)

            x_vecs, z_vecs = self._get_x_z(eig_vecs)
            x_vals, z_vals = self._get_x_z(np.diag(eig_vals))

            self.half_linear_depth_mcv(
                x_vecs, z_vecs, controls, target, self.ctrl_state, inverse=True
            )
            self.linear_depth_mcv(
                x_vals, z_vals, controls, target, self.ctrl_state, general_su2_optimization=True
            )
            self.half_linear_depth_mcv(x_vecs, z_vecs, controls, target, self.ctrl_state)

        else:
            x, z = self._get_x_z(self.su2_matrix)

            if not is_secondary_diag_real:
                self.definition.h(target)

            self.linear_depth_mcv(x, z, controls, target, self.ctrl_state)

            if not is_secondary_diag_real:
                self.definition.h(target)

    @staticmethod
    def _get_x_z(su2):
        is_secondary_diag_real = isclose(su2[0, 1].imag, 0.0) and isclose(su2[1, 0].imag, 0.0)

        if is_secondary_diag_real:
            x = su2[0, 1]
            z = su2[1, 1]
        else:
            x = -su2[0, 1].real
            z = su2[1, 1] - su2[0, 1].imag * 1.0j

        return x, z

    def linear_depth_mcv(
        self,
        x,
        z,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str = None,
        general_su2_optimization=False,
    ):
        alpha_r = np.sqrt((np.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
        alpha_i = z.imag / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
        alpha = alpha_r + 1.0j * alpha_i
        beta = x / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))

        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])

        # S gate definition
        s_gate = QuantumCircuit(1)
        s_gate.unitary(s_op, 0)

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.0))
        k_2 = int(np.floor(num_ctrl / 2.0))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if not general_su2_optimization:
            mcx_1 = MCXVChain(
                num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1
            ).definition
            self.definition.append(mcx_1, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
        self.definition.append(s_gate, [target])

        mcx_2 = MCXVChain(
            num_ctrl_qubits=k_2,
            dirty_ancillas=True,
            ctrl_state=ctrl_state_k_2,
            # action_only=general_su2_optimization
        ).definition
        self.definition.append(
            mcx_2.inverse(), controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1]
        )
        self.definition.append(s_gate.inverse(), [target])

        mcx_3 = MCXVChain(
            num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1
        ).definition
        self.definition.append(mcx_3, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
        self.definition.append(s_gate, [target])

        mcx_4 = MCXVChain(
            num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2
        ).definition
        self.definition.append(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
        self.definition.append(s_gate.inverse(), [target])

    def half_linear_depth_mcv(
        self,
        x,
        z,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str = None,
        inverse: bool = False,
    ):
        alpha_r = np.sqrt((z.real + 1.0) / 2.0)
        alpha_i = z.imag / np.sqrt(2 * (z.real + 1.0))
        alpha = alpha_r + 1.0j * alpha_i

        beta = x / np.sqrt(2 * (z.real + 1.0))

        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])

        # S gate definition
        s_gate = QuantumCircuit(1)
        s_gate.unitary(s_op, 0)

        # Hadamard equivalent definition
        h_gate = QuantumCircuit(1)
        h_gate.unitary(np.array([[-1, 1], [1, 1]]) * 1 / np.sqrt(2), 0)

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.0))
        k_2 = int(np.floor(num_ctrl / 2.0))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if inverse:
            self.definition.h(target)

            self.definition.append(s_gate, [target])
            mcx_2 = MCXVChain(
                num_ctrl_qubits=k_2,
                dirty_ancillas=True,
                ctrl_state=ctrl_state_k_2,
                # action_only=True
            ).definition
            self.definition.append(mcx_2, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])

            self.definition.append(s_gate.inverse(), [target])

            self.definition.append(h_gate, [target])

        else:
            mcx_1 = MCXVChain(
                num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1
            ).definition
            self.definition.append(mcx_1, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
            self.definition.append(h_gate, [target])

            self.definition.append(s_gate, [target])

            mcx_2 = MCXVChain(
                num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2
            ).definition
            self.definition.append(mcx_2, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
            self.definition.append(s_gate.inverse(), [target])

            self.definition.h(target)
