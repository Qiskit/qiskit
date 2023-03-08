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
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library.standard_gates.x import MCXVChain


def linear_depth_mcv(
    circuit,
    x,
    z,
    controls: Union[QuantumRegister, List[Qubit]],
    target: Qubit,
    ctrl_state: str = None,
):
    """
    Apply multi-controlled SU(2) gate [[z*, x],[-x, z]] with x real and z complex.
    https://arxiv.org/abs/2302.06377

    Args:
        circuit (QuantumCircuit): The QuantumCircuit object to apply the diagonal operator on.
        x (float): real parameter for the single-qubit operators
        z (complex): complex parameter for the single-qubit operators
        controls (QuantumRegister or list(Qubit)): The list of control qubits
        target (Qubit): The target qubit
        ctrl_state (str): control state of the operator SU(2) operator U
    """

    alpha_r = np.sqrt((np.sqrt((z.real + 1.0) / 2.0) + 1.0) / 2.0)
    alpha_i = z.imag / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))
    alpha = alpha_r + 1.0j * alpha_i
    beta = x / (2.0 * np.sqrt((z.real + 1.0) * (np.sqrt((z.real + 1.0) / 2.0) + 1.0)))

    s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])

    # S gate definition
    s_gate = QuantumCircuit(1)
    s_gate.unitary(s_op, 0)
    s_gate = s_gate.to_gate()

    num_ctrl = len(controls)
    k_1 = int(np.ceil(num_ctrl / 2.0))
    k_2 = int(np.floor(num_ctrl / 2.0))

    ctrl_state_k_1 = None
    ctrl_state_k_2 = None

    if ctrl_state is not None:
        str_ctrl_state = f"{ctrl_state:0{num_ctrl}b}"
        ctrl_state_k_1 = str_ctrl_state[::-1][:k_1][::-1]
        ctrl_state_k_2 = str_ctrl_state[::-1][k_1:][::-1]

    mcx_1 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_1, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.append(s_gate, [target])

    mcx_2 = MCXVChain(
        num_ctrl_qubits=k_2,
        dirty_ancillas=True,
        ctrl_state=ctrl_state_k_2,
        # action_only=general_su2_optimization # Requires PR #9687
    )
    circuit.append(mcx_2.inverse(), controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.append(s_gate.inverse(), [target])

    mcx_3 = MCXVChain(num_ctrl_qubits=k_1, dirty_ancillas=True, ctrl_state=ctrl_state_k_1)
    circuit.append(mcx_3, controls[:k_1] + [target] + controls[k_1 : 2 * k_1 - 2])
    circuit.append(s_gate, [target])

    mcx_4 = MCXVChain(num_ctrl_qubits=k_2, dirty_ancillas=True, ctrl_state=ctrl_state_k_2)
    circuit.append(mcx_4, controls[k_1:] + [target] + controls[k_1 - k_2 + 2 : k_1])
    circuit.append(s_gate.inverse(), [target])
