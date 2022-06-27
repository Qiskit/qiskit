# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Von-Neumann Entropy Module"""

import numpy as np


def compute_vn_entropy(ket: np.ndarray, num_qubits: int) -> float:
    """Returns the entangling capabilility using von-neumann entropy.
    Args:
        ket : (numpy.ndarray or list);Vector of amplitudes in 2**N dimensions
        num_qubits : (int)Number of qubits

    Returns:
        q: float; Q value for input ket
    """

    # Runtime imports to avoid circular imports causeed by QuantumInstance
    # getting initialized by imported utils/__init__ which is imported
    # by qiskit.circuit
    import qiskit.quantum_info as qi

    qubit = list(range(num_qubits))  # list of qubits to trace over

    vn_entropy = 0

    for k in range(num_qubits):
        rho_k = qi.partial_trace(ket, qubit[:k] + qubit[k + 1 :]).data
        vn_entropy += qi.entropy(rho_k, base=np.exp(1))
    q = vn_entropy / num_qubits
    return q
