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

"""Meyer-Wallach Measure Module"""

import numpy as np


def compute_ptrace(ket: np.ndarray, num_qubits: int) -> float:
    """Return the values of entanglement capability using meyer-wallach
        measure.
    Args:
        ket : (numpy.ndarray or list);Vector of amplitudes in 2**N dimensions
        num_qubits : (int)Number of qubits

    Returns:
        q: float; Q value for input ket
    """
    # Runtime imports to avoid circular imports causeed by QuantumInstance
    # getting initialized by imported utils/__init__ which is imported
    # by qiskit.circuit
    from qiskit.quantum_info import partial_trace

    entanglement_sum = 0
    for k in range(num_qubits):

        trace_over = [q for q in range(num_qubits) if q != k]
        rho_k = partial_trace(ket, trace_over).data
        entanglement_sum += np.real((np.linalg.matrix_power(rho_k, 2)).trace())

    q = 2 * (1 - (1 / num_qubits) * entanglement_sum)

    return q
