import numpy as np


def compute_Q_ptrace_qiskit(ket, N):
    """
        Args:
        =====
        ket : numpy.ndarray or list
            Vector of amplitudes in 2**N dimensions
        N : int
            Number of qubits
    ​
        Returns:
        ========
        Q : float
            Q value for input ket
    """
    # Runtime imports to avoid circular imports causeed by QuantumInstance
    # getting initialized by imported utils/__init__ which is imported
    # by qiskit.circuit
    from qiskit.quantum_info import partial_trace

    entanglement_sum = 0
    for k in range(N):

        trace_over = [q for q in range(N) if q != k]
        rho_k = partial_trace(ket, trace_over).data
        entanglement_sum += np.real((np.linalg.matrix_power(rho_k, 2)).trace())

    Q = 2 * (1 - (1 / N) * entanglement_sum)

    return Q
