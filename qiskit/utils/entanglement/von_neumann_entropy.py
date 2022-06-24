import numpy as np


def compute_vn_entropy_qiskit(ket, N):
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
    import qiskit.quantum_info as qi

    qubit = list(range(N))  # list of qubits to trace over

    vn_entropy = 0

    for k in range(N):
        rho_k = qi.partial_trace(ket, qubit[:k] + qubit[k + 1 :]).data
        vn_entropy += qi.entropy(rho_k, base=np.exp(1))
    Q = vn_entropy / N
    return Q
