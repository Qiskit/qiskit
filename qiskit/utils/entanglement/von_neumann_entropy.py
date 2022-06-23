import numpy as np
from numpy import e
# import qiskit.quantum_info as qi
import qutip


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

    qubit = list(range(N)) # list of qubits to trace over
    
    vn_entropy = 0

    for k in range(N):
        rho_k = qi.partial_trace(ket, qubit[:k]+qubit[k+1:]).data
        vn_entropy += qi.entropy(rho_k, base = np.exp(1))
    Q = vn_entropy/N
    return Q

def compute_vn_entropy_qutip(ket, N):
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

    # print("ket:", ket)
    ket = qutip.Qobj(ket, dims=[[2]*(N), [1]*(N)]).unit()
    # print('KET=  ', ket)
    
    vn_entropy = 0

    for k in range(N):
        rho_k = ket.ptrace([k])
        vn_entropy += qutip.entropy_vn(rho_k, base=e)
    Q = vn_entropy/N
    return Q