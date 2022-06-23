import numpy as np
import qutip
# from qiskit.quantum_info import partial_trace

def compute_Q_ptrace_qutip(ket, N):
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
    entanglement_sum = 0
    for k in range(N):
        # print('value of n', k, 'PTrace: ',ket.ptrace([k])**2 )
        rho_k_sq = ket.ptrace([k])**2
        entanglement_sum += rho_k_sq.tr()  
   
    Q = 2*(1 - (1/N)*entanglement_sum)
    return Q

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
    
    # qubit = list(range(N)) # list of qubits to trace over
    
    entanglement_sum = 0
    for k in range(N):
        trace_over = [q for q in range(N) if q != k] 
        # p_trace = qi.partial_trace(ket,qubit[:k]+qubit[k+1:])
        rho_k = partial_trace(ket, trace_over).data
        # rho_k_sq = np.dot(p_trace,p_trace)
        entanglement_sum += np.real((np.linalg.matrix_power(rho_k, 2)).trace())
    Q = 2*(1 - (1/N)*entanglement_sum)
    return Q