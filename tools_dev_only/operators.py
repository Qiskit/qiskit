"""
Quantum tools and common operators.

These are simple methods for making common matrices used in quantum computing.

Author: Jay Gambetta
"""
import numpy as np


def destroy(dim):
    """Annihilation operator.

    dim = integer dimension (for qubits dim = 2**n where n is number of qubits)
    returns a complex numpy array
    """
    a = np.zeros((dim, dim), dtype=complex)
    for jj in range(dim):
        for kk in range(dim):
            if kk - jj == 1:
                a[jj, kk] = np.sqrt(jj + 1)
    return a


def operator_qi(gate, qubit, number_of_qubits):
    """Apply the single-qubit gate.

    gate is the single-qubit gate
    qubit is the qubit to apply it on counts from 0 and order
        is q_{n-1} ... otimes q_1 otimes q_0
    number_of_qubits is the number of qubits in the system
    returns a complex numpy array
    """
    return np.kron(np.identity(2**(number_of_qubits - qubit - 1), dtype=complex),
                   np.kron(gate, np.identity(2**(qubit), dtype=complex)))


def operator_qij(gate, qubit_1, qubit_2, number_of_qubits):
    """Apply the two-qubit gate.

    gate is the two-qubit gate
    qubit_1 is the first qubit (control) counts from 0
    qubit_2 is the second qubit (target)
    number_of_qubits is the number of qubits in the system
    returns a complex numpy array
    """
    temp_1 = np.kron(np.identity(
        2**(number_of_qubits - 2), dtype=complex), gate)
    temp_2 = np.identity(2**(number_of_qubits), dtype=complex)
    for ii in range(2**number_of_qubits):
        iistring = bin(ii)[2:]
        bits = list(iistring.zfill(number_of_qubits))
        swap_1 = bits[0]
        swap_2 = bits[1]
        bits[0] = bits[qubit_1]
        bits[1] = bits[qubit_2]
        bits[qubit_1] = swap_1
        bits[qubit_2] = swap_2
        iistring = ''.join(bits)
        iip = int(iistring, 2)
        for jj in range(2**number_of_qubits):
            jjstring = bin(jj)[2:]
            bits = list(jjstring.zfill(number_of_qubits))
            swap_1 = bits[0]
            swap_2 = bits[1]
            bits[0] = bits[qubit_1]
            bits[1] = bits[qubit_2]
            bits[qubit_1] = swap_1
            bits[qubit_2] = swap_2
            jjstring = ''.join(bits)
            jjp = int(jjstring, 2)
            temp_2[iip, jjp] = temp_1[ii, jj]
    return temp_2
