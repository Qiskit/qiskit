"""
Quantum tools for common operators.

These are simple methods for making common matrices used in quantum computing.

Author: Jay Gambetta
"""
import numpy as np


def destroy(dim):
    """Annihilation operator.

    dim integer dimension (for qubits dim = 2**n where n is number of qubits)
    returns a complex numpy array
    """
    a = np.zeros((dim, dim), dtype=complex)
    for jj in range(dim):
        for kk in range(dim):
            if kk-jj == 1:
                a[jj, kk] = np.sqrt(jj + 1)
    return a


def operator_qi(gate, qubit, number_of_qubits):
    """Apply the single qubit gate.

    gate is the single qubit gate
    qubit is the qubit to apply it on counts from 0 and order
        is q_{n-1} ... otimes q_1 otimes q_0
    number_of_qubits is the number of qubits in the system
    returns a complex numpy array
    """
    return np.kron(np.identity(2**(number_of_qubits-qubit-1), dtype=complex),
                   np.kron(gate, np.identity(2**(qubit), dtype=complex)))


def operator_qij(gate, qubit_1, qubit_2, number_of_qubits):
    """Apply the two-qubit gate.

    gate is the two-qubit gate
    qubit_1 is the first qubit (control) counts from 0
    qubit_2 is the second qubit (target)
    number_of_qubits is the number of qubits in the system
    returns a complex numpy array
    """
    temp_1 = np.kron(np.identity(2**(number_of_qubits-2), dtype=complex), gate)
    temp_2 = temp_1.reshape([2]*2*number_of_qubits)
    new_order = list(range(number_of_qubits))
    print(new_order)
    if qubit_1 == 0 and qubit_2 == 1:
        # do noething (working)
        new_order[0] = qubit_1
        new_order[1] = qubit_2
    elif qubit_1 == 0 and qubit_2 != 1:
        # keep 1 constant and swap 2 (working)
        new_order[1] = qubit_2
        new_order[qubit_2] = 1
    elif qubit_1 != 0 and qubit_2 == 1:
        # keep 2 constant and swap 1 (working)
        new_order[0] = qubit_1
        new_order[qubit_1] = 0
    elif qubit_1 == 1 and qubit_2 == 0:
        # swap qubit 1 and 2 (working)
        new_order[0] = qubit_1
        new_order[1] = qubit_2
    elif qubit_1 > 1 and qubit_2 > 1:
        # a simple a swap
        new_order[0] = qubit_1
        new_order[qubit_1] = 1
        new_order[1] = qubit_2
        new_order[qubit_2] = 2
    else:
        # if one is in the first two (this is the confusing one)
        new_order[0] = qubit_1
        new_order[1] = qubit_2
        if qubit_1 <= 1:
            new_order[qubit_2] = 1
        elif qubit_2 <= 1:
            new_order[qubit_1] = 1

    print(new_order)
    # temp_3 = np.transpose(temp_2, axes=new_order)
    print("lev")
    print(temp_1)
    print("lev")
    print(temp_2)
    print("lev")
    print(temp_3)

    # for ii in range(2**number_of_qubits):
    #     iip = 0
    #     oldqubit = 1
    #     for kk in new_order:
    #         iip += 2**(number_of_qubits-kk)*(ii/2**(number_of_qubits
    #                                          - oldqubit) % 2)
    #         oldqubit += 1
    #     # print str (iip) +' ' + str (ii)
    #     for jj in range(2**number_of_qubits):
    #         jjp = 0
    #         oldqubitj = 1
    #         for kk in new_order:
    #             jjp += 2**(number_of_qubits-kk)*(jj/2**(number_of_qubits
    #                                              - oldqubitj) % 2)
    #             oldqubitj += 1
    #         temp_2[iip, jjp] = temp_1[ii, jj]
    # return temp_2
