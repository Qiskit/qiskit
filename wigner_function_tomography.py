"""

Wigner Function Tomography Module

Description:
    This module contains functions to build circuits to measure points in
    phase space and to calculate the Wigner function at the measured points.
    
References:
    [1] T. Tilma, M. J. Everitt, J. H. Samson, W. J. Munro,
and K. Nemoto,
        Phys. Rev. Lett. 117, 180401 (2016).
    [2] R. P. Rundle, P. W. Mills, T. Tilma, J. H. Samson, and
M. J. Everitt,
        Phys. Rev. A 96, 022117 (2017).
"""

import math
import numpy as np
import qiskit.tools.qcvv.tomography as tomo

def build_wigner_circuits(Q_program, name, phis, thetas, qubits,
                          qreg, creg, silent=False):
    
    """
    Create the circuits to rotate to points in phase space

    Args:
        Q_program (QuantumProgram): A quantum program to store the circuits.
        name (string): The name of the base circuit to be appended.
        phis (np.matrix[[complex]]):
        thetas (np.matrix[[complex]]):
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        silent (bool, optional): hide verbose output.

    Returns: A list of names of the added wigner function circuits.
        
    """
    
    orig = Q_program.get_circuit(name)
    labels = []
    points = len(phis[0])

    
    for point in range(points):
        label = '_wigner_phase_point'
        label += str(point)
        circuit = Q_program.create_circuit(label, [qreg], [creg])
        c_index = 0

        for qubit in range(len(qubits)):            
            circuit.u3(thetas[qubit][point], 0,
                       phis[qubit][point],qreg[qubits[qubit]])
            circuit.measure(qreg[qubits[qubit]],creg[qubits[qubit]])

        Q_program.add_circuit(name+label, orig+circuit)
        labels.append(name+label)
        

    if not silent:
        print('>> created Wigner function circuits for "%s"' % name)
    return labels


def wigner_data(Q_result, name, meas_qubits, labels, shots=None):
    
    """
    Get the value of the Wigner function from measurement results.

    Args:
        Q_result (Result): Results from execution of a state tomography
            circuits on a backend.
        name (string): The name of the base state preparation circuit.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        labels : a list of names of the circuits

    Returns: The values of the Wigner function at measured points in
             phase space
        
    """
    
    num = len(meas_qubits)

    dim = 2**num            
    P = [0.5+0.5*math.sqrt(3),0.5-0.5*math.sqrt(3)]
    parity = 1
    
    for i in range(num):
        parity = np.kron(parity,P)
        
    W = [0]*len(labels)
    wpt = 0
    counts = [tomo.marginal_counts(Q_result.get_counts(circ), meas_qubits)
              for circ in labels]
    for entry in counts:
        x =[0]*dim
    
        for i in range(dim):
            if bin(i)[2:].zfill(num) in entry:
                x[i] = float(entry[bin(i)[2:].zfill(num)])
    
        if shots is None:
            shots = np.sum(x)
    
        for i in range(dim):
            W[wpt] = W[wpt]+(x[i]/shots)*parity[i]

        wpt += 1
        
    return W
