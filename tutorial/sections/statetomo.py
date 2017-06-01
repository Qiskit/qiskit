# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# This file is intended only for use during the USEQIP Summer School 2017.
# Do not distribute.
# It is provided without warranty or conditions of any kind, either express or
# implied.
# An open source version of this file will be included in QISKIT-DEV-PY
# reposity in the future. Keep an eye on the Github repository for updates!
# https://github.com/IBM/qiskit-sdk-py
# =============================================================================

"""
Quantum state tomography using the maximum likelihood reconstruction method
from Smolin, Gambetta, Smith Phys. Rev. Lett. 108, 070502  (arXiv: 1106.5458)

Author: Christopher J. Wood <cjwood@us.ibm.com>
"""
import sys
if sys.version_info < (3,0):
    raise Exception("Please use Python version 3 or greater.")

import numpy as np
sys.path.append("../../")
from qiskit import QuantumProgram
import Qconfig

# Note this requires the notebook to load qiskit quantum program and Qconfig

###############################################################
# Tomographic Reconstruction functions.
###############################################################

def vectorize(op):
    """
    Flatten an operator to a column-major vector.
    """
    return op.flatten(order='F')

def devectorize(v):
    """
    Devectorize a column-major vectorized square matrix.
    """
    d = int(np.sqrt(v.size));
    return v.reshape(d,d, order='F')

def meas_basis_matrix(meas_basis):
    """
    Returns a matrix of vectorized measurement operators S = sum_j |j><M_j|.
    """
    n = len(meas_basis)
    d = meas_basis[0].size
    S = np.array([vectorize(m).conj() for m in meas_basis])
    return S.reshape(n,d)

def outer(v1, v2=None):
    """
    Returns the matrix |v1><v2| resulting from the outer product of two vectors.
    """
    if v2 is None:
        u = v1.conj()
    else:
        u = v2.conj()
    return np.outer(v1, u)

def wizard(rho, normalize_flag = True, epsilon = 0.):
    """
    Maps an operator to the nearest positive semidefinite operator
    by setting negative eigenvalues to zero and rescaling the positive
    eigenvalues.
    See arXiv:1106.5458 [quant-ph]
    """
    #print("Using wizard method to constrain positivity")
    if normalize_flag:
        rho = rho / np.trace(rho)
    dim = len(rho)
    rho_wizard = np.zeros([dim, dim])
    v, w = np.linalg.eigh(rho) # v eigenvecrors v[0] < v[1] <...
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.
            # redistribute loop
            x = 0.
            for k in range(j+1,dim):
                x += tmp / (dim-(j+1))
                v[k] = v[k] + tmp / (dim -(j+1))
    for j in range(dim):
        rho_wizard = rho_wizard + v[j] * outer(w[:,j])
    return rho_wizard

def fit_state(freqs, meas_basis, weights=None, normalize_flag = True, wizard_flag = False):
    """
    Returns a matrix reconstructed by unconstrained least-squares fitting.
    """
    if weights is None:
        W = np.eye(len(freqs)) # use uniform weights
    else:
        W = np.array(np.diag(weights))
    S = np.dot(W, meas_basis_matrix(meas_basis)) # actually W.S
    v = np.dot(W, freqs) # W|f>
    v = np.array(np.dot(S.T.conj(), v)) # S^*.W^*W.|f>
    inv = np.linalg.pinv(np.dot(S.T.conj(), S)) # pseudo inverse of  S^*.W^*.W.S
    v = np.dot(inv, v) # |rho>
    rho = devectorize(v)
    if normalize_flag:
        rho = rho / np.trace(rho)
    if wizard_flag:
        #rho_wizard = wizard(rho,normalize_flag)
        rho = wizard(rho, normalize_flag=normalize_flag)
    return rho

###############################################################
# Constructing tomographic measurement circuits
###############################################################
def build_keys_helper(keys, qubit):
    """
    Returns array of measurement strings ['Xj', 'Yj', 'Zj'] for qubit=j.
    """
    tmp = []
    for k in keys:
        for b in ["X","Y","Z"]:
            tmp.append(k + b + str(qubit))
    return tmp

def build_tomo_keys(circuit, qubit_list):
    """
    For input circuit string returns an array of all measurement circits orded in
    lexicographic order from last qubit to first qubit.
    Example:
    qubit_list = [0]: [circuitX0, circuitY0, circuitZ0].
    qubit_list = [0,1]: [circuitX1X0, circuitX1Y0, circuitX1Z0, circuitY1X0,..., circuitZ1Z0].
    """
    keys = [circuit]
    for j in sorted(qubit_list,reverse=True):
        keys = build_keys_helper(keys, j)
    return keys

# We need to build circuits in QASM lexical order, not standard!

def build_tomo_circuit_helper(Q_program, circuits, qreg: str, creg: str, qubit: int):
    """
    Adds measurements for the qubit=j to the input circuits, so if circuits = [c0, c1,...]
    circuits-> [c0Xj, c0Yj, c0Zj, c1Xj,...]
    """
    for c in circuits:
        circ = Q_program.get_circuit(c)
        for b in ["X","Y","Z"]:
            meas = b+str(qubit)
            tmp = Q_program.create_circuit(meas, [qreg],[creg])
            qr = Q_program.get_quantum_registers(qreg)
            cr = Q_program.get_classical_registers(creg)
            if b == "X":
                tmp.u2(0., np.pi, qr[qubit])
            if b == "Y":
                tmp.u2(0., np.pi / 2., qr[qubit])
            tmp.measure(qr[qubit], cr[qubit])
            Q_program.add_circuit(c+meas, circ + tmp)

def build_tomo_circuits(Q_program, circuit, qreg: str, creg: str, qubit_list):
    """
    Generates circuits in the input QuantumProgram for implementing complete quantum
    state tomography of the state of the qubits in qubit_list prepared by circuit.
    """
    circ = [circuit]
    for j in sorted(qubit_list, reverse=True):
        build_tomo_circuit_helper(Q_program, circ, qreg, creg, j)
        circ = build_keys_helper(circ, j)

###############################################################
# Parsing measurement data for reconstruction
###############################################################

def nqubit_basis(n):
    """
    Returns the measurement basis for n-qubits in the correct order for the
    meas_outcome_strings function.
    """
    b1 =  np.array([
            np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]]), # Xp, Xm
            np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]]), # Yp, Ym
            np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), # Zp, Zm
            ])
    if n == 1:
        return b1
    else:
        bnm1 = nqubit_basis(n-1)
        m = 2**(n-1)
        d = bnm1.size // m**3
        return np.kron( bnm1.reshape(d, m, m, m), b1.reshape(3, 2, 2, 2)).reshape(3*d * 2*m, 2*m, 2*m)

def meas_outcome_strings(nq):
    """
    Returns a list of the measurement outcome strings for nq qubits.
    """
    return [bin(j)[2:].zfill(nq) for j in range(2**nq)]

def tomo_outcome_strings(meas_qubits,nq=None):
    """
    Returns a list of the measurement outcome strings for meas_qubits qubits in an
    nq qubit system.
    """
    if nq is None:
        nq = len(meas_qubits)
    qs = sorted(meas_qubits, reverse=True)
    outs = meas_outcome_strings(len(qs))
    res = [];
    for s in outs:
        label = ""
        for j in range(nq):
            if j in qs:
                label = s[qs.index(j)] + label
            else:
                label = str(0) + label
        res.append(label)
    return res

def none_to_zero(val):
    """
    Returns 0 if the input argument is None, else it returns the input.
    """
    if val is None:
        return 0
    else:
        return val

###############################################################
# Putting it all together
###############################################################
def state_tomography(Q_program, tomo_circuits, shots, nq,
                                    meas_qubits, method='leastsq_wizard'):
    """
    Returns the reconstructed density matrix.
    """
    m = len(meas_qubits)
    counts = np.array([none_to_zero(Q_program.get_counts(c).get(s))
                        for c in tomo_circuits
                        for s in tomo_outcome_strings(meas_qubits, nq)])
    if method == 'leastsq_wizard':
        return fit_state(counts / shots, nqubit_basis(m), normalize_flag=True, wizard_flag=True)
    elif method == 'least_sq':
        return fit_state(counts / shots, nqubit_basis(m), normalize_flag=True, wizard_flag=False)
    else:
        print("error: unknown reconstruction method")

def fidelity(rho, psi):
    """
    Returns the state fidelity F between a density matrix rho
    and a target pure state psi.
    """
    return np.sqrt(np.abs(np.dot(psi, np.dot(rho, psi))))
