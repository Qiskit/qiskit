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
        Jay Gambetta

TODO: Process tomography, SDP fitting

"""
import numpy as np
from functools import reduce
from re import match
from itertools import product

from qiskit import QuantumProgram
from tools.qi import vectorize, devectorize, outer


###############################################################
# NEW CIRCUIT GENERATION
###############################################################


"""
Basis should be specified as a dictionary {'X': [X0, X1], 'Y': [Y0, Y1], 'Z': [Z0, Z1]}
where X0 is the projector onto the 0 outcome state of 'X'
"""


def build_state_tomography_circuits(Q_program, name, qubits, qreg, creg):
    """
    """
    labels = __add_meas_circuits(Q_program, name, qubits, qreg, creg)
    print('>> created state tomography circuits for "%s"' % name)
    return labels
    

# Make private for now, since fit method not yet implemented
def __build_process_tomography_circuits(Q_program, name, qubits, qreg, creg):
    """
    """
    # add preparation circuits
    preps = __add_prep_circuits(Q_program, name, qubits, qreg, creg)
    # add measurement circuits for each prep circuit
    labels = []
    for circ in preps:
        labels += __add_meas_circuits(Q_program, circ, qubits, qreg, creg)
        # delete temp prep output
        del Q_program._QuantumProgram__quantum_program[circ]
    print('>> created process tomography circuits for "%s"' % name)
    return labels


def __tomo_dicts(qubits, basis=None):
    """Helper function.

    Build a dictionary assigning a basis element to a qubit.

    Args:
        qubit (int): the qubit to add
        tomos (list[dict]): list of tomo_dicts to add to
        basis (list[str], optional): basis to use. If not specified
            the default is ['X', 'Y', 'Z']

    Returns:
        a new list of tomo_dict
    """
    """
    if basis is None:
        basis = ['X', 'Y', 'Z']
    elif isinstance(basis, dict):
        basis = basis.keys()

    if not tomos:
        return [{qubit: b} for b in basis]
    else:
        ret = []
        for t in tomos:
            for b in basis:
                t[qubit] = b
                ret.append(t.copy())
        return ret
    """
    if isinstance(qubits, int):
        qubits = [qubits]
    if basis is None:
        basis = ['X', 'Y', 'Z']
    elif isinstance(basis, dict):
        basis = basis.keys()
    
    return [dict(zip(qubits, b)) for b in product(basis, repeat=len(qubits))]

def __add_meas_circuits(Q_program, name, qubits, qreg, creg, prep=None):
    """Helper function.
    """

    if isinstance(qreg, str):
        qreg = Q_program.get_quantum_registers(qreg)
    if isinstance(creg, str):
        creg = Q_program.get_classical_registers(creg)
    orig = Q_program.get_circuit(name)

    labels = []

    for dic in __tomo_dicts(qubits):

        # Construct meas circuit name
        label = '_meas'
        for qubit, op in dic.items():
            label += op + str(qubit)
        circ = Q_program.create_circuit(label, [qreg], [creg])

        # add gates to circuit
        for qubit, op in dic.items():
            circ.barrier(qreg[qubit])
            if op == "X":
                circ.h(qreg[qubit])
            elif op == "Y":
                circ.sdg(qreg[qubit])
                circ.h(qreg[qubit])
            if prep:
                circ.barrier(qreg[qubit])
            else:
                circ.measure(qreg[qubit], creg[qubit])
        # add circuit to QuantumProgram
        Q_program.add_circuit(name+label, orig + circ)
        # add label to output
        labels.append(name+label)
        # delete temp circuit
        del Q_program._QuantumProgram__quantum_program[label]

    return labels

def __add_prep_circuits(Q_program, name, qubits, qreg, creg):
    """Helper function.
    """

    if isinstance(qreg, str):
        qreg = Q_program.get_quantum_registers(qreg)
    if isinstance(creg, str):
        creg = Q_program.get_classical_registers(creg)
    orig = Q_program.get_circuit(name)

    labels = []
    for dic in __tomo_dicts(qubits):

        # Construct meas circuit name
        label_p = '_prep'   # prepare in positive eigenstate
        label_m = '_prep'   # prepare in negative eigenstate
        for qubit, op in dic.items():
            label_p +=  op + 'p' + str(qubit)
            label_m+=  op + 'm' + str(qubit)
        circ_p = Q_program.create_circuit(label_p, [qreg], [creg])
        circ_m = Q_program.create_circuit(label_m, [qreg], [creg])

        # add gates to circuit
        for qubit, op in dic.items():
            if op == "X":
                circ_p.h(qreg[qubit])
                circ_m.x(qreg[qubit])
                circ_m.h(qreg[qubit])
            elif op == "Y":
                circ_p.h(qreg[qubit])
                circ_p.s(qreg[qubit])
                circ_m.x(qreg[qubit])
                circ_m.h(qreg[qubit])
                circ_m.s(qreg[qubit])
            elif op == "Z":
                circ_m.x(qreg[qubit])
        # add circuit to QuantumProgram
        Q_program.add_circuit(name+label_p, circ_p + orig)
        Q_program.add_circuit(name+label_m, circ_m + orig)
        # add label to output
        labels += [name+label_p, name+label_m]
        # delete temp circuit
        del Q_program._QuantumProgram__quantum_program[label_p]
        del Q_program._QuantumProgram__quantum_program[label_m]

    return labels

###############################################################
# Tomography circuit labels
###############################################################

def __tomo_labels(name, qubits, basis=None, subscript=None):
    """Helper function.
    """
    if subscript is None:
        subscript = ''
    labels = []
    for dic in __tomo_dicts(qubits, basis):
        label = ''
        for qubit, op in dic.items():
            label += op + subscript + str(qubit)
        labels.append(name+label)
    return labels

def state_tomography_labels(name, qubits, basis=None):
    """
    """
    return __tomo_labels(name + '_meas', qubits, basis)
    

# Make private for now, since fit method not yet implemented
def __process_tomography_labels(name, qubits, basis=None):
    """
    """
    preps = __tomo_labels(name + '_prep', qubits, basis, subscript='p')
    preps += __tomo_labels(name + '_prep', qubits, basis, subscript='m')
    return reduce(lambda acc, c: acc + __tomo_labels(c + '_meas', qubits, basis), preps, [])

###############################################################
# Tomography measurement outcomes
###############################################################

# Note: So far only for state tomography

# Default Pauli basis
__default_basis = { 'X': [np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0.5, -0.5], [-0.5, 0.5]])],
                    'Y': [np.array([[0.5, -0.5j], [0.5j, 0.5]]), np.array([[0.5, 0.5j], [-0.5j, 0.5]])],
                    'Z': [np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]])]}


def __counts_keys(n):
    """helper function.
    """
    return [bin(j)[2:].zfill(n) for j in range(2 ** n)]


def __get_basis_ops(tup, basis):
    return reduce(lambda acc, b: [np.kron(a,j) for a in acc for j in basis[b]], tup, [1])


def __counts_basis(n, basis):
    return [dict(zip(__counts_keys(n), __get_basis_ops(key, basis))) 
            for key in product(basis.keys(), repeat=n)]


def marginal_counts(counts, meas_qubits):
    """
    Returns a list of the measurement outcome strings for meas_qubits qubits in an
    nq qubit system.
    """
    
    # Extract total number of qubits from count keys
    nq = len(list(counts.keys())[0])
    
    # keys for measured qubits only 
    qs = sorted(meas_qubits, reverse=True)

    meas_keys = __counts_keys(len(qs))

    # get regex match strings for suming outcomes of other qubits
    rgx = [reduce(lambda x, y: (key[qs.index(y)] if y in qs else '\d')+x, range(nq), '') for key in meas_keys]
    
    # build the return list
    meas_counts = []
    for m in rgx:
        c = 0
        for key, val in counts.items():
            if match(m, key):
                c += val
        meas_counts.append(c)
        
    # return as counts dict on measured qubits only
    return dict(zip(meas_keys, meas_counts))


def state_tomography_data(Q_program, name, meas_qubits, backend=None, basis=None):
    """
    """
    if basis is None:
        basis = __default_basis
    labels = state_tomography_labels(name, meas_qubits)
    counts = [marginal_counts(Q_program.get_counts(circ, backend), meas_qubits) for circ in labels]
    shots = [sum(c.values()) for c in counts]
    meas_basis = __counts_basis(len(meas_qubits), basis)
    ret = [{'counts': i, 'meas_basis': j, 'shots': k} for i, j, k in zip(counts, meas_basis, shots)]
    return ret


###############################################################
# Tomographic Reconstruction functions.
###############################################################


def __tomo_basis_matrix(meas_basis):
    """Return a matrix of vectorized measurement operators.

    Measurement operators S = sum_j |j><M_j|.
    """
    n = len(meas_basis)
    d = meas_basis[0].size
    S = np.array([vectorize(m).conj() for m in meas_basis])
    return S.reshape(n, d)


def __tomo_linear_inv(freqs, ops, weights=None, trace=1):
    """
    """
    # get weights matrix
    if weights is not None:
        W = np.array(weights)
        if W.ndim == 1:
            W = np.diag(W)
            
    # Get basis S matrix
    S = np.array([vectorize(m).conj() for m in ops]).reshape(len(ops), ops[0].size)
    if weights is not None:
        S = np.dot(W, S)  # W.S
        
    # get frequencies vec
    v = np.array(freqs)  # |n>
    if weights is not None:
        v = np.dot(W, freqs)  #W.|n>
    Sdg = S.T.conj()  # S^*.W^*
    inv = np.linalg.pinv(np.dot(Sdg,S))

    # linear inversion of freqs
    ret = devectorize(np.dot(inv, np.dot(Sdg, v)))
    # renormalize to input trace value
    ret = trace * ret / np.trace(ret)
    return ret


def __state_leastsq_fit(state_data, weights=None, beta=0.5):
    """
    """
    ks = state_data[0]['counts'].keys()
    
    # Get counts and shots
    ns = np.array([dat['counts'][k] for dat in state_data for k in ks])
    shots = np.array([dat['shots'] for dat in state_data for k in ks])
    
    # convert to freqs
    freqs = (ns + beta) / (shots + 2 * beta)
    
    # Use standard least squares fitting weights
    if weights is None:
        weights = np.sqrt(shots / (freqs * (1 - freqs)))
    
    # Get measurement basis ops
    ops = [dat['meas_basis'][k] for dat in state_data for k in ks]
    
    return __tomo_linear_inv(freqs, ops, weights, trace=1)


def __wizard(rho, epsilon=0.):
    """
    Maps an operator to the nearest positive semidefinite operator
    by setting negative eigenvalues to zero and rescaling the positive
    eigenvalues.
    See arXiv:1106.5458 [quant-ph]
    """
    dim = len(rho)
    rho_wizard = np.zeros([dim, dim])
    v, w = np.linalg.eigh(rho)  # v eigenvecrors v[0] < v[1] <...
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.
            # redistribute loop
            x = 0.
            for k in range(j + 1, dim):
                x += tmp / (dim-(j+1))
                v[k] = v[k] + tmp / (dim - (j+1))
    for j in range(dim):
        rho_wizard = rho_wizard + v[j] * outer(w[:, j])
    return rho_wizard


def fit_state(state_data, method='wizard', options=None):
    """
    """
    if method in ['wizard', 'leastsq']:
        rho = __state_leastsq_fit(state_data)
        if method == 'wizard':
            rho = __wizard(rho)
    else:
        # TODO: raise exception for unknown method
        print('error: method unrecongnized')
        pass
    
    return rho