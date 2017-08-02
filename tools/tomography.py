# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Functions for performing quantum state and process tomography experiments.

TODO:
    - Find and fix bug with 2-qubit process tomography basis
    - SDP fitting
"""
import numpy as np
from functools import reduce
from re import match
from itertools import product

from tools.qi import vectorize, devectorize, outer


###############################################################
# Tomography circuit generation
###############################################################


"""
Basis should be specified as a dictionary
Eg: {'X': [X0, X1], 'Y': [Y0, Y1], 'Z': [Z0, Z1]}
where X0 is the projector onto the 0 outcome state of 'X'
"""


def build_state_tomography_circuits(Q_program, name, qubits, qreg, creg):
    """
    """
    labels = __add_meas_circuits(Q_program, name, qubits, qreg, creg)
    print('>> created state tomography circuits for "%s"' % name)
    return labels


# Make private for now, since fit method not yet implemented
def build_process_tomography_circuits(Q_program, name, qubits, qreg, creg):
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


def __tomo_dicts(qubits, basis=None, states=False):
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
        basis = __default_basis

    if states is True:
        ns = len(list(basis.values())[0])
        lst = [(b, s) for b in basis.keys() for s in range(ns)]
    else:
        lst = basis.keys()

    return [dict(zip(qubits, b)) for b in product(lst, repeat=len(qubits))]


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


def __add_prep_gates(circuit, qreg, qubit, op):
    """helper function
    """
    p, s = op
    if p == "X":
        if s == 1:
            circuit.x(qreg[qubit])
        circuit.h(qreg[qubit])
    if p == "Y":
        if s == 1:
            circuit.x(qreg[qubit])
        circuit.h(qreg[qubit])
        circuit.s(qreg[qubit])
    if p == "Z" and s == 1:
        circuit.x(qreg[qubit])


def __add_prep_circuits(Q_program, name, qubits, qreg, creg):
    """Helper function.
    """

    if isinstance(qreg, str):
        qreg = Q_program.get_quantum_registers(qreg)
    if isinstance(creg, str):
        creg = Q_program.get_classical_registers(creg)
    orig = Q_program.get_circuit(name)

    labels = []
    state = {0: 'p', 1: 'm'}
    for dic in __tomo_dicts(qubits, states=True):

        # make circuit label
        label = '_prep'
        for qubit, op in dic.items():
            label += op[0] + state[op[1]] + str(qubit)

        # create circuit and add gates
        circuit = Q_program.create_circuit(label, [qreg], [creg])
        for qubit, op in dic.items():
            __add_prep_gates(circuit, qreg, qubit, op)

        # add circuit to QuantumProgram
        Q_program.add_circuit(name + label, circuit + orig)
        # add label to output
        labels += [name+label]
        # delete temp circuit
        del Q_program._QuantumProgram__quantum_program[label]

    return labels


###############################################################
# Tomography circuit labels
###############################################################


def __tomo_labels(name, qubits, basis=None, states=False):
    """Helper function.
    """
    labels = []
    state = {0: 'p', 1: 'm'}
    for dic in __tomo_dicts(qubits, states=states):
        label = ''
        if states is True:
            for qubit, op in dic.items():
                label += op[0] + state[op[1]] + str(qubit)
        else:
            for qubit, op in dic.items():
                label += op[0] + str(qubit)
        labels.append(name+label)
    return labels


def state_tomography_labels(name, qubits, basis=None):
    """
    """
    return __tomo_labels(name + '_meas', qubits, basis)


# Make private for now, since fit method not yet implemented
def process_tomography_labels(name, qubits, basis=None):
    """
    """
    preps = __tomo_labels(name + '_prep', qubits, basis, states=True)
    return reduce(lambda acc, c:
                  acc + __tomo_labels(c + '_meas', qubits, basis),
                  preps, [])


###############################################################
# Tomography measurement outcomes
###############################################################

# Note: So far only for state tomography


# Default Pauli basis
__default_basis = {'X': [np.array([[0.5, 0.5],
                                   [0.5, 0.5]]),
                         np.array([[0.5, -0.5],
                                   [-0.5, 0.5]])],
                   'Y': [np.array([[0.5, -0.5j],
                                   [0.5j, 0.5]]),
                         np.array([[0.5, 0.5j],
                                   [-0.5j, 0.5]])],
                   'Z': [np.array([[1, 0],
                                   [0, 0]]),
                         np.array([[0, 0],
                                   [0, 1]])]}


def __counts_keys(n):
    """helper function.
    """
    return [bin(j)[2:].zfill(n) for j in range(2 ** n)]


def __get_meas_basis_ops(tup, basis):
    # reverse tuple so least significant qubit is to the right
    return reduce(lambda acc, b: [np.kron(a, j)
                                  for a in acc for j in basis[b]],
                  reversed(tup), [1])


def __meas_basis(n, basis):
    return [dict(zip(__counts_keys(n), __get_meas_basis_ops(key, basis)))
            for key in product(basis.keys(), repeat=n)]


def __get_prep_basis_op(dic, basis):
    keys = sorted(dic.keys())  # order qubits [0,1,...]
    tups = [dic[k] for k in keys]
    return reduce(lambda acc, b: np.kron(basis[b[0]][b[1]], acc),
                  tups, [1])


def __prep_basis(n, basis):
    # use same function as prep circuits to get order right
    ordered = __tomo_dicts(range(n), states=True)
    return [__get_prep_basis_op(dic, basis) for dic in ordered]


def marginal_counts(counts, meas_qubits):
    """
    Returns a list of the measurement outcome strings for meas_qubits qubits
    in an n-qubit system.
    """

    # Extract total number of qubits from count keys
    nq = len(list(counts.keys())[0])

    # keys for measured qubits only
    qs = sorted(meas_qubits, reverse=True)

    meas_keys = __counts_keys(len(qs))

    # get regex match strings for suming outcomes of other qubits
    rgx = [reduce(lambda x, y: (key[qs.index(y)] if y in qs else '\\d') + x,
                  range(nq), '')
           for key in meas_keys]

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


def state_tomography_data(data, name, meas_qubits, basis=None):
    """
    """
    if basis is None:
        basis = __default_basis
    labels = state_tomography_labels(name, meas_qubits)
    counts = [marginal_counts(data.get_counts(circ), meas_qubits)
              for circ in labels]
    shots = [sum(c.values()) for c in counts]
    meas_basis = __meas_basis(len(meas_qubits), basis)
    ret = [{'counts': i, 'meas_basis': j, 'shots': k}
           for i, j, k in zip(counts, meas_basis, shots)]
    return ret


def process_tomography_data(data, name, meas_qubits, basis=None):
    """
    """
    if basis is None:
        basis = __default_basis
    n = len(meas_qubits)
    labels = process_tomography_labels(name, meas_qubits)
    counts = [marginal_counts(data.get_counts(circ), meas_qubits)
              for circ in labels]
    shots = [sum(c.values()) for c in counts]
    meas_basis = __meas_basis(n, basis)
    prep_basis = __prep_basis(n, basis)

    ret = [{'meas_basis': meas, 'prep_basis': prep}
           for prep in prep_basis for meas in meas_basis]

    for dic, cts, sts in zip(ret, counts, shots):
        dic['counts'] = cts
        dic['shots'] = sts
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


def __tomo_linear_inv(freqs, ops, weights=None, trace=None):
    """
    """
    # get weights matrix
    if weights is not None:
        W = np.array(weights)
        if W.ndim == 1:
            W = np.diag(W)

    # Get basis S matrix
    S = np.array([vectorize(m).conj()
                  for m in ops]).reshape(len(ops), ops[0].size)
    if weights is not None:
        S = np.dot(W, S)  # W.S

    # get frequencies vec
    v = np.array(freqs)  # |f>
    if weights is not None:
        v = np.dot(W, freqs)  # W.|f>
    Sdg = S.T.conj()  # S^*.W^*
    inv = np.linalg.pinv(np.dot(Sdg, S))  # (S^*.W^*.W.S)^-1

    # linear inversion of freqs
    ret = devectorize(np.dot(inv, np.dot(Sdg, v)))
    # renormalize to input trace value
    if trace is not None:
        ret = trace * ret / np.trace(ret)
    return ret


def __leastsq_fit(data, weights=None, trace=None, beta=0.5):
    """
    """
    ks = data[0]['counts'].keys()
    K = len(ks)
    # Get counts and shots
    ns = np.array([dat['counts'][k] for dat in data for k in ks])
    shots = np.array([dat['shots'] for dat in data for k in ks])

    # convert to freqs using beta to hedge against zero counts
    freqs = (ns + beta) / (shots + K * beta)

    # Use standard least squares fitting weights
    if weights is None:
        weights = np.sqrt(shots / (freqs * (1 - freqs)))

    # Get measurement basis ops
    if 'prep_basis' in data[0]:
        # process tomography fit
        ops = [np.kron(dat['prep_basis'].T, dat['meas_basis'][k])
               for dat in data for k in ks]
    else:
        # state tomography fit
        ops = [dat['meas_basis'][k] for dat in data for k in ks]

    return __tomo_linear_inv(freqs, ops, weights, trace=trace)


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


def __get_option(opt, options):
    if options is not None:
        if opt in options:
            return options[opt]
    return None


def fit_tomography_data(data, method='wizard', options=None):
    """
    Reconstruct a state or Choi-matrix from tomography data.

    Args:
        data (dict): process tomography measurement data.
        method (str, optional): the fitting method to use.
            Available methods:
                - 'wizard' (default)
                - 'leastsq'
        options (dict, optional): additional options for fitting method.

    Returns:
        The fitted operator.
    """

    if method in ['wizard', 'leastsq']:
        # get options
        trace = __get_option('trace', options)
        if trace is None:
            trace = 1
        beta = __get_option('beta', options)
        if beta is None:
            beta = 0.50922
        # fit state
        rho = __leastsq_fit(data, trace=trace, beta=beta)
        if method == 'wizard':
            # Use wizard method to constrain positivity
            epsilon = __get_option('epsilon', options)
            if epsilon is None:
                epsilon = 0.
            rho = __wizard(rho, epsilon=epsilon)
    else:
        # TODO: raise exception for unknown method
        print('error: method unrecongnized')
        pass

    return rho
