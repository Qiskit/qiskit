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
Quantum Tomography Module

Description:
    This module contains functions for performing quantum state and quantum
    process tomography. This includes:
    - Functions for generating a set of circuits in a QuantumProgram to
      extract tomographically complete sets of measurement data.
    - Functions for generating a tomography data set from the QuantumProgram
      results after the circuits have been executed on a backend.
    - Functions for reconstructing a quantum state, or quantum process
      (Choi-matrix) from tomography data sets.

Reconstruction Methods:
    Currently implemented reconstruction methods are
    - Linear inversion by weighted least-squares fitting.
    - Fast maximum likelihood reconstruction using ref [1].

References:
    [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
        Open access: arXiv:1106.5458 [quant-ph].
"""

import numpy as np
from functools import reduce
from re import match
from itertools import product

from tools.qi.qi import vectorize, devectorize, outer


###############################################################
# Tomography circuit generation
###############################################################

def build_state_tomography_circuits(Q_program, name, qubits, qreg, creg,
                                    silent=False):
    """
    Add state tomography measurement circuits to a QuantumProgram.

    The quantum program must contain a circuit 'name', which is treated as a
    state preparation circuit. This function then appends the circuit with a
    tomographically overcomplete set of measurements in the Pauli basis for
    each qubit to be measured. For n-qubit tomography this result in 3 ** n
    measurement circuits being added to the quantum program.

    Args:
        Q_program (QuantumProgram): A quantum program to store the circuits.
        name (string): The name of the base circuit to be appended.
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        silent (bool, optional): hide verbose output.

    Returns:
        A list of names of the added quantum state tomography circuits.
        Example: ['circ_measX0', 'circ_measY0', 'circ_measZ0']
    """

    labels = __add_meas_circuits(Q_program, name, qubits, qreg, creg)
    if not silent:
        print('>> created state tomography circuits for "%s"' % name)
    return labels


def build_process_tomography_circuits(Q_program, name, qubits, qreg, creg,
                                      silent=False):
    """
    Add process tomography measurement circuits to a QuantumProgram.

    The quantum program must contain a circuit 'name', which is the circuit
    that will be reconstructed via tomographic measurements. This function
    then prepends and appends the circuit with a tomographically overcomplete
    set of preparations and measurements in the Pauli basis for
    each qubit to be measured. For n-qubit process tomography this result in
    (6 ** n) * (3 ** n) circuits being added to the quantum program:
        - 3 ** n measurements in the Pauli X, Y, Z bases.
        - 6 ** n preparations in the +1 and -1 eigenstates of X, Y, Z.

    Args:
        Q_program (QuantumProgram): A quantum program to store the circuits.
        name (string): The name of the base circuit to be appended.
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        silent (bool, optional): hide verbose output.

    Returns:
        A list of names of the added quantum process tomography circuits.
        Example:
        ['circ_prepXp0_measX0', 'circ_prepXp0_measY0', 'circ_prepXp0_measZ0',
         'circ_prepXm0_measX0', 'circ_prepXm0_measY0', 'circ_prepXm0_measZ0',
         'circ_prepYp0_measX0', 'circ_prepYp0_measY0', 'circ_prepYp0_measZ0',
         'circ_prepYm0_measX0', 'circ_prepYm0_measY0', 'circ_prepYm0_measZ0',
         'circ_prepZp0_measX0', 'circ_prepZp0_measY0', 'circ_prepZp0_measZ0',
         'circ_prepZm0_measX0', 'circ_prepZm0_measY0', 'circ_prepZm0_measZ0']
    """

    # add preparation circuits
    preps = __add_prep_circuits(Q_program, name, qubits, qreg, creg)
    # add measurement circuits for each prep circuit
    labels = []
    for circ in preps:
        labels += __add_meas_circuits(Q_program, circ, qubits, qreg, creg)
        # delete temp prep output
        del Q_program._QuantumProgram__quantum_program[circ]
    if not silent:
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
        A new list of tomo_dict
    """

    if isinstance(qubits, int):
        qubits = [qubits]

    if basis is None:
        basis = __DEFAULT_BASIS

    if states:
        ns = len(list(basis.values())[0])
        lst = [(b, s) for b in basis.keys() for s in range(ns)]
    else:
        lst = basis.keys()

    return [dict(zip(qubits, b)) for b in product(lst, repeat=len(qubits))]


def __add_meas_circuits(Q_program, name, qubits, qreg, creg):
    """
    Add measurement circuits to a quantum program.

    See: build_state_tomography_circuits.
         build_process_tomography_circuits.
    """

    orig = Q_program.get_circuit(name)

    labels = []

    for dic in __tomo_dicts(qubits):

        # Construct meas circuit name
        label = '_meas'
        for qubit, op in dic.items():
            label += op + str(qubit)
        circuit = Q_program.create_circuit(label, [qreg], [creg])

        # add gates to circuit
        for qubit, op in dic.items():
            circuit.barrier(qreg[qubit])
            if op == "X":
                circuit.u2(0., np.pi, qreg[qubit])  # H
            elif op == "Y":
                circuit.u2(0., 0.5 * np.pi, qreg[qubit])  # H.S^*
            circuit.measure(qreg[qubit], creg[qubit])
        # add circuit to QuantumProgram
        Q_program.add_circuit(name+label, orig + circuit)
        # add label to output
        labels.append(name+label)
        # delete temp circuit
        del Q_program._QuantumProgram__quantum_program[label]

    return labels


def __add_prep_gates(circuit, qreg, qubit, op):
    """
    Add state preparation gates to a circuit.
    """
    p, s = op
    if p == "X":
        if s == 1:
            circuit.u2(np.pi, np.pi, qreg[qubit])  # H.X
        else:
            circuit.u2(0., np.pi, qreg[qubit])  # H
    if p == "Y":
        if s == 1:
            circuit.u2(-0.5 * np.pi, np.pi, qreg[qubit])  # S.H.X
        else:
            circuit.u2(0.5 * np.pi, np.pi, qreg[qubit])  # S.H
    if p == "Z" and s == 1:
        circuit.u3(np.pi, 0., np.pi, qreg[qubit])  # X


def __add_prep_circuits(Q_program, name, qubits, qreg, creg):
    """
    Add preparation circuits to a quantum program.

    See: build_process_tomography_circuits.
    """

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
        if states:
            for qubit, op in dic.items():
                label += op[0] + state[op[1]] + str(qubit)
        else:
            for qubit, op in dic.items():
                label += op[0] + str(qubit)
        labels.append(name+label)
    return labels


def state_tomography_circuit_names(name, qubits):
    """
    Return a list of state tomography circuit names.

    This list is the same as that returned by the
    build_state_tomography_circuits function.

    Args:
        name (string): the name of the original state preparation
                       circuit.
        qubits: (list[int]): the qubits being measured.

    Returns:
        A list of circuit names.
    """
    return __tomo_labels(name + '_meas', qubits)


def process_tomography_circuit_names(name, qubits):
    """
    Return a list of process tomography circuit names.

    This list is the same as that returned by the
    build_process_tomography_circuits function.

    Args:
        name (string): the name of the original circuit to be
                       reconstructed.
        qubits: (list[int]): the qubits being measured.

    Returns:
        A list of circuit names.
    """
    preps = __tomo_labels(name + '_prep', qubits, states=True)
    return reduce(lambda acc, c:
                  acc + __tomo_labels(c + '_meas', qubits),
                  preps, [])


###############################################################
# Preformatting count data
###############################################################

def __counts_keys(n):
    """Generate outcome bitstrings for n-qubits.

    Args:
        n (int): the number of qubits.

    Returns:
        A list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
    """
    return [bin(j)[2:].zfill(n) for j in range(2 ** n)]


def marginal_counts(counts, meas_qubits):
    """
    Compute the marginal counts for a subset of measured qubits.

    Args:
        counts (dict{str:int}): the counts returned from a backend.
        meas_qubits (list[int]): the qubits to return the marginal
                                 counts distribution for.

    Returns:
        A counts dict for the meas_qubits.abs
        Example: if counts = {'00': 10, '01': 5}
            marginal_counts(counts, [0]) returns {'0': 15, '1': 0}.
            marginal_counts(counts, [0]) returns {'0': 10, '1': 5}.
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


###############################################################
# Tomography preparation and measurement bases
###############################################################

# Default Pauli basis
# This corresponds to measurements in the X, Y, Z basis where
# Outcomes 0,1 are the +1,-1 eigenstates respectively.
# State preparation is also done in the +1 and -1 eigenstates.

__DEFAULT_BASIS = {'X': [np.array([[0.5, 0.5],
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


def __get_meas_basis_ops(tup, basis):
    """
    Return a n-qubit projector for a given measurement.
    """
    # reverse tuple so least significant qubit is to the right
    return reduce(lambda acc, b: [np.kron(a, j)
                                  for a in acc for j in basis[b]],
                  reversed(tup), [1])


def __meas_basis(n, basis):
    """
    Return an ordered list of n-qubit measurment projectors.
    """
    return [dict(zip(__counts_keys(n), __get_meas_basis_ops(key, basis)))
            for key in product(basis.keys(), repeat=n)]


def __get_prep_basis_op(dic, basis):
    """
    Return an n-qubit projector for a given prepration.
    """
    keys = sorted(dic.keys())  # order qubits [0,1,...]
    tups = [dic[k] for k in keys]
    return reduce(lambda acc, b: np.kron(basis[b[0]][b[1]], acc),
                  tups, [1])


def __prep_basis(n, basis):
    """
    Return an ordered list of n-qubit preparation projectors.
    """
    # use same function as prep circuits to get order right
    ordered = __tomo_dicts(range(n), states=True)
    return [__get_prep_basis_op(dic, basis) for dic in ordered]


def state_tomography_data(Q_result, name, meas_qubits, basis=None):
    """
    Return a list of state tomography measurement outcomes.

    Args:
        Q_result (Result): Results from execution of a state tomography
            circuits on a backend.
        name (string): The name of the base state preparation circuit.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        basis (basis dict, optional): the basis used for measurement. Default
            is the Pauli basis.

    Returns:
        A list of dicts for the outcome of each state tomography
        measurement circuit. The keys of the dictionary are
        {
            'counts': dict('str': int),
                      <the marginal counts for measured qubits>,
            'shots': int,
                     <total number of shots for measurement circuit>
            'meas_basis': dict('str': np.array)
                          <the projector for the measurement outcomes>
        }
    """
    if basis is None:
        basis = __DEFAULT_BASIS
    labels = state_tomography_circuit_names(name, meas_qubits)
    counts = [marginal_counts(Q_result.get_counts(circ), meas_qubits)
              for circ in labels]
    shots = [sum(c.values()) for c in counts]
    meas_basis = __meas_basis(len(meas_qubits), basis)
    ret = [{'counts': i, 'meas_basis': j, 'shots': k}
           for i, j, k in zip(counts, meas_basis, shots)]
    return ret


def process_tomography_data(Q_result, name, meas_qubits, basis=None):
    """
    Return a list of process tomography measurement outcomes.

    Args:
        Q_result (Result): Results from execution of a process tomography
            circuits on a backend.
        name (string): The name of the circuit being reconstructed.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        basis (basis dict, optional): the basis used for measurement. Default
            is the Pauli basis.

    Returns:
        A list of dicts for the outcome of each process tomography
        measurement circuit. The keys of the dictionary are
        {
            'counts': dict('str': int),
                      <the marginal counts for measured qubits>,
            'shots': int,
                     <total number of shots for measurement circuit>
            'meas_basis': dict('str': np.array),
                          <the projector for the measurement outcomes>
            'prep_basis': np.array,
                          <the projector for the prepared input state>
        }
    """
    if basis is None:
        basis = __DEFAULT_BASIS
    n = len(meas_qubits)
    labels = process_tomography_circuit_names(name, meas_qubits)
    counts = [marginal_counts(Q_result.get_counts(circ), meas_qubits)
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

    Args:
        meas_basis(list(array_like)): measurement operators [M_j].
    Returns:
        The operators S = sum_j |j><M_j|.
    """
    n = len(meas_basis)
    d = meas_basis[0].size
    S = np.array([vectorize(m).conj() for m in meas_basis])
    return S.reshape(n, d)


def __tomo_linear_inv(freqs, ops, weights=None, trace=None):
    """
    Reconstruct a matrix through linear inversion.

    Args:
        freqs (list[float]): list of observed frequences.
        ops (list[np.array]): list of corresponding projectors.
        weights (list[float] or array_like, optional):
            weights to be used for weighted fitting.
        trace (float, optional): trace of returned operator.

    Returns:
        A numpy array of the reconstructed operator.
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


def __leastsq_fit(data, weights=None, trace=None, beta=None):
    """
    Reconstruct a state from unconstrained least-squares fitting.

    Args:
        data (list[dict]): state or process tomography data.
        weights (list or array, optional): weights to use for least squares
            fitting. The default is standard deviation from a binomial
            distribution.
        trace (float, optional): trace of returned operator. The default is 1.
        beta (float >=0, optional): hedge parameter for computing frequencies
            from zero-count data. The default value is 0.50922.

    Returns:
        A numpy array of the reconstructed operator.
    """
    if trace is None:
        trace = 1.  # default to unit trace

    ks = data[0]['counts'].keys()
    K = len(ks)
    # Get counts and shots
    ns = np.array([dat['counts'][k] for dat in data for k in ks])
    shots = np.array([dat['shots'] for dat in data for k in ks])
    # convert to freqs using beta to hedge against zero counts
    if beta is None:
        beta = 0.50922
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


def __wizard(rho, epsilon=None):
    """
    Returns the nearest postitive semidefinite operator to an operator.

    This method is based on reference [1]. It constrains positivity
    by setting negative eigenvalues to zero and rescaling the positive
    eigenvalues.

    Args:
        rho (array_like): the input operator.
        epsilon(float >=0, optional): threshold for truncating small
            eigenvalues values to zero.

    Returns:
        A positive semidefinite numpy array.
    """
    if epsilon is None:
        epsilon = 0.  # default value

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
    """
    Return an optional value or None if not found.
    """
    if options is not None:
        if opt in options:
            return options[opt]
    return None


def fit_tomography_data(data, method=None, options=None):
    """
    Reconstruct a density matrix or process-matrix from tomography data.

    If the input data is state_tomography_data the returned operator will
    be a density matrix. If the input data is process_tomography_data the
    returned operator will be a Choi-matrix in the column-vectorization
    convention.

    Args:
        data (dict): process tomography measurement data.
        method (str, optional): the fitting method to use.
            Available methods:
                - 'wizard' (default)
                - 'leastsq'
        options (dict, optional): additional options for fitting method.

    Returns:
        The fitted operator.

    Available methods:
        - 'wizard' (Default): The returned operator will be constrained to be
                              positive-semidefinite.
            Options:
            - 'trace': the trace of the returned operator.
                       The default value is 1.
            - 'beta': hedging parameter for computing frequencies from
                      zero-count data. The default value is 0.50922.
            - 'epsilon: threshold for truncating small eigenvalues to zero.
                        The default value is 0
        - 'leastsq': Fitting without postive-semidefinite constraint.
            Options:
            - 'trace': Same as for 'wizard' method.
            - 'beta': Same as for 'wizard' method.
    """
    if method is None:
        method = 'wizard'  # set default method

    if method in ['wizard', 'leastsq']:
        # get options
        trace = __get_option('trace', options)
        beta = __get_option('beta', options)
        # fit state
        rho = __leastsq_fit(data, trace=trace, beta=beta)
        if method == 'wizard':
            # Use wizard method to constrain positivity
            epsilon = __get_option('epsilon', options)
            rho = __wizard(rho, epsilon=epsilon)
        return rho
    else:
        print('error: method unknown reconstruction method "%s"' % method)
