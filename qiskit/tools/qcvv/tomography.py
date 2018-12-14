# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Quantum Tomography Module

Description:
    This module contains functions for performing quantum state and quantum
    process tomography. This includes:
    - Functions for generating a set of circuits to
      extract tomographically complete sets of measurement data.
    - Functions for generating a tomography data set from the
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

Workflow:
    The basic functions for performing state and tomography experiments are:
    - `tomography_set`, `state_tomography_set`, and `process_tomography_set`
       all generates data structures for tomography experiments.
    - `create_tomography_circuits` generates the quantum circuits specified
       in a `tomography_set` for performing state tomography of the output
    - `tomography_data` extracts the results after executing the tomography
       circuits and returns it in a data structure used by fitters for state
       reconstruction.
    - `fit_tomography_data` reconstructs a density matrix or Choi-matrix from
       the a set of tomography data.
"""

import logging
from functools import reduce
from itertools import product
from re import match

import numpy as np

from qiskit import QuantumCircuit
from qiskit import QiskitError
from qiskit.tools.qi.qi import vectorize, devectorize, outer

logger = logging.getLogger(__name__)

###############################################################################
# Tomography Bases
###############################################################################


class TomographyBasis(dict):
    """
    Dictionary subsclass that includes methods for adding gates to circuits.

    A TomographyBasis is a dictionary where the keys index a measurement
    and the values are a list of projectors associated to that measurement.
    It also includes two optional methods `prep_gate` and `meas_gate`:
        - `prep_gate` adds gates to a circuit to prepare the corresponding
          basis projector from an initial ground state.
        - `meas_gate` adds gates to a circuit to transform the default
          Z-measurement into a measurement in the basis.
    With the exception of built in bases, these functions do nothing unless
    they are specified by the user. They may be set by the data members
    `prep_fun` and `meas_fun`. We illustrate this with an example.

    Example:
        A measurement in the Pauli-X basis has two outcomes corresponding to
        the projectors:
            `Xp = [[0.5, 0.5], [0.5, 0.5]]`
            `Xm = [[0.5, -0.5], [-0.5, 0.5]]`
        We can express this as a basis by
            `BX = TomographyBasis( {'X': [Xp, Xm]} )`
        To specifiy the gates to prepare and measure in this basis we :
            ```
            def BX_prep_fun(circuit, qreg, op):
                bas, proj = op
                if bas == "X":
                    if proj == 0:
                        circuit.u2(0., np.pi, qreg)  # apply H
                    else:  # proj == 1
                        circuit.u2(np.pi, np.pi, qreg)  # apply H.X
            def BX_prep_fun(circuit, qreg, op):
                if op == "X":
                        circuit.u2(0., np.pi, qreg)  # apply H
            ```
        We can then attach these functions to the basis using:
            `BX.prep_fun = BX_prep_fun`
            `BX.meas_fun = BX_meas_fun`.

    Generating function:
        A generating function `tomography_basis` exists to create bases in a
        single step. Using the above example this can be done by:
        ```
        BX = tomography_basis({'X': [Xp, Xm]},
                              prep_fun=BX_prep_fun,
                              meas_fun=BX_meas_fun)
        ```
    """

    prep_fun = None
    meas_fun = None

    def prep_gate(self, circuit, qreg, op):
        """
        Add state preparation gates to a circuit.

        Args:
            circuit (QuantumCircuit): circuit to add a preparation to.
            qreg (tuple(QuantumRegister,int)): quantum register to apply
            preparation to.
            op (tuple(str, int)): the basis label and index for the
            preparation op.
        """
        if self.prep_fun is None:
            pass
        else:
            self.prep_fun(circuit, qreg, op)

    def meas_gate(self, circuit, qreg, op):
        """
        Add measurement gates to a circuit.

        Args:
            circuit (QuantumCircuit): circuit to add measurement to.
            qreg (tuple(QuantumRegister,int)): quantum register being measured.
            op (str): the basis label for the measurement.
        """
        if self.meas_fun is None:
            pass
        else:
            self.meas_fun(circuit, qreg, op)


def tomography_basis(basis, prep_fun=None, meas_fun=None):
    """
    Generate a TomographyBasis object.

    See TomographyBasis for further details.abs

    Args:
        prep_fun (callable) optional: the function which adds preparation
            gates to a circuit.
        meas_fun (callable) optional: the function which adds measurement
            gates to a circuit.

    Returns:
        TomographyBasis: A tomography basis.
    """
    ret = TomographyBasis(basis)
    ret.prep_fun = prep_fun
    ret.meas_fun = meas_fun
    return ret


# PAULI BASIS
# This corresponds to measurements in the X, Y, Z basis where
# Outcomes 0,1 are the +1,-1 eigenstates respectively.
# State preparation is also done in the +1 and -1 eigenstates.


def __pauli_prep_gates(circuit, qreg, op):
    """
    Add state preparation gates to a circuit.
    """
    bas, proj = op
    if bas not in ['X', 'Y', 'Z']:
        raise QiskitError("There's no X, Y or Z basis for this Pauli "
                          "preparation")

    if bas == "X":
        if proj == 1:
            circuit.u2(np.pi, np.pi, qreg)  # H.X
        else:
            circuit.u2(0., np.pi, qreg)  # H
    elif bas == "Y":
        if proj == 1:
            circuit.u2(-0.5 * np.pi, np.pi, qreg)  # S.H.X
        else:
            circuit.u2(0.5 * np.pi, np.pi, qreg)  # S.H
    elif bas == "Z" and proj == 1:
        circuit.u3(np.pi, 0., np.pi, qreg)  # X


def __pauli_meas_gates(circuit, qreg, op):
    """
    Add state measurement gates to a circuit.
    """
    if op not in ['X', 'Y', 'Z']:
        raise QiskitError("There's no X, Y or Z basis for this Pauli "
                          "measurement")

    if op == "X":
        circuit.u2(0., np.pi, qreg)  # H
    elif op == "Y":
        circuit.u2(0., 0.5 * np.pi, qreg)  # H.S^*


__PAULI_BASIS_OPS = {
    'X':
    [np.array([[0.5, 0.5], [0.5, 0.5]]),
     np.array([[0.5, -0.5], [-0.5, 0.5]])],
    'Y': [
        np.array([[0.5, -0.5j], [0.5j, 0.5]]),
        np.array([[0.5, 0.5j], [-0.5j, 0.5]])
    ],
    'Z': [np.array([[1, 0], [0, 0]]),
          np.array([[0, 0], [0, 1]])]
}

# Create the actual basis
PAULI_BASIS = tomography_basis(
    __PAULI_BASIS_OPS,
    prep_fun=__pauli_prep_gates,
    meas_fun=__pauli_meas_gates)


# SIC-POVM BASIS
def __sic_prep_gates(circuit, qreg, op):
    """
    Add state preparation gates to a circuit.
    """
    bas, proj = op

    if bas != 'S':
        raise QiskitError('Not in SIC basis!')

    theta = -2 * np.arctan(np.sqrt(2))
    if proj == 1:
        circuit.u3(theta, np.pi, 0.0, qreg)
    elif proj == 2:
        circuit.u3(theta, np.pi / 3, 0.0, qreg)
    elif proj == 3:
        circuit.u3(theta, -np.pi / 3, 0.0, qreg)


__SIC_BASIS_OPS = {
    'S': [
        np.array([[1, 0], [0, 0]]),
        np.array([[1, np.sqrt(2)], [np.sqrt(2), 2]]) / 3,
        np.array([[1, np.exp(np.pi * 2j / 3) * np.sqrt(2)],
                  [np.exp(-np.pi * 2j / 3) * np.sqrt(2), 2]]) / 3,
        np.array([[1, np.exp(-np.pi * 2j / 3) * np.sqrt(2)],
                  [np.exp(np.pi * 2j / 3) * np.sqrt(2), 2]]) / 3
    ]
}

SIC_BASIS = tomography_basis(__SIC_BASIS_OPS, prep_fun=__sic_prep_gates)

###############################################################################
# Tomography Set and labels
###############################################################################


def tomography_set(meas_qubits,
                   meas_basis='Pauli',
                   prep_qubits=None,
                   prep_basis=None):
    """
    Generate a dictionary of tomography experiment configurations.

    This returns a data structure that is used by other tomography functions
    to generate state and process tomography circuits, and extract tomography
    data from results after execution on a backend.

    Quantum State Tomography:
        Be default it will return a set for performing Quantum State
        Tomography where individual qubits are measured in the Pauli basis.
        A custom measurement basis may also be used by defining a user
        `tomography_basis` and passing this in for the `meas_basis` argument.

    Quantum Process Tomography:
        A quantum process tomography set is created by specifying a preparation
        basis along with a measurement basis. The preparation basis may be a
        user defined `tomography_basis`, or one of the two built in basis 'SIC'
        or 'Pauli'.
        - SIC: Is a minimal symmetric informationally complete preparation
               basis for 4 states for each qubit (4 ^ number of qubits total
               preparation states). These correspond to the |0> state and the 3
               other vertices of a tetrahedron on the Bloch-sphere.
        - Pauli: Is a tomographically overcomplete preparation basis of the six
                 eigenstates of the 3 Pauli operators (6 ^ number of qubits
                 total preparation states).

    Args:
        meas_qubits (list): The qubits being measured.
        meas_basis (tomography_basis or str): The qubit measurement basis.
            The default value is 'Pauli'.
        prep_qubits (list or None): The qubits being prepared. If None then
            meas_qubits will be used for process tomography experiments.
        prep_basis (tomography_basis or None): The optional qubit preparation
            basis. If no basis is specified state tomography will be performed
            instead of process tomography. A built in basis may be specified by
            'SIC' or 'Pauli'  (SIC basis recommended for > 2 qubits).

    Returns:
        dict: A dict of tomography configurations that can be parsed by
        `create_tomography_circuits` and `tomography_data` functions
        for implementing quantum tomography experiments. This output contains
        fields "qubits", "meas_basis", "circuits". It may also optionally
        contain a field "prep_basis" for process tomography experiments.
        ```
        {
            'qubits': qubits (list[ints]),
            'meas_basis': meas_basis (tomography_basis),
            'circuit_labels': (list[string]),
            'circuits': (list[dict])  # prep and meas configurations
            # optionally for process tomography experiments:
            'prep_basis': prep_basis (tomography_basis)
        }
        ```
    Raises:
        QiskitError: if the Qubits argument is not a list.
    """
    if not isinstance(meas_qubits, list):
        raise QiskitError('Qubits argument must be a list')
    num_of_qubits = len(meas_qubits)

    if prep_qubits is None:
        prep_qubits = meas_qubits
    if not isinstance(prep_qubits, list):
        raise QiskitError('prep_qubits argument must be a list')
    if len(prep_qubits) != len(meas_qubits):
        raise QiskitError('meas_qubits and prep_qubitsare different length')

    if isinstance(meas_basis, str):
        if meas_basis.lower() == 'pauli':
            meas_basis = PAULI_BASIS

    if isinstance(prep_basis, str):
        if prep_basis.lower() == 'pauli':
            prep_basis = PAULI_BASIS
        elif prep_basis.lower() == 'sic':
            prep_basis = SIC_BASIS

    circuits = []
    circuit_labels = []

    # add meas basis configs
    if prep_basis is None:
        # State Tomography
        for meas_product in product(meas_basis.keys(), repeat=num_of_qubits):
            meas = dict(zip(meas_qubits, meas_product))
            circuits.append({'meas': meas})
            # Make label
            label = '_meas_'
            for qubit, op in meas.items():
                label += '%s(%d)' % (op[0], qubit)
            circuit_labels.append(label)
        return {'qubits': meas_qubits,
                'circuits': circuits,
                'circuit_labels': circuit_labels,
                'meas_basis': meas_basis}

    # Process Tomography
    num_of_s = len(list(prep_basis.values())[0])
    plst_single = [(b, s)
                   for b in prep_basis.keys()
                   for s in range(num_of_s)]
    for plst_product in product(plst_single, repeat=num_of_qubits):
        for meas_product in product(meas_basis.keys(),
                                    repeat=num_of_qubits):
            prep = dict(zip(prep_qubits, plst_product))
            meas = dict(zip(meas_qubits, meas_product))
            circuits.append({'prep': prep, 'meas': meas})
            # Make label
            label = '_prep_'
            for qubit, op in prep.items():
                label += '%s%d(%d)' % (op[0], op[1], qubit)
            label += '_meas_'
            for qubit, op in meas.items():
                label += '%s(%d)' % (op[0], qubit)
            circuit_labels.append(label)
    return {'qubits': meas_qubits,
            'circuits': circuits,
            'circuit_labels': circuit_labels,
            'prep_basis': prep_basis,
            'meas_basis': meas_basis}


def state_tomography_set(qubits, meas_basis='Pauli'):
    """
    Generate a dictionary of state tomography experiment configurations.

    This returns a data structure that is used by other tomography functions
    to generate state and process tomography circuits, and extract tomography
    data from results after execution on a backend.

    Quantum State Tomography:
        Be default it will return a set for performing Quantum State
        Tomography where individual qubits are measured in the Pauli basis.
        A custom measurement basis may also be used by defining a user
        `tomography_basis` and passing this in for the `meas_basis` argument.

    Quantum Process Tomography:
        A quantum process tomography set is created by specifying a preparation
        basis along with a measurement basis. The preparation basis may be a
        user defined `tomography_basis`, or one of the two built in basis 'SIC'
        or 'Pauli'.
        - SIC: Is a minimal symmetric informationally complete preparation
               basis for 4 states for each qubit (4 ^ number of qubits total
               preparation states). These correspond to the |0> state and the 3
               other vertices of a tetrahedron on the Bloch-sphere.
        - Pauli: Is a tomographically overcomplete preparation basis of the six
                 eigenstates of the 3 Pauli operators (6 ^ number of qubits
                 total preparation states).

    Args:
        qubits (list): The qubits being measured.
        meas_basis (tomography_basis or str): The qubit measurement basis.
            The default value is 'Pauli'.

    Returns:
        dict: A dict of tomography configurations that can be parsed by
        `create_tomography_circuits` and `tomography_data` functions
        for implementing quantum tomography experiments. This output contains
        fields "qubits", "meas_basis", "circuits".
        ```
        {
            'qubits': qubits (list[ints]),
            'meas_basis': meas_basis (tomography_basis),
            'circuit_labels': (list[string]),
            'circuits': (list[dict])  # prep and meas configurations
        }
        ```
    """
    return tomography_set(qubits, meas_basis=meas_basis)


def process_tomography_set(meas_qubits, meas_basis='Pauli',
                           prep_qubits=None, prep_basis='SIC'):
    """
    Generate a dictionary of process tomography experiment configurations.

    This returns a data structure that is used by other tomography functions
    to generate state and process tomography circuits, and extract tomography
    data from results after execution on a backend.

   A quantum process tomography set is created by specifying a preparation
    basis along with a measurement basis. The preparation basis may be a
    user defined `tomography_basis`, or one of the two built in basis 'SIC'
    or 'Pauli'.
    - SIC: Is a minimal symmetric informationally complete preparation
           basis for 4 states for each qubit (4 ^ number of qubits total
           preparation states). These correspond to the |0> state and the 3
           other vertices of a tetrahedron on the Bloch-sphere.
    - Pauli: Is a tomographically overcomplete preparation basis of the six
             eigenstates of the 3 Pauli operators (6 ^ number of qubits
             total preparation states).

    Args:
        meas_qubits (list): The qubits being measured.
        meas_basis (tomography_basis or str): The qubit measurement basis.
            The default value is 'Pauli'.
        prep_qubits (list or None): The qubits being prepared. If None then
            meas_qubits will be used for process tomography experiments.
        prep_basis (tomography_basis or str): The qubit preparation basis.
            The default value is 'SIC'.

    Returns:
        dict: A dict of tomography configurations that can be parsed by
        `create_tomography_circuits` and `tomography_data` functions
        for implementing quantum tomography experiments. This output contains
        fields "qubits", "meas_basis", "prep_basus", circuits".
        ```
        {
            'qubits': qubits (list[ints]),
            'meas_basis': meas_basis (tomography_basis),
            'prep_basis': prep_basis (tomography_basis),
            'circuit_labels': (list[string]),
            'circuits': (list[dict])  # prep and meas configurations
        }
        ```
    """
    return tomography_set(meas_qubits, meas_basis=meas_basis,
                          prep_qubits=prep_qubits, prep_basis=prep_basis)


def tomography_circuit_names(tomo_set, name=''):
    """
    Return a list of tomography circuit names.

    The returned list is the same as the one returned by
    `create_tomography_circuits` and can be used by a QuantumProgram
    to execute tomography circuits and extract measurement results.

    Args:
        tomo_set (tomography_set): a tomography set generated by
            `tomography_set`.
        name (str): the name of the base QuantumCircuit used by the
        tomography experiment.

    Returns:
        list: A list of circuit names.
    """
    return [name + l for l in tomo_set['circuit_labels']]


###############################################################################
# Tomography circuit generation
###############################################################################


def create_tomography_circuits(circuit, qreg, creg, tomoset):
    """
    Add tomography measurement circuits to a QuantumProgram.

    The quantum program must contain a circuit 'name', which is treated as a
    state preparation circuit for state tomography, or as teh circuit being
    measured for process tomography. This function then appends the circuit
    with a set of measurements specified by the input `tomography_set`,
    optionally it also prepends the circuit with state preparation circuits if
    they are specified in the `tomography_set`.

    For n-qubit tomography with a tomographically complete set of preparations
    and measurements this results in $4^n 3^n$ circuits being added to the
    quantum program.

    Args:
        circuit (QuantumCircuit): The circuit to be appended with tomography
                                  state preparation and/or measurements.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        tomoset (tomography_set): the dict of tomography configurations.

    Returns:
        list: A list of quantum tomography circuits for the input circuit.

    Raises:
        QiskitError: if circuit is not a valid QuantumCircuit

    Example:
        For a tomography set specifying state tomography of qubit-0 prepared
        by a circuit 'circ' this would return:
        ```
        ['circ_meas_X(0)', 'circ_meas_Y(0)', 'circ_meas_Z(0)']
        ```
        For process tomography of the same circuit with preparation in the
        SIC-POVM basis it would return:
        ```
        [
            'circ_prep_S0(0)_meas_X(0)', 'circ_prep_S0(0)_meas_Y(0)',
            'circ_prep_S0(0)_meas_Z(0)', 'circ_prep_S1(0)_meas_X(0)',
            'circ_prep_S1(0)_meas_Y(0)', 'circ_prep_S1(0)_meas_Z(0)',
            'circ_prep_S2(0)_meas_X(0)', 'circ_prep_S2(0)_meas_Y(0)',
            'circ_prep_S2(0)_meas_Z(0)', 'circ_prep_S3(0)_meas_X(0)',
            'circ_prep_S3(0)_meas_Y(0)', 'circ_prep_S3(0)_meas_Z(0)'
        ]
        ```
    """

    if not isinstance(circuit, QuantumCircuit):
        raise QiskitError('Input circuit must be a QuantumCircuit object')

    dics = tomoset['circuits']
    labels = tomography_circuit_names(tomoset, circuit.name)
    tomography_circuits = []
    for label, conf in zip(labels, dics):
        tmp = circuit
        # Add prep circuits
        if 'prep' in conf:
            prep = QuantumCircuit(qreg, creg, name='tmp_prep')
            for qubit, op in conf['prep'].items():
                tomoset['prep_basis'].prep_gate(prep, qreg[qubit], op)
                prep.barrier(qreg[qubit])  # pylint: disable=no-member
            tmp = prep + tmp
        # Add measurement circuits
        meas = QuantumCircuit(qreg, creg, name='tmp_meas')
        for qubit, op in conf['meas'].items():
            meas.barrier(qreg[qubit])  # pylint: disable=no-member
            tomoset['meas_basis'].meas_gate(meas, qreg[qubit], op)
            meas.measure(qreg[qubit], creg[qubit])
        tmp = tmp + meas
        # Add label to the circuit
        tmp.name = label
        tomography_circuits.append(tmp)

    logger.info('>> created tomography circuits for "%s"', circuit.name)
    return tomography_circuits


###############################################################################
# Get results data
###############################################################################


def tomography_data(results, name, tomoset):
    """
    Return a results dict for a state or process tomography experiment.

    Args:
        results (Result): Results from execution of a process tomography
            circuits on a backend.
        name (string): The name of the circuit being reconstructed.
        tomoset (tomography_set): the dict of tomography configurations.

    Returns:
        list: A list of dicts for the outcome of each process tomography
        measurement circuit.
    """

    labels = tomography_circuit_names(tomoset, name)
    circuits = tomoset['circuits']
    data = []
    prep = None
    for j, _ in enumerate(labels):
        counts = marginal_counts(results.get_counts(labels[j]),
                                 tomoset['qubits'])
        shots = sum(counts.values())
        meas = circuits[j]['meas']
        prep = circuits[j].get('prep', None)
        meas_qubits = sorted(meas.keys())
        if prep:
            prep_qubits = sorted(prep.keys())
        circuit = {}
        for c in counts.keys():
            circuit[c] = {}
            circuit[c]['meas'] = [(meas[meas_qubits[k]], int(c[-1 - k]))
                                  for k in range(len(meas_qubits))]
            if prep:
                circuit[c]['prep'] = [prep[prep_qubits[k]]
                                      for k in range(len(prep_qubits))]
        data.append({'counts': counts, 'shots': shots, 'circuit': circuit})

    ret = {'data': data, 'meas_basis': tomoset['meas_basis']}
    if prep:
        ret['prep_basis'] = tomoset['prep_basis']
    return ret


def marginal_counts(counts, meas_qubits):
    """
    Compute the marginal counts for a subset of measured qubits.

    Args:
        counts (dict): the counts returned from a backend ({str: int}).
        meas_qubits (list[int]): the qubits to return the marginal
                                 counts distribution for.

    Returns:
        dict: A counts dict for the meas_qubits.abs
        Example: if `counts = {'00': 10, '01': 5}`
            `marginal_counts(counts, [0])` returns `{'0': 15, '1': 0}`.
            `marginal_counts(counts, [0])` returns `{'0': 10, '1': 5}`.
    """
    # pylint: disable=cell-var-from-loop
    # Extract total number of qubits from count keys
    num_of_qubits = len(list(counts.keys())[0])

    # keys for measured qubits only
    qs = sorted(meas_qubits, reverse=True)

    meas_keys = count_keys(len(qs))

    # get regex match strings for summing outcomes of other qubits
    rgx = [
        reduce(lambda x, y: (key[qs.index(y)] if y in qs else '\\d') + x,
               range(num_of_qubits), '') for key in meas_keys
    ]

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


def count_keys(n):
    """Generate outcome bitstrings for n-qubits.

    Args:
        n (int): the number of qubits.

    Returns:
        list: A list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
    """
    return [bin(j)[2:].zfill(n) for j in range(2**n)]


###############################################################################
# Tomographic Reconstruction functions.
###############################################################################


def fit_tomography_data(tomo_data, method='wizard', options=None):
    """
    Reconstruct a density matrix or process-matrix from tomography data.

    If the input data is state_tomography_data the returned operator will
    be a density matrix. If the input data is process_tomography_data the
    returned operator will be a Choi-matrix in the column-vectorization
    convention.

    Args:
        tomo_data (dict): process tomography measurement data.
        method (str): the fitting method to use.
            Available methods:
                - 'wizard' (default)
                - 'leastsq'
        options (dict or None): additional options for fitting method.

    Returns:
        numpy.array: The fitted operator.

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
        - 'leastsq': Fitting without positive-semidefinite constraint.
            Options:
            - 'trace': Same as for 'wizard' method.
            - 'beta': Same as for 'wizard' method.
    Raises:
        Exception: if the `method` parameter is not valid.
    """

    if isinstance(method, str) and method.lower() in ['wizard', 'leastsq']:
        # get options
        trace = __get_option('trace', options)
        beta = __get_option('beta', options)
        # fit state
        rho = __leastsq_fit(tomo_data, trace=trace, beta=beta)
        if method == 'wizard':
            # Use wizard method to constrain positivity
            epsilon = __get_option('epsilon', options)
            rho = __wizard(rho, epsilon=epsilon)
        return rho
    else:
        raise Exception('Invalid reconstruction method "%s"' % method)


def __get_option(opt, options):
    """
    Return an optional value or None if not found.
    """
    if options is not None:
        if opt in options:
            return options[opt]
    return None


###############################################################################
# Fit Method: Linear Inversion
###############################################################################


def __leastsq_fit(tomo_data, weights=None, trace=None, beta=None):
    """
    Reconstruct a state from unconstrained least-squares fitting.

    Args:
        tomo_data (list[dict]): state or process tomography data.
        weights (list or array or None): weights to use for least squares
            fitting. The default is standard deviation from a binomial
            distribution.
        trace (float or None): trace of returned operator. The default is 1.
        beta (float or None): hedge parameter (>=0) for computing frequencies
            from zero-count data. The default value is 0.50922.

    Returns:
        numpy.array: A numpy array of the reconstructed operator.
    """
    if trace is None:
        trace = 1.  # default to unit trace

    data = tomo_data['data']
    keys = data[0]['circuit'].keys()

    # Get counts and shots
    counts = []
    shots = []
    ops = []
    for dat in data:
        for key in keys:
            counts.append(dat['counts'][key])
            shots.append(dat['shots'])
            projectors = dat['circuit'][key]
            op = __projector(projectors['meas'], tomo_data['meas_basis'])
            if 'prep' in projectors:
                op_prep = __projector(projectors['prep'],
                                      tomo_data['prep_basis'])
                op = np.kron(op_prep.conj(), op)
            ops.append(op)

    # Convert counts to frequencies
    counts = np.array(counts)
    shots = np.array(shots)
    freqs = counts / shots

    # Use hedged frequencies to calculate least squares fitting weights
    if weights is None:
        if beta is None:
            beta = 0.50922
        K = len(keys)
        freqs_hedged = (counts + beta) / (shots + K * beta)
        weights = np.sqrt(shots / (freqs_hedged * (1 - freqs_hedged)))

    return __tomo_linear_inv(freqs, ops, weights, trace=trace)


def __projector(op_list, basis):
    """Returns a projectors.
    """
    ret = 1
    # list is from qubit 0 to 1
    for op in op_list:
        label, eigenstate = op
        ret = np.kron(basis[label][eigenstate], ret)
    return ret


def __tomo_linear_inv(freqs, ops, weights=None, trace=None):
    """
    Reconstruct a matrix through linear inversion.

    Args:
        freqs (list[float]): list of observed frequences.
        ops (list[np.array]): list of corresponding projectors.
        weights (list[float] or array_like):
            weights to be used for weighted fitting.
        trace (float or None): trace of returned operator.

    Returns:
        numpy.array: A numpy array of the reconstructed operator.
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


###############################################################################
# Fit Method: Wizard
###############################################################################


def __wizard(rho, epsilon=None):
    """
    Returns the nearest positive semidefinite operator to an operator.

    This method is based on reference [1]. It constrains positivity
    by setting negative eigenvalues to zero and rescaling the positive
    eigenvalues.

    Args:
        rho (array_like): the input operator.
        epsilon(float or None): threshold (>=0) for truncating small
            eigenvalues values to zero.

    Returns:
        numpy.array: A positive semidefinite numpy array.
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
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))
    for j in range(dim):
        rho_wizard = rho_wizard + v[j] * outer(w[:, j])
    return rho_wizard


###############################################################
# Wigner function tomography
###############################################################

def build_wigner_circuits(circuit, phis, thetas, qubits,
                          qreg, creg):
    """Create the circuits to rotate to points in phase space
    Args:
        circuit (QuantumCircuit): The circuit to be appended with tomography
                                  state preparation and/or measurements.
        phis (np.matrix[[complex]]): phis
        thetas (np.matrix[[complex]]): thetas
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                    store measurement outcomes.

    Returns:
        list: A list of names of the added wigner function circuits.

    Raises:
        QiskitError: if circuit is not a valid QuantumCircuit.
    """

    if not isinstance(circuit, QuantumCircuit):
        raise QiskitError('Input circuit must be a QuantumCircuit object')

    tomography_circuits = []
    points = len(phis[0])
    for point in range(points):
        label = '_wigner_phase_point'
        label += str(point)
        tmp_circ = QuantumCircuit(qreg, creg, name=label)
        for qubit, _ in enumerate(qubits):
            tmp_circ.u3(thetas[qubit][point], 0,  # pylint: disable=no-member
                        phis[qubit][point], qreg[qubits[qubit]])
            tmp_circ.measure(qreg[qubits[qubit]], creg[qubits[qubit]])
        # Add to original circuit
        tmp_circ = circuit + tmp_circ
        tmp_circ.name = circuit.name + label
        tomography_circuits.append(tmp_circ)

    logger.info('>> Created Wigner function circuits for "%s"', circuit.name)
    return tomography_circuits


def wigner_data(q_result, meas_qubits, labels, shots=None):
    """Get the value of the Wigner function from measurement results.

    Args:
        q_result (Result): Results from execution of a state tomography
                            circuits on a backend.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        labels (list[str]): a list of names of the circuits
        shots (int): number of shots

    Returns:
        list: The values of the Wigner function at measured points in
            phase space
    """
    num = len(meas_qubits)

    dim = 2**num
    p = [0.5 + 0.5 * np.sqrt(3), 0.5 - 0.5 * np.sqrt(3)]
    parity = 1

    for i in range(num):
        parity = np.kron(parity, p)

    w = [0] * len(labels)
    wpt = 0
    counts = [marginal_counts(q_result.get_counts(circ), meas_qubits)
              for circ in labels]
    for entry in counts:
        x = [0] * dim

        for i in range(dim):
            if bin(i)[2:].zfill(num) in entry:
                x[i] = float(entry[bin(i)[2:].zfill(num)])

        if shots is None:
            shots = np.sum(x)

        for i in range(dim):
            w[wpt] = w[wpt] + (x[i] / shots) * parity[i]
        wpt += 1

    return w
