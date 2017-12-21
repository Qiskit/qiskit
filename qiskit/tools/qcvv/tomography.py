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

Workflow:
    The basic functions for performing state and tomography experiments are:
    - `tomography_set`, `state_tomography_set`, and `process_tomography_set` 
       all generates data structures for tomography experiments.
    - `create_tomography_circuits` generates the quantum circuits specified
       in a `tomography_set` and adds them to a `QuantumProgram` for perform
       state tomography of the output of a state preparation circuit, or
       process tomography of a circuit.
    - `tomography_data` extracts the results after executing the tomography
       circuits and returns it in a data structure used by fitters for state
       reconstruction.
    - `fit_tomography_data` reconstructs a density matrix or Choi-matrix from
       the a set of tomography data.
"""

import numpy as np
import itertools as it
import random
from functools import reduce
from re import match

from qiskit.tools.qi.qi import vectorize, devectorize, outer

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
          basis projector from an inital ground state.
        - `meas_gate` adds gates to a circuit to transform the default
          Z-measurement into a measurement in the basis.
    With the exception of built in bases, these functions do nothing unless
    they are specified by the user. They may be set by the data members
    `prep_fun` and `meas_fun`. We illustrate this with an example.

    Example:
        A measurement in the Pauli-X basis has two outcomes corresponding to
        the projectors:
            `Xp = [[0.5, 0.5], [0.5, 0.5]]`
            `Zm = [[0.5, -0.5], [-0.5, 0.5]]`
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
        basis (dict): the dictionary of basis labels and corresponding
        operators for projectors.
        prep_fun (function) optional: the function which adds preparation
        gates to a circuit.
        meas_fun (function) optional: the function which adds measurement
        gates to a circuit.

    Returns:
        A tomography basis which is 
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
    assert (bas in ['X', 'Y', 'Z'])
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
    assert (op in ['X', 'Y', 'Z'])
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
    assert (bas == 'S')
    if bas == "S":
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


def tomography_set(qubits,
                   meas_basis='Pauli',
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
        A quantum process tomography set is created by specifying a prepration
        basis along with a measurement basis. The preparation basis may be a
        user defined `tomography_basis`, or one of the two built in basis 'SIC'
        or 'Pauli'.
        - SIC: Is a minimal symmetric informationally complete preparaiton
               basis for 4 states for each qubit (4 ^ number of qubits total
               preparation states). These correspond to the |0> state and the 3
               other verteces of a tetrahedron on the Bloch-sphere.
        - Pauli: Is a tomographically overcomplete preparation basis of the six
                 eigenstates of the 3 Pauli operaters (6 ^ number of qubits
                 total preparation states).

    Args:
        qubits (list): The qubits being measured.
        meas_basis (tomography_basis) optional: The qubit measurement basis.
            The default value is 'Pauli'.
        prep_basis (tomography_basis) optional: The optional qubit preparation
            basis. If no basis is specified state tomography will be performed
            instead of process tomography. A built in basis may be specified by
            'SIC' or 'Pauli'  (SIC basis recommended for > 2 qubits).

    Returns:
        A dict of tomography configurations that can be parsed by
        `create_tomography_circuits` and `tomography_data` functions
        for implementing quantum tomography experiments. This output contains
        fields "qubits", "meas_basis", "circuits". It may also optionally
        contain a field "prep_basis" for process tomography experiments.
        ```
        {
            'qubits': qubits (list[ints]),
            'meas_basis': meas_basis (tomography_basis),
            'circuits': (list[dict])  # prep and meas configurations
            # optionally for process tomography experiments:
            'prep_basis': prep_basis (tomography_basis)
        }
        ```
    """

    assert (isinstance(qubits, list))
    nq = len(qubits)

    if meas_basis == 'Pauli':
        meas_basis = PAULI_BASIS

    if prep_basis == 'Pauli':
        prep_basis = PAULI_BASIS
    elif prep_basis == 'SIC':
        prep_basis = SIC_BASIS

    ret = {'qubits': qubits, 'meas_basis': meas_basis}

    # add meas basis configs
    mlst = meas_basis.keys()
    meas = [dict(zip(qubits, b)) for b in it.product(mlst, repeat=nq)]
    ret['circuits'] = [{'meas': m} for m in meas]

    if prep_basis is not None:
        ret['prep_basis'] = prep_basis
        ns = len(list(prep_basis.values())[0])
        plst = [(b, s) for b in prep_basis.keys() for s in range(ns)]
        ret['circuits'] = [{
            'prep': dict(zip(qubits, b)),
            'meas': dic['meas']
        } for b in it.product(plst, repeat=nq) for dic in ret['circuits']]

    return ret


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
        A quantum process tomography set is created by specifying a prepration
        basis along with a measurement basis. The preparation basis may be a
        user defined `tomography_basis`, or one of the two built in basis 'SIC'
        or 'Pauli'.
        - SIC: Is a minimal symmetric informationally complete preparaiton
               basis for 4 states for each qubit (4 ^ number of qubits total
               preparation states). These correspond to the |0> state and the 3
               other verteces of a tetrahedron on the Bloch-sphere.
        - Pauli: Is a tomographically overcomplete preparation basis of the six
                 eigenstates of the 3 Pauli operaters (6 ^ number of qubits
                 total preparation states).

    Args:
        qubits (list): The qubits being measured.
        meas_basis (tomography_basis) optional: The qubit measurement basis.
            The default value is 'Pauli'.
        prep_basis (tomography_basis) optional: The optional qubit preparation
            basis. If no basis is specified state tomography will be performed
            instead of process tomography. A built in basis may be specified by
            'SIC' or 'Pauli'  (SIC basis recommended for > 2 qubits).

    Returns:
        A dict of tomography configurations that can be parsed by
        `create_tomography_circuits` and `tomography_data` functions
        for implementing quantum tomography experiments. This output contains
        fields "qubits", "meas_basis", "circuits".
        ```
        {
            'qubits': qubits (list[ints]),
            'meas_basis': meas_basis (tomography_basis),
            'circuits': (list[dict])  # prep and meas configurations
        }
        ```

    Example:
        State tomography of a single qubit in the Pauli basis:
        ```
        state_tset = state_tomography_set([0, 1], meas_basis='Pauli')
        state_tset = {
            'qubits': [0, 1],
            'circuits': [
                {'meas': {0: 'X', 1: 'X'}},
                {'meas': {0: 'X', 1: 'Y'}},
                {'meas': {0: 'X', 1: 'Z'}},
                {'meas': {0: 'Y', 1: 'X'}},
                {'meas': {0: 'Y', 1: 'Y'}},
                {'meas': {0: 'Y', 1: 'Z'}},
                {'meas': {0: 'Z', 1: 'X'}},
                {'meas': {0: 'Z', 1: 'Y'}},
                {'meas': {0: 'Z', 1: 'Z'}}
                ],
            'meas_basis': {
                'X': [array([[ 0.5, 0.5], [ 0.5, 0.5]]),
                    array([[ 0.5, -0.5], [-0.5, 0.5]])],
                'Y': [array([[ 0.5+0.j , -0.0-0.5j], [ 0.0+0.5j,  0.5+0.j ]]),
                    array([[ 0.5+0.j ,  0.0+0.5j], [-0.0-0.5j,  0.5+0.j ]])],
                'Z': [array([[1, 0], [0, 0]]),
                    array([[0, 0], [0, 1]])]
                }
            }
        ```
    """
    return tomography_set(qubits, meas_basis=meas_basis)


def process_tomography_set(qubits, meas_basis='Pauli', prep_basis='SIC'):
    """
    Generate a dictionary of process tomography experiment configurations.

    This returns a data structure that is used by other tomography functions
    to generate state and process tomography circuits, and extract tomography
    data from results after execution on a backend.

   A quantum process tomography set is created by specifying a prepration
    basis along with a measurement basis. The preparation basis may be a
    user defined `tomography_basis`, or one of the two built in basis 'SIC'
    or 'Pauli'.
    - SIC: Is a minimal symmetric informationally complete preparaiton
           basis for 4 states for each qubit (4 ^ number of qubits total
           preparation states). These correspond to the |0> state and the 3
           other verteces of a tetrahedron on the Bloch-sphere.
    - Pauli: Is a tomographically overcomplete preparation basis of the six
             eigenstates of the 3 Pauli operaters (6 ^ number of qubits
             total preparation states).

    Args:
        qubits (list): The qubits being measured.
        meas_basis (tomography_basis) optional: The qubit measurement basis.
            The default value is 'Pauli'.
        prep_basis (tomography_basis) optional: The qubit preparation basis.
            The default value is 'SIC'.

    Returns:
        A dict of tomography configurations that can be parsed by
        `create_tomography_circuits` and `tomography_data` functions
        for implementing quantum tomography experiments. This output contains
        fields "qubits", "meas_basis", "prep_basus", circuits".
        ```
        {
            'qubits': qubits (list[ints]),
            'meas_basis': meas_basis (tomography_basis),
            'prep_basis': prep_basis (tomography_basis)
            'circuits': (list[dict])  # prep and meas configurations
        }
        ```

    Example:
        Process tomography in preparation in the SIC-POVM basis and
        measurement in the Pauli basis:
        ```
        proc_tset = tomography_set([0], meas_basis='Pauli', prep_basis='SIC')
        proc_tset = {
            'qubits': [0]
            'circuits': [
                {'meas': {0: 'X'}, 'prep': {0: ('S', 0)}},
                {'meas': {0: 'Y'}, 'prep': {0: ('S', 0)}},
                {'meas': {0: 'Z'}, 'prep': {0: ('S', 0)}},
                {'meas': {0: 'X'}, 'prep': {0: ('S', 1)}},
                {'meas': {0: 'Y'}, 'prep': {0: ('S', 1)}},
                {'meas': {0: 'Z'}, 'prep': {0: ('S', 1)}},
                {'meas': {0: 'X'}, 'prep': {0: ('S', 2)}},
                {'meas': {0: 'Y'}, 'prep': {0: ('S', 2)}},
                {'meas': {0: 'Z'}, 'prep': {0: ('S', 2)}},
                {'meas': {0: 'X'}, 'prep': {0: ('S', 3)}},
                {'meas': {0: 'Y'}, 'prep': {0: ('S', 3)}},
                {'meas': {0: 'Z'}, 'prep': {0: ('S', 3)}
                ],
            'meas_basis': {
                'X': [array([[ 0.5,  0.5], [ 0.5,  0.5]]),
                    array([[ 0.5, -0.5], [-0.5,  0.5]])],
                'Y': [array([[ 0.5+0.j , -0.0-0.5j], [ 0.0+0.5j,  0.5+0.j ]]),
                    array([[ 0.5+0.j ,  0.0+0.5j], [-0.0-0.5j,  0.5+0.j ]])],
                'Z': [array([[1, 0], [0, 0]]),
                    array([[0, 0], [0, 1]])]
                }
            'prep_basis': {
                'S': [
                    array([[1, 0],[0, 0]]),
                    array([[ 0.33333333,  0.47140452],
                            [ 0.47140452, 0.66666667]]),
                    array([[ 0.33333333+0.j, -0.23570226+0.40824829j],
                            [-0.23570226-0.40824829j,  0.66666667+0.j]]),
                    array([[ 0.33333333+0.j, -0.23570226-0.40824829j],
                            [-0.23570226+0.40824829j,  0.66666667+0.j]])
                    ]
                }
            }
        ```
    """
    return tomography_set(qubits, meas_basis=meas_basis, prep_basis=prep_basis)


def tomography_circuit_names(tomo_set, name=''):
    """
    Return a list of tomography circuit names.

    The returned list is the same as the one returned by
    `create_tomography_circuits` and can be used by a QuantumProgram
    to execute tomography circuits and extract measurement results.

    Args:
        tomo_set (tomography_set): a tomography set generated by
        `tomography_set`.
        name: (str): the name of the base QuantumCircuit used by the
        tomography experiment.

    Returns:
        A list of circuit names.
    """

    labels = []
    for circ in tomo_set['circuits']:
        label = ''
        # add prep
        if 'prep' in circ:
            label += '_prep_'
            for qubit, op in circ['prep'].items():
                label += '%s%d(%d)' % (op[0], op[1], qubit)
        # add meas
        label += '_meas_'
        for qubit, op in circ['meas'].items():
            label += '%s(%d)' % (op[0], qubit)
        labels.append(name + label)
    return labels


###############################################################################
# Tomography circuit generation
###############################################################################


def create_tomography_circuits(qp, name, qreg, creg, tomoset, silent=True):
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
        qp (QuantumProgram): A quantum program to store the circuits.
        name (string): The name of the base circuit to be appended.
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        tomoset (tomography_set): the dict of tomography configurations.
        silent (bool, optional): hide verbose output.

    Returns:
        A list of names of the added quantum state tomography circuits.

    Example:
        For a tomography set  specififying state tomography of qubit-0 prepared
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

    dics = tomoset['circuits']
    labels = tomography_circuit_names(tomoset, name)
    circuit = qp.get_circuit(name)

    for label, conf in zip(labels, dics):
        tmp = circuit
        # Add prep circuits
        if 'prep' in conf:
            prep = qp.create_circuit('tmp_prep', [qreg], [creg])
            for q, op in conf['prep'].items():
                tomoset['prep_basis'].prep_gate(prep, qreg[q], op)
                prep.barrier(qreg[q])
            tmp = prep + tmp
            del qp._QuantumProgram__quantum_program['tmp_prep']
        # Add measurement circuits
        meas = qp.create_circuit('tmp_meas', [qreg], [creg])
        for q, op in conf['meas'].items():
            meas.barrier(qreg[q])
            tomoset['meas_basis'].meas_gate(meas, qreg[q], op)
            meas.measure(qreg[q], creg[q])
        tmp = tmp + meas
        del qp._QuantumProgram__quantum_program['tmp_meas']
        # Add tomography circuit
        qp.add_circuit(label, tmp)

    if not silent:
        print('>> created tomography circuits for "%s"' % name)
    return labels


###############################################################################
# Get results data
###############################################################################


def tomography_data(results, name, tomoset):
    """
    Return a results dict for a state or process tomography experiment.

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

    labels = tomography_circuit_names(tomoset, name)
    counts = [
        marginal_counts(results.get_counts(circ), tomoset['qubits'])
        for circ in labels
    ]
    shots = [sum(c.values()) for c in counts]

    ret = {'meas_basis': tomoset['meas_basis']}
    if 'prep_basis' in tomoset:
        ret['prep_basis'] = tomoset['prep_basis']

    ret['data'] = [{
        'counts': c,
        'shots': s,
        'circuit': conf
    } for c, s, conf in zip(counts, shots, tomoset['circuits'])]

    return ret


def marginal_counts(counts, meas_qubits):
    """
    Compute the marginal counts for a subset of measured qubits.

    Args:
        counts (dict{str:int}): the counts returned from a backend.
        meas_qubits (list[int]): the qubits to return the marginal
                                 counts distribution for.

    Returns:
        A counts dict for the meas_qubits.abs
        Example: if `counts = {'00': 10, '01': 5}`
            `marginal_counts(counts, [0])` returns `{'0': 15, '1': 0}`.
            `marginal_counts(counts, [0])` returns `{'0': 10, '1': 5}`.
    """

    # Extract total number of qubits from count keys
    nq = len(list(counts.keys())[0])

    # keys for measured qubits only
    qs = sorted(meas_qubits, reverse=True)

    meas_keys = count_keys(len(qs))

    # get regex match strings for suming outcomes of other qubits
    rgx = [
        reduce(lambda x, y: (key[qs.index(y)] if y in qs else '\\d') + x,
               range(nq), '') for key in meas_keys
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
        A list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
    """
    return [bin(j)[2:].zfill(n) for j in range(2**n)]


###############################################################################
# Tomographic Reconstruction functions.
###############################################################################


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


def __leastsq_fit(tomodata, weights=None, trace=None, beta=None):
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

    data = tomodata['data']
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

    # Convert tomography data bases to projectors
    meas_basis = tomodata['meas_basis']
    ops_m = []
    for dic in data:
        ops_m += __meas_projector(dic['circuit']['meas'], meas_basis)

    if 'prep' in data[0]['circuit']:
        prep_basis = tomodata['prep_basis']
        ops_p = []
        for dic in data:
            p = __prep_projector(dic['circuit']['prep'], prep_basis)
            ops_p += [p for k in ks]  # pad for each meas outcome
        ops = [np.kron(p.T, m) for p, m in zip(ops_p, ops_m)]
    else:
        ops = ops_m

    return __tomo_linear_inv(freqs, ops, weights, trace=trace)


def __meas_projector(dic, basis):
    """Returns a list of measurement outcome projectors.
    """
    meas_opts = [basis[dic[i]] for i in sorted(dic.keys(), reverse=True)]
    ops = []
    for b in it.product(*meas_opts):
        ops.append(reduce(lambda acc, j: np.kron(acc, j), b, [1]))
    return ops


def __prep_projector(dic, basis):
    """Returns a state preparation projector.
    """
    ops = [dic[i] for i in sorted(dic.keys(), reverse=True)]
    ret = [1]
    for b, i in ops:
        ret = np.kron(ret, basis[b][i])
    return ret


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


###############################################################################
# Fit Method: Wizard
###############################################################################


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
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))
    for j in range(dim):
        rho_wizard = rho_wizard + v[j] * outer(w[:, j])
    return rho_wizard


###############################################################################
# DEPRECIATED r0.3 TOMOGRAPHY FUNCTIONS
#
# The following functions are here for backwards compatability with the
# r0.3 QISKit notebooks, but will be removed in the following release.
###############################################################################


def build_state_tomography_circuits(Q_program,
                                    name,
                                    qubits,
                                    qreg,
                                    creg,
                                    meas_basis='pauli',
                                    silent=False):
    """Depreciated function: Use `create_tomography_circuits` function instead.
    """

    tomoset = tomography_set(qubits, meas_basis)
    print('WARNING: `build_state_tomography_circuits` is depreciated. ' +
          'Use `tomography_set` and `create_tomography_circuits` instead')

    return create_tomography_circuits(
        Q_program, name, qreg, creg, tomoset, silent=silent)


def build_process_tomography_circuits(Q_program,
                                      name,
                                      qubits,
                                      qreg,
                                      creg,
                                      prep_basis='sic',
                                      meas_basis='pauli',
                                      silent=False):
    """Depreciated function: Use `create_tomography_circuits` function instead.
    """

    print('WARNING: `build_process_tomography_circuits` is depreciated. ' +
          'Use `tomography_set` and `create_tomography_circuits` instead')

    tomoset = tomography_set(qubits, meas_basis, prep_basis)
    return create_tomography_circuits(
        Q_program, name, qreg, creg, tomoset, silent=silent)


def state_tomography_circuit_names(name, qubits, meas_basis='pauli'):
    """Depreciated function: Use `tomography_circuit_names` function instead.
    """

    print('WARNING: `state_tomography_circuit_names` is depreciated. ' +
          'Use `tomography_set` and `tomography_circuit_names` instead')
    tomoset = tomography_set(qubits, meas_basis=meas_basis)
    return tomography_circuit_names(tomoset, name)


def process_tomography_circuit_names(name,
                                     qubits,
                                     prep_basis='sic',
                                     meas_basis='pauli'):
    """Depreciated function: Use `tomography_circuit_names` function instead.
    """

    print('WARNING: `process_tomography_circuit_names` is depreciated.' +
          'Use `tomography_set` and `tomography_circuit_names` instead')
    tomoset = tomography_set(
        qubits, meas_basis=meas_basis, prep_basis=prep_basis)
    return tomography_circuit_names(tomoset, name)


def state_tomography_data(Q_result, name, meas_qubits, meas_basis='pauli'):
    """Depreciated function: Use `tomography_data` function instead."""

    print('WARNING: `state_tomography_data` is depreciated. ' +
          'Use `tomography_set` and `tomography_data` instead')
    tomoset = tomography_set(meas_qubits, meas_basis=meas_basis)
    return tomography_data(Q_result, name, tomoset)


def process_tomography_data(Q_result,
                            name,
                            meas_qubits,
                            prep_basis='sic',
                            meas_basis='pauli'):
    """Depreciated function: Use `tomography_data` function instead."""

    print('WARNING: `process_tomography_data` is depreciated. ' +
          'Use `tomography_set` and `tomography_data` instead')
    tomoset = tomography_set(
        meas_qubits, meas_basis=meas_basis, prep_basis=prep_basis)
    return tomography_data(Q_result, name, tomoset)


###############################################################
# Wigner function tomography
###############################################################

def build_wigner_circuits(q_program, name, phis, thetas, qubits,
                          qreg, creg, silent=False):
    """Create the circuits to rotate to points in phase space
    Args:
        q_program (QuantumProgram): A quantum program to store the circuits.
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
    orig = q_program.get_circuit(name)
    labels = []
    points = len(phis[0])

    for point in range(points):
        label = '_wigner_phase_point'
        label += str(point)
        circuit = q_program.create_circuit(label, [qreg], [creg])

        for qubit in range(len(qubits)):
            circuit.u3(thetas[qubit][point], 0,
                       phis[qubit][point], qreg[qubits[qubit]])
            circuit.measure(qreg[qubits[qubit]], creg[qubits[qubit]])

        q_program.add_circuit(name + label, orig + circuit)
        labels.append(name + label)

    if not silent:
        print('>> created Wigner function circuits for "%s"' % name)
    return labels


def wigner_data(q_result, meas_qubits, labels, shots=None):
    """Get the value of the Wigner function from measurement results.

    Args:
        q_result (Result): Results from execution of a state tomography
                            circuits on a backend.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        labels : a list of names of the circuits
        shots (int): number of shots

    Returns: The values of the Wigner function at measured points in
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
