# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Implementation of the GraySynth algorithm for synthesizing CNOT-Phase
circuits with efficient CNOT cost, and the Patel-Hayes-Markov algorithm
for optimal synthesis of linear (CNOT-only) reversible circuits.
"""

import copy
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister


def graysynth(cnots, angles, number, nsections):
    """
    This function is an implementation of the GraySynth algorithm.

    GraySynth is a heuristic algorithm for synthesizing small parity networks.
    It is inspired by Gray codes. Given a set of binary strings S
    (called "cnots" bellow), the algorithm synthesizes a parity network for S by
    repeatedly choosing an index ito expand and then effectively recurring on the
    co-factors S_0 and S_1, consisting of the strings y in S with y_i = 0 or 1,
    respectively. As a subset S is recursively expanded, CNOT gates are applied
    so that a designated target bit contains the (partial) parity ksi_y(x) where
    y_i = 1 if and only if y'_i = 1 for for all y' in S. If S is a singleton {y'},
    then y = y', hence the target bit contains the value ksi_y'(x) as desired.
    Notably, rather than uncomputing this sequence of CNOT gates when a subset S
    is finished being synthesized, the algorithm maintains the invariant
    that the remaining parities to be computed are expressed over the current state
    of bits. This allows the algorithm to avoid the 'backtracking' inherent in
    uncomputing-based methods.

    The algorithm is described in detail in the following paper in section 4:
    "On the controlled-NOT complexity of controlled-NOT–phase circuits."
    Amy, Matthew, Parsiad Azimzadeh, and Michele Mosca.
    Quantum Science and Technology 4.1 (2018): 015002.

    Args:
        cnots (list): a binary string called "S" (see function description)
        angles (list): a list containing all the phase-shift gates which are to be applied,
            in the same order as in "cnots". A number is interpreted as the angle
            of u1(angle), otherwise the elements have to be 't', 'tdg', 's', 'sdg' or 'z'
        number (int): the number of quantum bits in the circuit
        nsections (int): number of sections, used in _lwr_cnot_synth(), in the
            Patel–Markov–Hayes algorithm. nsections must be a factor of number.

    Returns:
        QuantumCircuit: the quantum circuit

    Raises:
        Exception: when dimensions of cnots and angles don't align
    """

    # Create a Quantum Register with n quantum bits.
    qreg = QuantumRegister(number, 'q')
    # Create a Quantum Circuit acting on the q register
    qcir = QuantumCircuit(qreg)

    if len(cnots[0]) != len(angles):
        raise Exception('Size of "cnots" and "angles" do not match.')

    range_list = list(range(number))
    epsilon = number
    sta = []
    cnots_copy = np.transpose(np.array(copy.deepcopy(cnots)))
    state = np.eye(number).astype('int')  # This matrix keeps track of the state in the algorithm

    # Check if some T-gates can be applied, before adding any C-NOT gates
    for qubit in range(number):
        index = 0
        for icnots in cnots_copy:
            if np.array_equal(icnots, state[qubit]):
                if angles[index] == 't':
                    qcir.t(qreg[qubit])
                elif angles[index] == 'tdg':
                    qcir.tdg(qreg[qubit])
                elif angles[index] == 's':
                    qcir.s(qreg[qubit])
                elif angles[index] == 'sdg':
                    qcir.sdg(qreg[qubit])
                elif angles[index] == 'z':
                    qcir.z(qreg[qubit])
                else:
                    qcir.u1(angles[index] % np.pi, qreg[qubit])
                del angles[index]
                cnots_copy = np.delete(cnots_copy, index, axis=0)
                if index == len(cnots_copy):
                    break
                index -= 1
            index += 1

    # Implementation of the pseudo-code (Algorithm 1) in the aforementioned paper
    sta.append([cnots, range_list, epsilon])
    while sta != []:
        [cnots, ilist, qubit] = sta.pop()
        if cnots == []:
            continue
        elif 0 <= qubit < number:
            condition = True
            while condition:
                condition = False
                for j in range(number):
                    if (j != qubit) and (sum(cnots[j]) == len(cnots[j])):
                        condition = True
                        qcir.cx(qreg[j], qreg[qubit])
                        state[qubit] ^= state[j]
                        index = 0
                        for icnots in cnots_copy:
                            if np.array_equal(icnots, state[qubit]):
                                if angles[index] == 't':
                                    qcir.t(qreg[qubit])
                                elif angles[index] == 'tdg':
                                    qcir.tdg(qreg[qubit])
                                elif angles[index] == 's':
                                    qcir.s(qreg[qubit])
                                elif angles[index] == 'sdg':
                                    qcir.sdg(qreg[qubit])
                                elif angles[index] == 'z':
                                    qcir.z(qreg[qubit])
                                else:
                                    qcir.u1(angles[index] % np.pi, qreg[qubit])
                                del angles[index]
                                cnots_copy = np.delete(cnots_copy, index, axis=0)
                                if index == len(cnots_copy):
                                    break
                                index -= 1
                            index += 1
                        for x in remove_duplicates(sta + [[cnots, ilist, qubit]]):
                            [cnotsp, _, _] = x
                            if cnotsp == []:
                                continue
                            for ttt in range(len(cnotsp[j])):
                                cnotsp[j][ttt] ^= cnotsp[qubit][ttt]
        if ilist == []:
            continue
        # See line 18 in pseudo-code of Algorithm 1 in the aforementioned paper
        j = ilist[np.argmax([[max(row.count(0), row.count(1)) for row in cnots][k] for k in ilist])]
        cnots0 = []
        cnots1 = []
        for y in list(map(list, zip(*cnots))):
            if y[j] == 0:
                cnots0.append(y)
            elif y[j] == 1:
                cnots1.append(y)
        cnots0 = list(map(list, zip(*cnots0)))
        cnots1 = list(map(list, zip(*cnots1)))
        if qubit == epsilon:
            sta.append([cnots1, list(set(ilist).difference([j])), j])
        else:
            sta.append([cnots1, list(set(ilist).difference([j])), qubit])
        sta.append([cnots0, list(set(ilist).difference([j])), qubit])
    qcir = cnot_synth(qcir, state, number, nsections)
    return qcir


def cnot_synth(qcir, state, number, nsections):
    """
    This function is an implementation of the Patel–Markov–Hayes algorithm
    for optimal synthesis of linear reversible circuits. It takes a CNOT-only
    quantum circuit "qcir", and uncomputes all its CNOT gates in an optimal way.

    The algorithm is described in detail in the following paper:
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Args:
        qcir (QuantumCircuit): the initial Quantum Circuit
        state (numpy.matrix): n x n matrix, describing the state of the input circuit
        number (int): the number of quantum bits in the circuit
        nsections (int): number of sections, used in _lwr_cnot_synth(), in the
            Patel–Markov–Hayes algorithm. nsections must be a factor of number.

    Returns:
        QuantumCircuit: a Quantum Circuit with added CNOT gates which
            unfompute the original circuit

    Raises:
        Exception: when variable "state" isn't of type numpy.matrix
    """

    if not isinstance(state, np.ndarray):
        raise Exception('state should be of type numpy.ndarray, but was '
                        'of the type {}'.format(type(state)))
    # Synthesize lower triangular part
    [state, circuit_l] = _lwr_cnot_synth(state, number, nsections)
    state = np.transpose(state)
    # Synthesize upper triangular part
    [state, circuit_u] = _lwr_cnot_synth(state, number, nsections)
    circuit_u.reverse()
    for i in circuit_u:
        i.reverse()
    # Convert the list into a circuit of C-NOT gates
    for i in circuit_l + circuit_u:
        qcir.cx(i[0], i[1])
    return qcir


def _lwr_cnot_synth(state, number, nsections):
    """
    This function is a helper function of the algorithm for optimal synthesis
    of linear reversible circuits (the Patel–Markov–Hayes algorithm). It works
    like gaussian elimination, except that it works a lot faster, and requires
    fewer steps (and therefore fewer CNOTs). It takes the matrix "state" and
    splits it into "nsections" sections. Then it eliminates all non-zero
    sub-rows within each sections, which are the same as a non-zero sub-section
    above. Once this has been done, it continues with normal gaussian elimination.

    The algorithm is described in detail in the following paper
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Args:
        state (numpy.matrix): n by n matrix, describing the state of the input circuit
        number (int): the number of quantum bits in the circuit
        nsections (int): the number of sections used in the below algorithm (see description)

    Returns:
        numpy.matrix: n by n matrix, describing the state of the output circuit
        list: a k by 2 list of C-NOT operations that need to be applied
    """

    circuit = []
    # If the matrix is already an upper triangular one, there is no need for any transformations
    if np.allclose(state, np.triu(state)):
        return [state, circuit]
    # Iterate over column sections
    for sec in range(1, int(np.ceil(number/nsections)+1)):
        # Remove duplicate sub-rows in section sec
        patt = {}
        for row in range((sec-1)*nsections, number):
            sub_row_patt = copy.deepcopy(state[row, (sec-1)*nsections:sec*nsections])
            if np.sum(sub_row_patt) == 0:
                continue
            if str(sub_row_patt) not in patt:
                patt[str(sub_row_patt)] = row
            else:
                state[row, :] ^= state[patt[str(sub_row_patt)], :]
                circuit.append([patt[str(sub_row_patt)], row])
        # Use gaussian elimination for remaining entries in column section
        for col in range((sec-1)*nsections, sec*nsections):
            # Check if 1 on diagonal
            diag_one = 1
            if state[col, col] == 0:
                diag_one = 0
            # Remove ones in rows below column col
            for row in range(col+1, number):
                if state[row, col] == 1:
                    if diag_one == 0:
                        state[col, :] ^= state[row, :]
                        circuit.append([row, col])
                        diag_one = 1
                    state[row, :] ^= state[col, :]
                    circuit.append([col, row])
    return [state, circuit]


def remove_duplicates(lists):
    """
    Remove duplicates in list

    Args:
        lists (list): a list which may contain duplicate elements.

    Returns:
        list: a list which contains only unique elements.
    """

    unique_list = []
    for element in lists:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list
