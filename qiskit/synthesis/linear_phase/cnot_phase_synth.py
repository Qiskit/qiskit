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
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.synthesis.linear import synth_cnot_count_full_pmh


def synth_cnot_phase_aam(cnots, angles, section_size=2):
    """This function is an implementation of the GraySynth algorithm of
    Amy, Azimadeh and Mosca.

    GraySynth is a heuristic algorithm from [1] for synthesizing small parity networks.
    It is inspired by Gray codes. Given a set of binary strings S
    (called "cnots" bellow), the algorithm synthesizes a parity network for S by
    repeatedly choosing an index i to expand and then effectively recursing on
    the co-factors S_0 and S_1, consisting of the strings y in S,
    with y_i = 0 or 1 respectively. As a subset S is recursively expanded,
    CNOT gates are applied so that a designated target bit contains the
    (partial) parity ksi_y(x) where y_i = 1 if and only if y'_i = 1 for all
    y' in S. If S is a singleton {y'}, then y = y', hence the target bit contains
    the value ksi_y'(x) as desired.

    Notably, rather than uncomputing this sequence of CNOT gates when a subset S
    is finished being synthesized, the algorithm maintains the invariant
    that the remaining parities to be computed are expressed over the current state
    of bits. This allows the algorithm to avoid the 'backtracking' inherent in
    uncomputing-based methods.

    The algorithm is described in detail in section 4 of [1].

    Args:
        cnots (list[list]): a matrix whose columns are the parities to be synthesized
            e.g.::

                [[0, 1, 1, 1, 1, 1],
                 [1, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 1, 0]]

            corresponds to::

                 x1^x2 + x0 + x0^x3 + x0^x1^x2 + x0^x1^x3 + x0^x1

        angles (list): a list containing all the phase-shift gates which are
            to be applied, in the same order as in "cnots". A number is
            interpreted as the angle of p(angle), otherwise the elements
            have to be 't', 'tdg', 's', 'sdg' or 'z'.

        section_size (int): the size of every section, used in _lwr_cnot_synth(), in the
            Patel–Markov–Hayes algorithm. section_size must be a factor of num_qubits.

    Returns:
        QuantumCircuit: the decomposed quantum circuit.

    Raises:
        QiskitError: when dimensions of cnots and angles don't align.

    References:
        1. Matthew Amy, Parsiad Azimzadeh, and Michele Mosca.
           *On the controlled-NOT complexity of controlled-NOT–phase circuits.*,
           Quantum Science and Technology 4.1 (2018): 015002.
           `arXiv:1712.01859 <https://arxiv.org/abs/1712.01859>`_
    """
    num_qubits = len(cnots)

    # Create a quantum circuit on num_qubits
    qcir = QuantumCircuit(num_qubits)

    if len(cnots[0]) != len(angles):
        raise QiskitError('Size of "cnots" and "angles" do not match.')

    range_list = list(range(num_qubits))
    epsilon = num_qubits
    sta = []
    cnots_copy = np.transpose(np.array(copy.deepcopy(cnots)))
    # This matrix keeps track of the state in the algorithm
    state = np.eye(num_qubits).astype("int")

    # Check if some phase-shift gates can be applied, before adding any C-NOT gates
    for qubit in range(num_qubits):
        index = 0
        for icnots in cnots_copy:
            if np.array_equal(icnots, state[qubit]):
                if angles[index] == "t":
                    qcir.t(qubit)
                elif angles[index] == "tdg":
                    qcir.tdg(qubit)
                elif angles[index] == "s":
                    qcir.s(qubit)
                elif angles[index] == "sdg":
                    qcir.sdg(qubit)
                elif angles[index] == "z":
                    qcir.z(qubit)
                else:
                    qcir.p(angles[index] % np.pi, qubit)
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
        if 0 <= qubit < num_qubits:
            condition = True
            while condition:
                condition = False
                for j in range(num_qubits):
                    if (j != qubit) and (sum(cnots[j]) == len(cnots[j])):
                        condition = True
                        qcir.cx(j, qubit)
                        state[qubit] ^= state[j]
                        index = 0
                        for icnots in cnots_copy:
                            if np.array_equal(icnots, state[qubit]):
                                if angles[index] == "t":
                                    qcir.t(qubit)
                                elif angles[index] == "tdg":
                                    qcir.tdg(qubit)
                                elif angles[index] == "s":
                                    qcir.s(qubit)
                                elif angles[index] == "sdg":
                                    qcir.sdg(qubit)
                                elif angles[index] == "z":
                                    qcir.z(qubit)
                                else:
                                    qcir.p(angles[index] % np.pi, qubit)
                                del angles[index]
                                cnots_copy = np.delete(cnots_copy, index, axis=0)
                                if index == len(cnots_copy):
                                    break
                                index -= 1
                            index += 1
                        for x in _remove_duplicates(sta + [[cnots, ilist, qubit]]):
                            [cnotsp, _, _] = x
                            if cnotsp == []:
                                continue
                            for ttt in range(len(cnotsp[j])):
                                cnotsp[j][ttt] ^= cnotsp[qubit][ttt]
        if ilist == []:
            continue
        # See line 18 in pseudo-code of Algorithm 1 in the aforementioned paper
        # this choice of j maximizes the size of the largest subset (S_0 or S_1)
        # and the larger a subset, the closer it gets to the ideal in the
        # Gray code of one CNOT per string.
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
    qcir &= synth_cnot_count_full_pmh(state, section_size).inverse()
    return qcir


def _remove_duplicates(lists):
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
