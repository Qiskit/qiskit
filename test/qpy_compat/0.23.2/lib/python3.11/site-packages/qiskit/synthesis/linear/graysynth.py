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


def graysynth(cnots, angles, section_size=2):
    """This function is an implementation of the GraySynth algorithm.

    GraySynth is a heuristic algorithm for synthesizing small parity networks.
    It is inspired by Gray codes. Given a set of binary strings S
    (called "cnots" bellow), the algorithm synthesizes a parity network for S by
    repeatedly choosing an index i to expand and then effectively recursing on
    the co-factors S_0 and S_1, consisting of the strings y in S,
    with y_i = 0 or 1 respectively. As a subset S is recursively expanded,
    CNOT gates are applied so that a designated target bit contains the
    (partial) parity ksi_y(x) where y_i = 1 if and only if y'_i = 1 for for all
    y' in S. If S is a singleton {y'}, then y = y', hence the target bit contains
    the value ksi_y'(x) as desired.

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
            interpreted as the angle of u1(angle), otherwise the elements
            have to be 't', 'tdg', 's', 'sdg' or 'z'.

        section_size (int): the size of every section, used in _lwr_cnot_synth(), in the
            Patel–Markov–Hayes algorithm. section_size must be a factor of num_qubits.

    Returns:
        QuantumCircuit: the quantum circuit

    Raises:
        QiskitError: when dimensions of cnots and angles don't align
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


def synth_cnot_count_full_pmh(state, section_size=2):
    """
    Synthesize linear reversible circuits for all-to-all architecture
    using Patel, Markov and Hayes method.

    This function is an implementation of the Patel, Markov and Hayes algorithm from [1]
    for optimal synthesis of linear reversible circuits for all-to-all architecture,
    as specified by an n x n matrix.

    Args:
        state (list[list] or ndarray): n x n boolean invertible matrix, describing the state
            of the input circuit
        section_size (int): the size of each section, used in the
            Patel–Markov–Hayes algorithm [1]. section_size must be a factor of num_qubits.

    Returns:
        QuantumCircuit: a CX-only circuit implementing the linear transformation.

    Raises:
        QiskitError: when variable "state" isn't of type numpy.ndarray

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """
    if not isinstance(state, (list, np.ndarray)):
        raise QiskitError(
            "state should be of type list or numpy.ndarray, "
            "but was of the type {}".format(type(state))
        )
    state = np.array(state)
    # Synthesize lower triangular part
    [state, circuit_l] = _lwr_cnot_synth(state, section_size)
    state = np.transpose(state)
    # Synthesize upper triangular part
    [state, circuit_u] = _lwr_cnot_synth(state, section_size)
    circuit_l.reverse()
    for i in circuit_u:
        i.reverse()
    # Convert the list into a circuit of C-NOT gates
    circ = QuantumCircuit(state.shape[0])
    for i in circuit_u + circuit_l:
        circ.cx(i[0], i[1])
    return circ


def _lwr_cnot_synth(state, section_size):
    """
    This function is a helper function of the algorithm for optimal synthesis
    of linear reversible circuits (the Patel–Markov–Hayes algorithm). It works
    like gaussian elimination, except that it works a lot faster, and requires
    fewer steps (and therefore fewer CNOTs). It takes the matrix "state" and
    splits it into sections of size section_size. Then it eliminates all non-zero
    sub-rows within each section, which are the same as a non-zero sub-row
    above. Once this has been done, it continues with normal gaussian elimination.
    The benefit is that with small section sizes (m), most of the sub-rows will
    be cleared in the first step, resulting in a factor m fewer row row operations
    during Gaussian elimination.

    The algorithm is described in detail in the following paper
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Note:
    This implementation tweaks the Patel, Markov, and Hayes algorithm by adding
    a "back reduce" which adds rows below the pivot row with a high degree of
    overlap back to it. The intuition is to avoid a high-weight pivot row
    increasing the weight of lower rows.

    Args:
        state (ndarray): n x n matrix, describing a linear quantum circuit
        section_size (int): the section size the matrix columns are divided into

    Returns:
        numpy.matrix: n by n matrix, describing the state of the output circuit
        list: a k by 2 list of C-NOT operations that need to be applied
    """
    circuit = []
    num_qubits = state.shape[0]
    cutoff = 1

    # Iterate over column sections
    for sec in range(1, int(np.floor(num_qubits / section_size) + 1)):
        # Remove duplicate sub-rows in section sec
        patt = {}
        for row in range((sec - 1) * section_size, num_qubits):
            sub_row_patt = copy.deepcopy(state[row, (sec - 1) * section_size : sec * section_size])
            if np.sum(sub_row_patt) == 0:
                continue
            if str(sub_row_patt) not in patt:
                patt[str(sub_row_patt)] = row
            else:
                state[row, :] ^= state[patt[str(sub_row_patt)], :]
                circuit.append([patt[str(sub_row_patt)], row])
        # Use gaussian elimination for remaining entries in column section
        for col in range((sec - 1) * section_size, sec * section_size):
            # Check if 1 on diagonal
            diag_one = 1
            if state[col, col] == 0:
                diag_one = 0
            # Remove ones in rows below column col
            for row in range(col + 1, num_qubits):
                if state[row, col] == 1:
                    if diag_one == 0:
                        state[col, :] ^= state[row, :]
                        circuit.append([row, col])
                        diag_one = 1
                    state[row, :] ^= state[col, :]
                    circuit.append([col, row])
                # Back reduce the pivot row using the current row
                if sum(state[col, :] & state[row, :]) > cutoff:
                    state[col, :] ^= state[row, :]
                    circuit.append([row, col])
    return [state, circuit]


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
