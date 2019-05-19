# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Implementation of the Gray-Synth algorithm and a efficient synthesis algorithm of linear
reversible circuits.
"""
import copy
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def graysynth(cnots, number, nsections):
    """
    This function is an implementation of the GraySynth algorithm.
    The algorithm is described in the following paper in section 4:
    "On the controlled-NOT complexity of controlled-NOT–phase circuits."
    Amy, Matthew, Parsiad Azimzadeh, and Michele Mosca.
    Quantum Science and Technology 4.1 (2018): 015002.

    Args:
        cnots (list): as described in the aforementioned paper.
        number (int): the number of quantum bits in the circuit
        nsections (int): the number of sections

    Returns:
        QuantumCircuit: the quantum circuit
        QuantumRegister: the quantum register
        numpy.matrix: an n by n matrix describing the output state of the circuit
    """

    # Create a Quantum Register with n quantum bits.
    qreg = QuantumRegister(number, 'q')
    # Create a Quantum Circuit acting on the q register
    qcir = QuantumCircuit(qreg)

    range_list = list(range(number))
    epsilon = number
    sta = Stack()
    cnots_copy = np.transpose(np.array(copy.deepcopy(cnots)))
    state = np.eye(number).astype('int')  # This matrix keeps track of the state in the algorithm

    # Check if some T-gates can be applied, before adding any C-NOT gates
    for qubit in range(number):
        index = 0
        for icnots in cnots_copy:
            if np.array_equal(icnots, state[qubit]):
                qcir.t(qreg[qubit])
                cnots_copy = np.delete(cnots_copy, index, axis=0)
                if index == len(cnots_copy):
                    break
                index -= 1
            index += 1

    # Implementation of the pseudo-code (Algorithm 1) in the aforementioned paper
    sta.push([cnots, range_list, epsilon])
    while not sta.isempty():
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
                                qcir.t(qreg[qubit])
                                cnots_copy = np.delete(cnots_copy, index, axis=0)
                                if index == len(cnots_copy):
                                    break
                                index -= 1
                            index += 1
                        for x in union(sta.elements() + [[cnots, ilist, qubit]]):
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
            sta.push([cnots1, list(set(ilist).difference([j])), j])
        else:
            sta.push([cnots1, list(set(ilist).difference([j])), qubit])
        sta.push([cnots0, list(set(ilist).difference([j])), qubit])
    [qcir, state] = cnot_synth(qcir, state, qreg, number, nsections)
    return [qcir, qreg, state]


def cnot_synth(qcir, state, qreg, number, nsections):
    """
    This function is an implementation of the algorithm for optimal synthesis of linear
    reversible circuits, as described in the following paper:
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Args:
        qcir (QuantumCircuit): the initial Quantum Circuit
        state (numpy.matrix): n by n matrix, describing the state of the input circuit
        qreg (QuantumRegister): a Quantum Register
        number (int): the number of quantum bits in the circuit
        nsections (int): the number of partitions used in the below algorithm

    Returns:
        QuantumCircuit: a Quantum Circuit with added C-NOT gates
        numpy.matrix: n by n matrix, describing the state of the output circuit
    """

    state = np.matrix(state)  # Making sure that state is a numpy matrix
    # Synthesize lower triangular part
    [state, circuit_l] = lwr_cnot_synth(state, number, nsections)
    circuit_l.reverse()
    state = np.transpose(state)
    # Synthesize upper triangular part
    [state, circuit_u] = lwr_cnot_synth(state, number, nsections)
    for i in circuit_u:
        i.reverse()
    # Convert the list into a circuit of C-NOT gates
    for i in circuit_u+circuit_l:
        qcir.cx(qreg[i[0]], qreg[i[1]])
    return [qcir, state]


def lwr_cnot_synth(state, number, nsections):
    """
    This function is a helper function of the algorithm for optimal synthesis of
    linear reversible circuits, as described in the following paper:
    "Optimal synthesis of linear reversible circuits."
    Patel, Ketan N., Igor L. Markov, and John P. Hayes.
    Quantum Information & Computation 8.3 (2008): 282-294.

    Args:
        state (numpy.matrix): n by n matrix, describing the state of the input circuit
        number (int): the number of quantum bits in the circuit
        nsections (int): the number of partitions used in the below algorithm

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


class Stack:
    """
    Implementation of a Stack with the possibility of returning a list of all elements at once
    """

    def __init__(self):
        self.items = []

    def isempty(self):
        """
        Check if stack is empty
        """
        return self.items == []

    def push(self, item):
        """
        Add element to stack
        """
        self.items.append(item)

    def pop(self):
        """
        Remove and return last element in stack
        """
        return self.items.pop()

    def size(self):
        """
        Return length of stack
        """
        return len(self.items)

    def elements(self):
        """
        Return all elements in stack
        """
        return self.items


def union(lists):
    """
    Remove duplicates in list

    Args:
        lists (list): a list which may contain duplicate elements.

    Returns:
        list: a list which contains only unique elements.
    """

    union_list = []
    for element in lists:
        if element not in union_list:
            union_list.append(element)
    return union_list
