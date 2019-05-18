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

"""Implementation of the Gray-Synth algorithm.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
import copy

class Stack:
    """
    Run one pass of cx cancellation on the circuit

    Args:
        dag (DAGCircuit): the directed acyclic graph to run on.
    Returns:
        DAGCircuit: Transformed DAG.
    """
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

    def elements(self):
        return self.items


def Union(lists):
    """
    Run one pass of cx cancellation on the circuit

    Args:
        dag (DAGCircuit): the directed acyclic graph to run on.
    Returns:
        DAGCircuit: Transformed DAG.
    """
    Union_list = []
    for element in lists:
        if not(element in Union_list):
            Union_list.append(element)
    return Union_list


def GraySynth(S, n):
    """
    Run one pass of cx cancellation on the circuit

    Args:
        dag (DAGCircuit): the directed acyclic graph to run on.
    Returns:
        DAGCircuit: Transformed DAG.
    """

    q = QuantumRegister(n, 'q') # Create a Quantum Register with n qubits.
    C = QuantumCircuit(q) # Create a Quantum Circuit acting on the q register

    range_list = list(range(n))
    epsilon = n
    Q = Stack()
    S_copy = np.transpose(np.array(copy.deepcopy(S)))
    state = np.eye(n).astype('int')

    for i in range(n):
        for ii in range(len(S_copy)):
            if np.array_equal(S_copy[ii], state[i]):
                C.t(q[i])
                S_copy = np.delete(S_copy, (ii), axis=0)
                break
    Q.push([S, range_list, epsilon])
    while not (Q.isEmpty()):
        [S, I, i] = Q.pop()
        if len(S) == 0:
            continue
        elif (i >= 0 and i < n):
            condition = True
            while condition:
                condition = False
                for j in range(n):
                    if (j != i) and (sum(S[j]) == len(S[j])):
                        condition = True
                        C.cx(q[j], q[i])
                        state[i] ^= state[j]
                        for ii in range(len(S_copy)):
                            if np.array_equal(S_copy[ii], state[i]):
                                C.t(q[i])
                                S_copy = np.delete(S_copy, (ii), axis=0)
                                break
                        for x in Union(Q.elements() + [[S, I, i]]):
                            [Sp, Ip, ip] = x
                            if len(Sp) == 0:
                                continue
                            for t in range(len(Sp[j])):
                                Sp[j][t] ^= Sp[i][t]
        if len(I) == 0:
            continue
        j = I[np.argmax([[max(row.count(0), row.count(1)) for row in S][k] for k in
                         I])]  # see line 18 in pseudo-code
        S0 = []
        S1 = []
        for y in list(map(list, zip(*S))):
            if y[j] == 0:
                S0.append(y)
            elif y[j] == 1:
                S1.append(y)
        S0 = list(map(list, zip(*S0)))
        S1 = list(map(list, zip(*S1)))
        if i == epsilon:
            Q.push([S1, list(set(I).difference([j])), j])
        else:
            Q.push([S1, list(set(I).difference([j])), i])
        Q.push([S0, list(set(I).difference([j])), i])
    return [C, q, state]


def Lwr_CNOT_Synth(A, n, m):
    circuit = []
    if np.allclose(A, np.triu(A)):
        return [A, circuit]
    for sec in range(1,int(np.ceil(n/m)+1)): #iterate over column sections
        #remove duplicate sub-rows in section sec
        patt = {}
        for row in range((sec-1)*m,n):
            sub_row_patt = copy.deepcopy(A[row,(sec-1)*m:sec*m])
            if not str(sub_row_patt) in patt:
                patt[str(sub_row_patt)] = row
            else:
                A[row,:] ^= A[patt[str(sub_row_patt)],:]
                circuit.append([patt[str(sub_row_patt)], row])
        #Use gaussian elimination for remaining entries in column section
        for col in range((sec-1)*m,sec*m):
            #Check if 1 on diagonal
            diag_one = 1
            if A[col,col] == 0:
                diag_one = 0
            #Remove ones in rows below column col
            for row in range(col+1,n):
                if A[row,col] == 1:
                    if diag_one == 0:
                        A[col,:] ^= A[row,:]
                        circuit.append([row, col])
                        diag_one = 1
                    A[row,:] ^= A[col,:]
                    circuit.append([col, row])
    return [A, circuit]


def CNOT_Synth(state, C, q, n, m):
    state = np.matrix(state)
    #Synthesize lower/upper tringular part
    [state, circuit_l] = Lwr_CNOT_Synth(state, n, m)
    circuit_l.reverse()
    state = np.transpose(state)
    [state, circuit_u] = Lwr_CNOT_Synth(state, n, m)
    for i in range(len(circuit_u)):
        circuit_u[i].reverse()
    for i in range(len(circuit_u+circuit_l)):
        C.cx(q[circuit_u[i][0]], q[circuit_u[i][1]])
    return [state, C]










