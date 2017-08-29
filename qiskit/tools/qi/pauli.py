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
Tools for working with Pauli Operators.

A simple pauli class and some tools.
"""
import random
import numpy as np


class Pauli:
    """A simple class representing Pauli Operators.

    The form is P = (-i)^dot(v,w) Z^v X^w where v and w are elements of Z_2^n.
    That is, there are 4^n elements (no phases in this group).

    For example, for 1 qubit
    P_00 = Z^0 X^0 = I
    P_01 = X
    P_10 = Z
    P_11 = -iZX = (-i) iY = Y

    Multiplication is P1*P2 = (-i)^dot(v1+v2,w1+w2) Z^(v1+v2) X^(w1+w2)
    where the sums are taken modulo 2.

    Ref.
    Jeroen Dehaene and Bart De Moor
    Clifford group, stabilizer states, and linear and quadratic operations over GF(2)
    Phys. Rev. A 68, 042318 – Published 20 October 2003
    """

    def __init__(self, v, w):
        """Make the Pauli class."""
        self.numberofqubits = len(v)
        self.v = v
        self.w = w

    def __str__(self):
        """Output the Pauli as first row v and second row w."""
        stemp = 'v = '
        for i in self.v:
            stemp += str(i) + '\t'
        stemp = stemp + '\nw = '
        for j in self.w:
            stemp += str(j) + '\t'
        return stemp

    def __mul__(self, other):
        """Multiply two Paulis."""
        if self.numberofqubits != other.numberofqubits:
            print('Paulis cannot be multiplied - different number of qubits')
        vnew = (self.v + other.v) % 2
        wnew = (self.w + other.w) % 2
        paulinew = Pauli(vnew, wnew)
        return paulinew

    def to_label(self):
        """Print out the labels in X, Y, Z format."""
        plabel = ''
        for jindex in range(self.numberofqubits):
            if self.v[jindex] == 0 and self.w[jindex] == 0:
                plabel += 'I'
            elif self.v[jindex] == 0 and self.w[jindex] == 1:
                plabel += 'X'
            elif self.v[jindex] == 1 and self.w[jindex] == 1:
                plabel += 'Y'
            elif self.v[jindex] == 1 and self.w[jindex] == 0:
                plabel += 'Z'
        return plabel

    def to_matrix(self):
        """Convert Pauli to a matrix representation.

        Order is q_n x q_{n-1} .... q_0
        """
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        Id = np.array([[1, 0], [0, 1]], dtype=complex)
        Xtemp = 1
        for k in range(self.numberofqubits):
            if self.v[k] == 0:
                tempz = Id
            elif self.v[k] == 1:
                tempz = Z
            else:
                print('the z string is not of the form 0 and 1')
            if self.w[k] == 0:
                tempx = Id
            elif self.w[k] == 1:
                tempx = X
            else:
                print('the x string is not of the form 0 and 1')
            ope = np.dot(tempz, tempx)
            Xtemp = np.kron(ope, Xtemp)
        pauli_mat = (-1j)**np.dot(self.v, self.w) * Xtemp
        return pauli_mat


def random_pauli(numberofqubits):
    """Return a random Pauli on numberofqubits."""
    v = np.array(list(bin(random.getrandbits(numberofqubits))
                      [2:].zfill(numberofqubits))).astype(np.int)
    w = np.array(list(bin(random.getrandbits(numberofqubits))
                      [2:].zfill(numberofqubits))).astype(np.int)
    return Pauli(v, w)

def sgn_prod(P1, P2):
    """Multiply two Paulis P1*P2 and track the sign.

    P3 = P1*P2: X*Y
    """

    if P1.numberofqubits != P2.numberofqubits:
        print('Paulis cannot be multiplied - different number of qubits')
    vnew = (P1.v + P2.v) % 2
    wnew = (P1.w + P2.w) % 2
    paulinew = Pauli(vnew, wnew)
    phase=1
    for i in range(len(P1.v)):
        if P1.v[i]==1 and P1.w[i]==0 and P2.v[i]==0 and P2.w[i]==1:  # Z*X
            phase=1j*phase
        elif P1.v[i]==0 and P1.w[i]==1 and P2.v[i]==1 and P2.w[i]==0:  # X*Z
            phase=-1j*phase
        elif P1.v[i]==0 and P1.w[i]==1 and P2.v[i]==1 and P2.w[i]==1:  # X*Y
            phase=1j*phase
        elif P1.v[i]==1 and P1.w[i]==1 and P2.v[i]==0 and P2.w[i]==1:  # Y*X
            phase=-1j*phase
        elif P1.v[i]==1 and P1.w[i]==1 and P2.v[i]==1 and P2.w[i]==0:  # Y*Z
            phase=1j*phase
        elif P1.v[i]==1 and P1.w[i]==0 and P2.v[i]==1 and P2.w[i]==1:  # Z*Y
            phase=-1j*phase

    return paulinew, phase

def inverse_pauli(other):
    """Return the inverse of a Pauli."""
    v = other.v
    w = other.w
    return Pauli(v, w)

def label_to_pauli(label):
    """Return the pauli of a string ."""
    v = np.zeros(len(label))
    w = np.zeros(len(label))
    for j in range(len(label)):
        if label[j] == 'I':
            v[j] = 0
            w[j] = 0
        elif label[j] == 'Z':
            v[j] = 1
            w[j] = 0
        elif label[j] == 'Y':
            v[j] = 1
            w[j] = 1
        elif label[j] == 'X':
            v[j] = 0
            w[j] = 1
        else:
            print('something went wrong')
            return -1
    return Pauli(v, w)

def pauli_group(numberofqubits, case=0):
    """Return the Pauli group with 4^n elements.

    The phases have been removed.
    case 0 is ordered by Pauli weights and
    case 1 is ordered by I,X,Y,Z counting last qubit fastest.
    @param numberofqubits is number of qubits
    @param case determines ordering of group elements (0=weight, 1=tensor)
    @return list of Pauli objects
    WARNING THIS IS EXPONENTIAL
    """
    if numberofqubits < 5:
        tempset = []
        if case == 0:
            tmp = pauli_group(numberofqubits, case=1)
            # sort on the weight of the Pauli operator
            return sorted(tmp, key=lambda x: -
                          np.count_nonzero(np.array(x.to_label(), 'c') == b'I'))

        elif case == 1:
            # the Pauli set is in tensor order II IX IY IZ XI ...
            for kindex in range(4**numberofqubits):
                v = np.zeros(numberofqubits)
                w = np.zeros(numberofqubits)
                # looping over all the qubits
                for jindex in range(numberofqubits):
                    # making the Pauli for each kindex i fill it in from the
                    # end first
                    element = int((kindex) / (4**(jindex))) % 4
                    if element == 0:
                        v[jindex] = 0
                        w[jindex] = 0
                    elif element == 1:
                        v[jindex] = 0
                        w[jindex] = 1
                    elif element == 2:
                        v[jindex] = 1
                        w[jindex] = 1
                    elif element == 3:
                        v[jindex] = 1
                        w[jindex] = 0
                tempset.append(Pauli(v, w))
            return tempset
    else:
        print('please set the number of qubits to less than 5')
        return -1

def pauli_singles(jindex, numberofqubits):
    """Return the single qubit pauli in numberofqubits."""
    # looping over all the qubits
    tempset =[]
    v = np.zeros(numberofqubits)
    w = np.zeros(numberofqubits)
    v[jindex] = 0
    w[jindex] = 1
    tempset.append(Pauli(v, w))
    v = np.zeros(numberofqubits)
    w = np.zeros(numberofqubits)
    v[jindex] = 1
    w[jindex] = 1
    tempset.append(Pauli(v, w))
    v = np.zeros(numberofqubits)
    w = np.zeros(numberofqubits)
    v[jindex] = 1
    w[jindex] = 0
    tempset.append(Pauli(v, w))
    return tempset
