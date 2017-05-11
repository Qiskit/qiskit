# Pauli operators
#
# A Pauli Opt.
#
# Jay Gambetta <jay.gambetta@us.ibm.com>
# Andrew Cross <awcross@us.ibm.com>

import random
import numpy as np


class Pauli:
    """A simple class representing Pauli Operators
        The Form is P = (-i)^dot(v,w) Z^v X^w
            where v element of Z_2^n
            where w element of Z_2^n
        That is there are 4^n elements (no phases in this group)

       For example for 1 qubit
       P_00 = Z^0 X^0 = I
       P_01 = X
       P_10 = Z
       P_11 = -iZX = (-i) iY = Y

       Multiplication is P1*P2 = (-i)^dot(v1+v2,w1+w2) Z^(v1+v2) X^(w1+w2) where sums are mod 2
    """

    def __init__(self, v, w):
        """ makes the Pauli class  """
        self.v = v
        self.w = w
        self.numberofqubits = len(v)

    def __str__(self):
        """ Outputs the pauli as first row v and second row w"""
        stemp = '\nv = '
        for i in self.v:
            stemp += str(i) + '\t'
        stemp = stemp + '\nw = '
        for j in self.w:
            stemp += str(j) + '\t'
        return stemp + '\n'

    def __mul__(self, other):
        """ Multiples two Paulis """
        if self.numberofqubits != other.numberofqubits:
            print('Paulis cannot be multipled - different number of qubits')
        vnew = (self.v + other.v) % 2
        wnew = (self.w + other.w) % 2
        # print vnew
        # print wnew
        paulinew = Pauli(vnew, wnew)
        return paulinew

    def toLabel(self):
        """ prints out the labels in X, Y, Z format"""
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

    def toQASM(self, qubits):
        """ prints out the qasm format for the Pauli"""
        if len(qubits) == self.numberofqubits:
            qasmlabel = ''
            for jindex in qubits:
                if self.v[jindex] == 0 and self.w[jindex] == 0:
                    qasmlabel += 'id q[' + str(jindex) + '];\n'  # identity
                elif self.v[jindex] == 0 and self.w[jindex] == 1:
                    qasmlabel += 'u3(-pi,0,-pi) q[' + str(jindex) + '];\n'  # x
                elif self.v[jindex] == 1 and self.w[jindex] == 1:
                    # y
                    qasmlabel += 'u3(-pi,0,2*pi) q[' + str(jindex) + '];\n'
                elif self.v[jindex] == 1 and self.w[jindex] == 0:
                    qasmlabel += 'u1(pi) q[' + str(jindex) + '];\n'  # z
            qasmlabel += 'barrier q;\n'
            return qasmlabel
        else:
            print('the qubit vector did match the pauli')
            return -1

    def to_matrix(self):
        """ converts pauli to a matrix representation"""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        Xtemp = 1
        # print self.numberofqubits
        for k in range(self.numberofqubits):
            # print k
            # print self.v[k]
            # print self.w[k]
            if self.v[k] == 0:
                tempz = I
            elif self.v[k] == 1:
                tempz = Z
            else:
                print('the z string is not of the form 0 and 1')
            if self.w[k] == 0:
                tempx = I
            elif self.w[k] == 1:
                tempx = X
            else:
                print('the x string is not of the form 0 and 1')
            ope = np.dot(tempz, tempx)
            Xtemp = np.kron(Xtemp, ope)
            # print Xtemp
        paulimat = (-1j)**np.dot(self.v, self.w) * Xtemp
        # print paulimat
        return paulimat


def random_pauli(numberofqubits):
    '''This function returns a random Pauli on numberofqubits
    '''
    v = np.array(list(bin(random.getrandbits(numberofqubits))
                      [2:].zfill(numberofqubits))).astype(np.int)
    w = np.array(list(bin(random.getrandbits(numberofqubits))
                      [2:].zfill(numberofqubits))).astype(np.int)
    return Pauli(v, w)


def InversePauli(other):
    '''This function returns the inverse of a pauli. This is honestly a trival function but to make sure that it is consistent with the clifford benchmarking
    '''
    v = other.v
    w = other.w
    return Pauli(v, w)


def pauli_group(numberofqubits, case=0):
    ''' Returns the Pauli group -- 4^n elements where the phases have been removed.
    case 0 is by ordering by Pauli weights and
    case 1 is by ordering by I,X,Y,Z counting last qubit fastest
    @param numberofqubits
    @param case determines ordering of group elements (0=weight,1=tensor)
    @return list of Pauli objects
    '''
    if numberofqubits < 5:
        tempset = []
        if case == 0:
            tmp = PauliGroup(numberofqubits, case=1)
            # sort on the weight of the Pauli operator
            return sorted(tmp, key=lambda x: -
                          np.count_nonzero(np.array(x.toLabel(), 'c') == b'I'))

        elif case == 1:
            # the pauli set is in tensor order II IX IY IZ XI ...
            for kindex in range(4**numberofqubits):
                v = np.zeros(numberofqubits)
                w = np.zeros(numberofqubits)
                # looping over all the qubits
                for jindex in range(numberofqubits):
                    # making the pauli for each kindex i fill it in from the
                    # end first
                    element = int((kindex) / (4**(jindex))) % 4
                    if element == 0:
                        v[numberofqubits - jindex - 1] = 0
                        w[numberofqubits - jindex - 1] = 0
                    elif element == 1:
                        v[numberofqubits - jindex - 1] = 0
                        w[numberofqubits - jindex - 1] = 1
                    elif element == 2:
                        v[numberofqubits - jindex - 1] = 1
                        w[numberofqubits - jindex - 1] = 1
                    elif element == 3:
                        v[numberofqubits - jindex - 1] = 1
                        w[numberofqubits - jindex - 1] = 0
                tempset.append(Pauli(v, w))
            return tempset
    else:
        print('please set the number of qubits less then 5')
        return -1
