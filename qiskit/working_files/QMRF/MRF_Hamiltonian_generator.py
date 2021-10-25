import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize, linewidth=1024)

from typing import List, Iterable

import itertools

class MrfHamiltonianGenerator():
    def __init(self):
        return

    # Mark all elements in the diagonal whose binary representation has the bits in index_set
    # asserted
    def Phi_slow(self, c, y, n):
        result = np.zeros(2 ** n * 2 ** n).reshape(2 ** n, 2 ** n)

        X = list(itertools.product([0, 1], repeat=n))  # state space of size 2^n
        #  This generates all ones in Phi with complexity 2^n!! We need something else here!
        for j, x in enumerate(X):
            valid = True
            for i, v in enumerate(c):
                if y[i] != x[v]:
                    valid = False
                    break
            if valid:
                result[j, j] = 1
        return result


    # Mark all elements in the diagonal whose binary representation has the bits in index_set
    # asserted
    def Phi_fast(self, c, y, n):
        I = np.array([[1, 0], [0, 1]])
        Z = np.array([[1, 0], [0, -1]])
        result = 1
        plus = [v for i, v in enumerate(c) if not y[i]]
        minus = [v for i, v in enumerate(c) if y[i]]

        s = 2.0 ** (len(plus) + len(minus))

        # This is the solution
        for i in range(n):
            f = I
            if i in minus:
                f = I - Z
            elif i in plus:
                f = I + Z

            result = np.kron(result, f)

        return result / s


    # Mark all elements in the diagonal whose binary representation has the bits in index_set
    # asserted
    def Phi_fast_pauli(self, c, y, n):
        from qiskit.opflow import I, Z
        result = None
        plus = [v for i, v in enumerate(c) if not y[i]]
        minus = [v for i, v in enumerate(c) if y[i]]

        s = 2.0 ** (len(plus) + len(minus))

        # This is the solution
        for i in range(n):
            if i in minus:
                f = I - Z
            elif i in plus:
                f = I + Z
            else:
                f = I

            if result is None:
                result = f
            else:
                result = result ^ f

        return result / s

    def isXOR(self, x, y, z):
        r = (z == (x != y))
        # print(x,y,z,r) # yes, this checks indeed wheter z = x XOR y
        return r

    def gen_Hamiltonian(self,
                        clique_structure: List[List],
                        n: int,
                        mode: str = True) -> List[List]:
        """

        Args:
            clique_structure: Clique structure
            n: Number of nodes/qubits
            mode: Use inefficient matrix construction ('slow')
                  Use efficient matrix construction ('fast')
                  Use efficient Pauli construction ('fast_pauli')

        Returns: MRF Hamiltonian as a matrix or PauliSummedOp

        """
        if not mode == 'fast_pauli':
            H = np.zeros(2 ** n * 2 ** n).reshape(2 ** n, 2 ** n)  # Hamitonian of size 2^n X 2^n
        else:
            H = None

        for l, c in enumerate(clique_structure):
            Y = list(itertools.product([0, 1], repeat=len(c)))
            for y in Y:
                if mode == 'fast':
                    Phi = self.Phi_fast(c, y, n)
                elif mode == 'fast_pauli':
                    Phi = self.Phi_fast_pauli(c, y, n)
                else:
                    Phi = self.Phi_slow(c, y, n)

                theta = -1  # theta will be negated when 3-cliques are not in valid XOR state or the
                # 2-clique is 00 and 11
                if (l == 0 or l == 1) and self.isXOR(y[0], y[1], y[2]):
                    theta *= -1
                elif l == 2 and (y[0] != y[1]):
                    theta *= -1
                if H is None:
                    H = theta * Phi
                else:
                    H += theta * Phi

        # H[H > 0] = 3
        # H[H < 0] = -3

        return H



# C = [[0, 1, 2], [3, 4, 5], [2, 3]]  # clique structure
# n = 6  # number of (qu)bits
#
# H0 = MrfHamiltonianGenerator().gen_Hamiltonian(C, n, 'slow')
# H1 = MrfHamiltonianGenerator().gen_Hamiltonian(C, n, 'fast')
#
# print(hex(hash(H0.tobytes())))
# print(hex(hash(H1.tobytes())))
#
# assert (H0 == H1).all()
#
# print(H0)
# print(H1.tolist())
