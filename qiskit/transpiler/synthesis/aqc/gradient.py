# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Defines classes to compute gradient."""

from abc import abstractmethod, ABC
from typing import Union, List

import numpy as np
from numpy import linalg as la

from .elementary_operations import op_ry, op_rz, op_unitary, op_cnot, op_rx, X, Y, Z


class GradientBase(ABC):
    """Interface to any class that computes gradient and objective function."""

    def __init__(self):
        pass

    @abstractmethod
    def get_gradient(
        self, thetas: Union[List[float], np.ndarray], target_matrix: np.ndarray
    ) -> (float, np.ndarray):
        """
        Computes gradient and objective function.
        Args:
            thetas: an array of angles.
            target_matrix: an original circuit represented as a unitary matrix.
        Returns:
            objective function value, gradient.
        """
        raise NotImplementedError("Abstract method is called!")


# TODO: replace with FastGradient?
class DefaultGradient(GradientBase):
    """A default implementation of a gradient computation."""

    def __init__(self, num_qubits: int, cnots: np.ndarray) -> None:
        """
        Args:
            num_qubits: number of qubits.
            cnots: a CNOT structure to be used.
        """
        super().__init__()
        assert isinstance(num_qubits, int) and 1 <= num_qubits <= 16
        assert isinstance(cnots, np.ndarray)
        assert cnots.ndim == 2 and cnots.shape[0] == 2
        self._num_qubits = num_qubits
        self._cnots = cnots
        self._num_cnots = cnots.shape[1]

    def get_gradient(self, thetas: Union[List[float], np.ndarray], target_matrix: np.ndarray):
        # Get the gradient of the cost function
        # todo: compare to Rx, etc
        x = np.multiply(-1j / 2, X)
        y = np.multiply(-1j / 2, Y)
        z = np.multiply(-1j / 2, Z)
        n = self._num_qubits
        cnots = self._cnots

        L = np.shape(cnots)[1]

        # compute parametric circuit and prepare required matrices for gradient computations
        # we start from the parametric circuit
        C = np.zeros((2 ** n, 2 ** n * L)) + 0j
        S = np.zeros((2 ** n, 2 ** n * L)) + 0j
        T = np.zeros((2 ** n, 2 ** n * L)) + 0j
        for l in range(L):
            p = 4 * l
            a = op_ry(thetas[0 + p])
            b = op_rz(thetas[1 + p])
            c = op_ry(thetas[2 + p])
            d = op_rx(thetas[3 + p])
            q1 = int(cnots[0, l])
            q2 = int(cnots[1, l])
            u1 = np.dot(b, a)
            u2 = np.dot(d, c)
            U1 = op_unitary(u1, n, q1)
            U2 = op_unitary(u2, n, q2)
            cnot1 = op_cnot(n, q1, q2)
            C[:, 2 ** n * l : 2 ** n * (l + 1)] = la.multi_dot([U2, U1, cnot1])
        V = np.eye(2 ** n)
        for l in range(L - 1, -1, -1):
            V = np.dot(V, C[:, 2 ** n * l : 2 ** n * (l + 1)])
            S[:, 2 ** n * l : 2 ** n * (l + 1)] = V
        V = np.eye(2 ** n)
        for l in range(L):
            V = np.dot(C[:, 2 ** n * l : 2 ** n * (l + 1)], V)
            T[:, 2 ** n * l : 2 ** n * (l + 1)] = V
        V2 = V
        V1 = 1
        for k in range(n):
            p = 4 * L + 3 * k
            # a = Rx(thetas[0 + p])
            a = op_rz(thetas[0 + p])
            b = op_ry(thetas[1 + p])
            c = op_rz(thetas[2 + p])
            V1 = np.kron(V1, la.multi_dot([a, b, c]))
        V = np.dot(V2, V1)

        # compute error
        err = 0.5 * (la.norm(V - target_matrix, "fro") ** 2)

        # compute gradient
        der = np.zeros(4 * L + 3 * n)
        for l in range(L):
            p = 4 * l
            a = op_ry(thetas[0 + p])
            b = op_rz(thetas[1 + p])
            c = op_ry(thetas[2 + p])
            d = op_rx(thetas[3 + p])
            q1 = int(cnots[0, l])
            q2 = int(cnots[1, l])
            cnot1 = op_cnot(n, q1, q2)
            for i in range(4):
                if i == 0:
                    u1 = la.multi_dot([b, y, a])
                    u2 = np.dot(d, c)
                # TODO: replace with elif
                if i == 1:
                    u1 = la.multi_dot([z, b, a])
                    u2 = np.dot(d, c)
                if i == 2:
                    u1 = np.dot(b, a)
                    u2 = la.multi_dot([d, y, c])
                if i == 3:
                    u1 = np.dot(b, a)
                    u2 = la.multi_dot([x, d, c])
                U1 = op_unitary(u1, n, q1)
                U2 = op_unitary(u2, n, q2)
                dC = la.multi_dot([U2, U1, cnot1])
                if l == 0:
                    dV = np.dot(S[:, 2 ** n * (l + 1) : 2 ** n * (l + 2)], dC)
                elif L - 1 == l:
                    dV = np.dot(dC, T[:, 2 ** n * (l - 1) : 2 ** n * l])
                else:
                    dV = la.multi_dot(
                        [
                            S[:, 2 ** n * (l + 1) : 2 ** n * (l + 2)],
                            dC,
                            T[:, 2 ** n * (l - 1) : 2 ** n * l],
                        ]
                    )
                dV = np.dot(dV, V1)
                der[i + p] = -np.real(np.trace(np.dot(dV.conj().T, target_matrix)))
        for i in range(3 * n):
            dV1 = 1
            for k in range(n):
                p = 4 * L + 3 * k
                # a = Rx(thetas[0 + p])
                a = op_rz(thetas[0 + p])
                b = op_ry(thetas[1 + p])
                c = op_rz(thetas[2 + p])
                if i - 3 * k == 0:
                    # a = np.dot(x, a)
                    a = np.dot(z, a)
                elif i - 3 * k == 1:
                    b = np.dot(y, b)
                elif i - 3 * k == 2:
                    c = np.dot(z, c)
                dV1 = np.kron(dV1, la.multi_dot([a, b, c]))
            dV = np.dot(V2, dV1)
            der[4 * L + i] = -np.real(np.trace(np.dot(dV.conj().T, target_matrix)))  # V-

        # return error, gradient
        return err, der

    # TODO: this method is not used now!
    # def _get_smallgrad(self, thetas):
    #     # This method is called only when num_cnots == 1
    #     # todo: double check we need this method
    #     n = self._num_qubits
    #     cnots = self._cnots
    #     err = self._get_cost(thetas)
    #     L = np.shape(cnots)[1]
    #     p = 4 * L + 3 * n
    #     der = np.zeros(p)
    #     for l in range(p):
    #         ddV = self._dd_circuit(thetas, l, l)
    #         der[l] = -np.real(np.trace(np.dot(ddV.conj().T, self.U)))
    #     return err, der

    # TODO: this method is referenced in the _get_smallgrad, which is not used
    # def _dd_circuit(self, thetas, j, m):
    #     # TODO: No idea of what this does
    #     n = self._num_qubits
    #     cnots = self._cnots
    #     x = np.multiply(-1j / 2, X)
    #     y = np.multiply(-1j / 2, Y)
    #     z = np.multiply(-1j / 2, Z)
    #     L = np.shape(cnots)[1]
    #     ddV = np.eye(2 ** n)
    #     for l in range(L):
    #         p = 4 * l
    #         a = Ry(thetas[0 + p])
    #         b = Rz(thetas[1 + p])
    #         c = Ry(thetas[2 + p])
    #         d = Rx(thetas[3 + p])
    #         if 0 + p == j or 0 + p == m:
    #             a = np.dot(y, a)
    #         if 1 + p == j or 1 + p == m:
    #             b = np.dot(z, b)
    #         if 2 + p == j or 2 + p == m:
    #             c = np.dot(y, c)
    #         if 3 + p == j or 3 + p == m:
    #             d = np.dot(x, d)
    #         q1 = int(cnots[0, l])
    #         q2 = int(cnots[1, l])
    #         u1 = np.dot(b, a)
    #         u2 = np.dot(d, c)
    #         U1 = unitary(u1, n, q1)
    #         U2 = unitary(u2, n, q2)
    #         CNOT1 = CNOT(n, q1, q2)
    #         C = la.multi_dot([U2, U1, CNOT1])
    #         ddV = np.dot(C, ddV)
    #     ddV1 = 1
    #     for k in range(n):
    #         p = 4 * L + 3 * k
    #         # a = Rx(thetas[0 + p])
    #         a = Rz(thetas[0 + p])
    #         b = Ry(thetas[1 + p])
    #         c = Rz(thetas[2 + p])
    #         if 0 + p == j or 0 + p == m:
    #             # a = np.dot(x, a)
    #             a = np.dot(z, a)
    #         if 1 + p == j or 1 + p == m:
    #             b = np.dot(y, b)
    #         if 2 + p == j or 2 + p == m:
    #             c = np.dot(z, c)
    #         ddV1 = np.kron(ddV1, la.multi_dot([a, b, c]))
    #     ddV = np.dot(ddV, ddV1)
    #     return ddV
