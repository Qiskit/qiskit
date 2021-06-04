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

import unittest
import numpy as np

from qiskit.transpiler.synthesis.aqc.optimizers import GDOptimizer, FISTAOptimizer
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit
from test_sample_data import (
    GD_THETAS,
    GD_OBJECTIVE,
    GD_GRADIENT,
    GD_MIN_THETAS,
    FISTA_THETAS,
    FISTA_OBJECTIVE,
    FISTA_GRADIENT,
    ORIGINAL_CIRCUIT,
    INITIAL_THETAS,
    FISTA_GD_THETAS,
    FISTA_GD_OBJECTIVE,
    FISTA_GD_GRADIENT,
    FISTA_GD_MIN_THETA,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase


class TestMain(QiskitTestCase):
    def setUp(self) -> None:
        self._seed = 12345
        np.random.seed(self._seed)
        self._num_qubits = 3  # num qubits
        self._links = full_conn(self._num_qubits)

        # self._U = random_su_gate(self._num_qubits)
        self._circuit_matrix = ORIGINAL_CIRCUIT

        # array 2 x num_cnots
        self._cnots = spin(self._num_qubits, self._links, num_cnots=0)

        # L == num is cnots (paper)
        self._num_cnots = np.shape(self._cnots)[1]
        # p == num rotation angles (paper)
        self._num_rotations = 4 * self._num_cnots + 3 * self._num_qubits

        # self._thetas0 = np.random.uniform(0, 2 * np.pi, self._p)
        # todo: remove this variable
        self._thetas0 = INITIAL_THETAS

        self._maxiter = int(5e3)
        self._eta = 0.1  # .1 for n=3, .01 for n=5
        self._tol = 0.01

    def test_main_gradient_descent(self):
        # print('L start value: ', self._L)
        # print(self._cnots)

        parametric_circuit = ParametricCircuit(self._num_qubits, self._cnots)
        # optimizer = Optimizer(self._num_qubits, self._cnots, self._circuit_matrix, parametric_circuit)
        optimizer = GDOptimizer(
            method="nesterov", maxiter=self._maxiter, eta=self._eta, tol=self._tol, eps=0.01
        )
        # self._num_qubits, self._cnots, self._circuit_matrix, parametric_circuit)

        # thetas0 = [i for i in range(p)]
        original_circuit = parametric_circuit.make_circuit(self._thetas0)
        # V.Plot(original_circuit)

        original_depth = original_circuit.depth()
        print("Original depth ", original_depth)
        self.assertEqual(original_depth, 45)

        # transpile by Qiskit
        # todo: why do we need this? we don't use the output
        # qiskit_circuit = parametric_circuit.my_transpile2(original_circuit, plot_flag=0, seed_transpiler=self._seed)

        # todo: Check that your circuit is equivalent to the original or transpiled circuit ?
        # why we need this?
        check_qiskit_equivalence(
            parametric_circuit.compute_unitary(self._thetas0), original_circuit
        )

        # Optimize with gradient descent
        (thetas, obj, gra, min_theta) = optimizer.optimize(
            self._circuit_matrix, self._thetas0, parametric_circuit
        )
        np.testing.assert_array_almost_equal(thetas, GD_THETAS, 4)
        np.testing.assert_array_almost_equal(obj, GD_OBJECTIVE, 4)
        np.testing.assert_array_almost_equal(gra, GD_GRADIENT, 4)
        np.testing.assert_array_almost_equal(min_theta, GD_MIN_THETAS, 4)

        # plotting approximate error
        # i = np.shape(obj)[0]
        # iteration = np.linspace(1, i, num=i)
        # print("Iteration", iteration)
        # plt.loglog(iteration, obj)
        # plt.xlabel('Iteration')
        # plt.ylabel('Approximation error')
        # plt.show()

    def test_main_fista(self):
        # Optimize with FISTA, compress, re-optimize
        # print(self._cnots)
        V = ParametricCircuit(self._num_qubits, self._cnots)
        # optimizer = Optimizer(self._num_qubits, self._cnots, self._circuit_matrix, V)
        optimizer = FISTAOptimizer(
            method="nesterov",
            maxiter=self._maxiter,
            eta=self._eta,
            tol=self._tol,
            eps=0.0,
            reg=0.7,
            group=True,
        )

        print("FISTA")
        (thetas, obj, gra, _) = optimizer.optimize(self._circuit_matrix, self._thetas0, V)
        # print(list(thetas))
        # print(list(obj))
        # print(list(gra))

        np.testing.assert_array_almost_equal(thetas, FISTA_THETAS, 4)
        np.testing.assert_array_almost_equal(obj, FISTA_OBJECTIVE, 4)
        np.testing.assert_array_almost_equal(gra, FISTA_GRADIENT, 4)

        print("COMPRESSING")
        compr_cnots, cmp_thetas, spar = V.compress(thetas, synth=False)

        # todo: add check that compression is correct, see QiskitReducer

        # V.Plot(V.make_circuit(thetas))

        L = np.shape(compr_cnots)[1]
        # todo: L and p are the same as initial
        self.assertEqual(L, 14)
        p = 4 * L + 3 * self._num_qubits
        self.assertEqual(p, 65)

        # thetas0 = np.random.uniform(0, 2 * np.pi, p)
        V_compr = ParametricCircuit(self._num_qubits, compr_cnots)
        # V.Plot(V_compr.make_circuit(thetas0))

        # optimizer_compr = Optimizer(self._num_qubits, compr_cnots, self._circuit_matrix, V_compr)
        optimizer_compr = GDOptimizer(
            method="nesterov", maxiter=self._maxiter, eta=self._eta, tol=self._tol, eps=0.0
        )
        print("GRADIENT")

        (thetas, obj, gra, min_theta) = optimizer_compr.optimize(
            self._circuit_matrix, cmp_thetas, V_compr
        )
        np.testing.assert_array_almost_equal(thetas, FISTA_GD_THETAS)
        np.testing.assert_array_almost_equal(obj, FISTA_GD_OBJECTIVE)
        np.testing.assert_array_almost_equal(gra, FISTA_GD_GRADIENT)
        np.testing.assert_array_almost_equal(min_theta, FISTA_GD_MIN_THETA)

        # plot error
        # i = np.shape(obj)[0]
        # iteration = np.linspace(1, i, num=i)
        #
        # plt.loglog(iteration, obj)
        # plt.xlabel('Iteration')
        # plt.ylabel('Approximation error')
        # plt.show()

        # print(obj[i - 1])
        # print(thetas)

        target_circuit = V_compr.make_circuit(thetas, tol=1e-2)
        print("Final depth ", target_circuit.depth())
        # todo: check depth


def check_qiskit_equivalence(Vm: np.ndarray, qc: QuantumCircuit) -> float:
    Vc = Operator(qc).data
    alpha = np.angle(np.trace(np.dot(Vm.conj().T, Vc)))
    Vnew = np.exp(-1j * alpha) * Vc
    equiv_norm = np.linalg.norm(Vnew - Vm)
    print("Equivalence norm: ", equiv_norm)
    return equiv_norm
