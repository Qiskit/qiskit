# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the QAOA algorithm with opflow."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from functools import partial
import numpy as np

from scipy.optimize import minimize as scipy_minimize
from ddt import ddt, idata, unpack

import rustworkx as rx

from qiskit import QuantumCircuit
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA, NELDER_MEAD
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Pauli
from qiskit.result import QuasiDistribution
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals

I = PauliSumOp.from_list([("I", 1)])
X = PauliSumOp.from_list([("X", 1)])

W1 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
P1 = 1
M1 = (I ^ I ^ I ^ X) + (I ^ I ^ X ^ I) + (I ^ X ^ I ^ I) + (X ^ I ^ I ^ I)
S1 = {"0101", "1010"}


W2 = np.array(
    [
        [0.0, 8.0, -9.0, 0.0],
        [8.0, 0.0, 7.0, 9.0],
        [-9.0, 7.0, 0.0, -8.0],
        [0.0, 9.0, -8.0, 0.0],
    ]
)
P2 = 1
M2 = None
S2 = {"1011", "0100"}

CUSTOM_SUPERPOSITION = [1 / np.sqrt(15)] * 15 + [0]


@ddt
class TestQAOA(QiskitAlgorithmsTestCase):
    """Test QAOA with MaxCut."""

    def setUp(self):
        super().setUp()
        self.seed = 10598
        algorithm_globals.random_seed = self.seed
        self.sampler = Sampler()

    @idata(
        [
            [W1, P1, M1, S1],
            [W2, P2, M2, S2],
        ]
    )
    @unpack
    def test_qaoa(self, w, reps, mixer, solutions):
        """QAOA test"""
        self.log.debug("Testing %s-step QAOA with MaxCut on graph\n%s", reps, w)

        qubit_op, _ = self._get_operator(w)

        qaoa = QAOA(self.sampler, COBYLA(), reps=reps, mixer=mixer)
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, solutions)

    @idata(
        [
            [W1, P1, S1],
            [W2, P2, S2],
        ]
    )
    @unpack
    def test_qaoa_qc_mixer(self, w, prob, solutions):
        """QAOA test with a mixer as a parameterized circuit"""
        self.log.debug(
            "Testing %s-step QAOA with MaxCut on graph with a mixer as a parameterized circuit\n%s",
            prob,
            w,
        )

        optimizer = COBYLA()
        qubit_op, _ = self._get_operator(w)

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        theta = Parameter("θ")
        mixer.rx(theta, range(num_qubits))

        qaoa = QAOA(self.sampler, optimizer, reps=prob, mixer=mixer)
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, solutions)

    def test_qaoa_qc_mixer_many_parameters(self):
        """QAOA test with a mixer as a parameterized circuit with the num of parameters > 1."""
        optimizer = COBYLA()
        qubit_op, _ = self._get_operator(W1)

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            theta = Parameter("θ" + str(i))
            mixer.rx(theta, range(num_qubits))

        qaoa = QAOA(self.sampler, optimizer, reps=2, mixer=mixer)
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        self.log.debug(x)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, S1)

    def test_qaoa_qc_mixer_no_parameters(self):
        """QAOA test with a mixer as a parameterized circuit with zero parameters."""
        qubit_op, _ = self._get_operator(W1)

        num_qubits = qubit_op.num_qubits
        mixer = QuantumCircuit(num_qubits)
        # just arbitrary circuit
        mixer.rx(np.pi / 2, range(num_qubits))

        qaoa = QAOA(self.sampler, COBYLA(), reps=1, mixer=mixer)
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        # we just assert that we get a result, it is not meaningful.
        self.assertIsNotNone(result.eigenstate)

    def test_change_operator_size(self):
        """QAOA change operator size test"""
        qubit_op, _ = self._get_operator(
            np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        )
        qaoa = QAOA(self.sampler, COBYLA(), reps=1)
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        with self.subTest(msg="QAOA 4x4"):
            self.assertIn(graph_solution, {"0101", "1010"})

        qubit_op, _ = self._get_operator(
            np.array(
                [
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0],
                ]
            )
        )
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        with self.subTest(msg="QAOA 6x6"):
            self.assertIn(graph_solution, {"010101", "101010"})

    @idata([[W2, S2, None], [W2, S2, [0.0, 0.0]], [W2, S2, [1.0, 0.8]]])
    @unpack
    def test_qaoa_initial_point(self, w, solutions, init_pt):
        """Check first parameter value used is initial point as expected"""
        qubit_op, _ = self._get_operator(w)

        first_pt = []

        def cb_callback(eval_count, parameters, mean, metadata):
            nonlocal first_pt
            if eval_count == 1:
                first_pt = list(parameters)

        qaoa = QAOA(
            self.sampler,
            COBYLA(),
            initial_point=init_pt,
            callback=cb_callback,
        )
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)

        with self.subTest("Initial Point"):
            # If None the preferred random initial point of QAOA variational form
            if init_pt is None:
                self.assertLess(result.eigenvalue, -0.97)
            else:
                self.assertListEqual(init_pt, first_pt)

        with self.subTest("Solution"):
            self.assertIn(graph_solution, solutions)

    def test_qaoa_random_initial_point(self):
        """QAOA random initial point"""
        w = rx.adjacency_matrix(
            rx.undirected_gnp_random_graph(5, 0.5, seed=algorithm_globals.random_seed)
        )
        qubit_op, _ = self._get_operator(w)
        qaoa = QAOA(self.sampler, NELDER_MEAD(disp=True), reps=2)
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(operator=qubit_op)

        self.assertLess(result.eigenvalue, -0.97)

    def test_optimizer_scipy_callable(self):
        """Test passing a SciPy optimizer directly as callable."""
        w = rx.adjacency_matrix(
            rx.undirected_gnp_random_graph(5, 0.5, seed=algorithm_globals.random_seed)
        )
        qubit_op, _ = self._get_operator(w)
        qaoa = QAOA(
            self.sampler,
            partial(scipy_minimize, method="Nelder-Mead", options={"maxiter": 2}),
        )
        with self.assertWarns(DeprecationWarning):
            result = qaoa.compute_minimum_eigenvalue(qubit_op)
        self.assertEqual(result.cost_function_evals, 5)

    def _get_operator(self, weight_matrix):
        """Generate Hamiltonian for the max-cut problem of a graph.

        Args:
            weight_matrix (numpy.ndarray) : adjacency matrix.

        Returns:
            PauliSumOp: operator for the Hamiltonian
            float: a constant shift for the obj function.

        """
        num_nodes = weight_matrix.shape[0]
        pauli_list = []
        shift = 0
        for i in range(num_nodes):
            for j in range(i):
                if weight_matrix[i, j] != 0:
                    x_p = np.zeros(num_nodes, dtype=bool)
                    z_p = np.zeros(num_nodes, dtype=bool)
                    z_p[i] = True
                    z_p[j] = True
                    pauli_list.append([0.5 * weight_matrix[i, j], Pauli((z_p, x_p))])
                    shift -= 0.5 * weight_matrix[i, j]
        opflow_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
        with self.assertWarns(DeprecationWarning):
            return PauliSumOp.from_list(opflow_list), shift

    def _get_graph_solution(self, x: np.ndarray) -> str:
        """Get graph solution from binary string.

        Args:
            x : binary string as numpy array.

        Returns:
            a graph solution as string.
        """

        return "".join([str(int(i)) for i in 1 - x])

    def _sample_most_likely(self, state_vector: QuasiDistribution) -> np.ndarray:
        """Compute the most likely binary string from state vector.
        Args:
            state_vector: Quasi-distribution.

        Returns:
            Binary string as numpy.ndarray of ints.
        """
        values = list(state_vector.values())
        n = int(np.log2(len(values)))
        k = np.argmax(np.abs(values))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x


if __name__ == "__main__":
    unittest.main()
