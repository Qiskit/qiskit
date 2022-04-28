# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test AdaptQAOA """
import unittest
# from test.python.algorithms import QiskitAlgorithmsTestCase

import math
import numpy as np
from ddt import ddt, idata, unpack

from qiskit.algorithms import AdaptQAOA
from qiskit.circuit.library.n_local.adaptqaoa_ansatz import adapt_mixer_pool

from qiskit.opflow import I, X, Z, PauliSumOp

from qiskit import BasicAer, QuantumCircuit, QuantumRegister

from qiskit.quantum_info import Pauli
from qiskit.utils import QuantumInstance, algorithm_globals

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
P2 = 2      # a higher number of reps are required to generate the correct solution
M2 = None   # when the complete mixer pool is used, (i.e. mixer_pool = None)
S2 = {"1011", "0100"}

CUSTOM_SUPERPOSITION = [1 / math.sqrt(15)] * 15 + [0]

from qiskit.test import QiskitTestCase

@ddt
# class TestAdaptQAOA(QiskitAlgorithmsTestCase):
class TestAdaptQAOA(QiskitTestCase):
    """Test AdaptQAOA with MaxCut."""

    def setUp(self):
        super().setUp()
        self.seed = 10598
        algorithm_globals.random_seed = self.seed

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=4096,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=2**14,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

    @idata(
        [
            [W1, P1, M1, S1, False],
            [W2, P2, M2, S2, False],
            [W1, P1, M1, S1, True],
            [W2, P2, M2, S2, True],
        ]
    )
    @unpack
    def test_adapt_qaoa(self, w, prob, m, solutions, convert_to_matrix_op):
        """AdaptQAOA test"""
        self.log.debug("Testing %s-step AdaptQAOA with MaxCut on graph\n%s", prob, w)
        qubit_op, _ = self._get_operator(w)
        if convert_to_matrix_op:
            qubit_op = qubit_op.to_matrix_op()

        adapt_qaoa = AdaptQAOA(
            max_reps=prob,
            mixer_pool=m,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, solutions)

    @idata(
        [
            [W1, P1, S1, False],    #TODO: mismatch err
            [W2, P2, S2, False],
            [W1, P1, S1, True],
            [W2, P2, S2, True],
        ]
    )
    @unpack
    def test_adapt_qaoa_qc_mixer(self, w, prob, solutions, convert_to_matrix_op):
        """AdaptQAOA test with a mixer as a circuit"""
        self.log.debug(
            "Testing %s-step AdaptQAOA with MaxCut on graph with "
            "a mixer as a parameterized circuit\n%s",
            prob,
            w,
        )

        qubit_op, _ = self._get_operator(w)
        if convert_to_matrix_op:
            qubit_op = qubit_op.to_matrix_op()

        num_qubits = qubit_op.num_qubits
        mixer = adapt_mixer_pool(
            num_qubits = num_qubits, add_multi=True, 
            circuit_rep=True, circ_params = np.arange(num_qubits))

        adapt_qaoa = AdaptQAOA(
            max_reps=prob,
            mixer_pool=mixer,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, solutions)

    @idata([["multi"],["singular"],["single"]])
    @unpack
    def test_adapt_qaoa_qc_mixer_type(self, pool_type):
        """AdaptQAOA test with no mixer_pool specified but mixer_pool_type is specified"""
        qubit_op, _ = self._get_operator(W2)

        adapt_qaoa = AdaptQAOA(
            mixer_pool_type = pool_type,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, S2)

    def test_adapt_qaoa_qc_mixer_many_parameters(self):
        """AdaptQAOA test with a mixer as a parameterized circuit with the num of parameters > 1."""
        qubit_op, _ = self._get_operator(W1)

        num_qubits = qubit_op.num_qubits
        mixers = adapt_mixer_pool(num_qubits = num_qubits, pool_type = 'multi', circuit_rep = True,
                circ_params = np.arange(num_qubits), num_params = num_qubits)
        adapt_qaoa = AdaptQAOA(
            mixer_pool=mixers,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        self.log.debug(x)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, S1)

    def test_adapt_qaoa_qc_mixer_no_parameters(self):
        """AdaptQAOA test with a mixer pool as a list of circuits with zero parameters."""
        qubit_op, _ = self._get_operator(W1)

        num_qubits = qubit_op.num_qubits
        mixer = adapt_mixer_pool(num_qubits = num_qubits, pool_type='multi', circuit_rep=True)

        adapt_qaoa = AdaptQAOA(
            mixer_pool=mixer,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        # we just assert that we get a result, it is not meaningful.
        self.assertIsNotNone(result.eigenstate)

    @idata([[None], [[1.0, 1.0, 0.8, 0.8]]])
    @unpack
    def test_change_operator_size(self, init_pt):
        """AdaptQAOA change operator size test"""
        qubit_op, _ = self._get_operator(W2)
        adapt_qaoa = AdaptQAOA(
            max_reps=P2,
            initial_point=init_pt,
            quantum_instance=self.statevector_simulator,
        )

        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        with self.subTest(msg="AdaptQAOA 4x4"):
            self.assertIn(graph_solution, S2)

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

        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        with self.subTest(msg="AdaptQAOA 6x6"):
            self.assertIn(graph_solution, {"010101", "101010"})

    @idata([[W2, S2, None], [W2, S2, [0.0, 0.0]], [W2, S2, [1.0, 0.8]]])
    @unpack
    def test_adapt_qaoa_initial_point(self, w, solutions, init_pt):
        """Check first parameter value used is initial point as expected"""
        qubit_op, _ = self._get_operator(w)

        first_pt = []

        def cb_callback(eval_count, parameters, mean, std):
            nonlocal first_pt
            if eval_count == 1:
                first_pt = list(parameters)

        adapt_qaoa = AdaptQAOA(
            initial_point=init_pt,
            callback=cb_callback,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)

        with self.subTest("Initial Point"):
            # If None the preferred random initial point of QAOA variational form
            if init_pt is None:
                self.assertLess(result.eigenvalue.real, -0.97)
            else:
                self.assertListEqual(init_pt, first_pt)

        with self.subTest("Solution"):
            self.assertIn(graph_solution, solutions)

    @idata([[W2, None], [W2, [1.0] + 15 * [0.0]], [W2, CUSTOM_SUPERPOSITION]])
    @unpack
    def test_adapt_qaoa_initial_state(self, w, init_state):
        """AdaptQAOA initial state test"""
        qubit_op, _ = self._get_operator(w)
        num_qubits = qubit_op.num_qubits

        init_pt = np.asarray([0.0, 0.0])  # Avoid generating random initial point

        if init_state is None:
            initial_state = None
        else:
            initial_state = QuantumCircuit(QuantumRegister(num_qubits, "q"))
            initial_state.initialize(init_state, initial_state.qubits)

        zero_init_state = QuantumCircuit(QuantumRegister(num_qubits, "q"))
        adapt_qaoa_zero_init_state = AdaptQAOA(
            mixer_pool = M1,
            initial_state=zero_init_state,
            initial_point=init_pt,
            quantum_instance=self.statevector_simulator,
        )
        adapt_qaoa = AdaptQAOA(
            mixer_pool = X ^ num_qubits,
            initial_state=initial_state,
            initial_point=init_pt,
            quantum_instance=self.statevector_simulator,
        )

        adapt_qaoa.compute_minimum_eigenvalue(qubit_op)
        adapt_qaoa_zero_init_state.compute_minimum_eigenvalue(qubit_op)

        zero_circuits = adapt_qaoa_zero_init_state.construct_circuit(parameter = init_pt, operator = qubit_op)
        custom_circuits = adapt_qaoa.construct_circuit(parameter = init_pt, operator = qubit_op)

        self.assertEqual(len(zero_circuits), len(custom_circuits))

        z_length = len(zero_circuits.data)
        c_length = len(custom_circuits.data)

        self.assertGreaterEqual(c_length, z_length)
        for i,j in zip(zero_circuits.data, custom_circuits.data[-z_length:]):
            for ii, jj in zip(i,j):
                if isinstance(ii, list):
                    self.assertEqual(ii, jj)
                else:
                    zero_circ_dict, custom_circ_dict = dict(ii.__dict__), dict(jj.__dict__)
                    for k in zero_circ_dict.keys():
                        if k == '_definition':
                            for dataj in range(len(custom_circ_dict[k].data)):
                                if dataj == num_qubits: # check for same cost operator
                                    self.assertEqual(zero_circ_dict[k][0], custom_circ_dict[k].data[dataj])
                                if dataj == num_qubits + 1: # check for different mixer operator
                                    self.assertNotEqual(zero_circ_dict[k][1], custom_circ_dict[k].data[dataj][0])
                        if k != '_definition':  # remaining data should be equal
                            self.assertEqual(zero_circ_dict[k], custom_circ_dict[k])

    def test_qaoa_construct_circuit_update(self):
        """Test updating operators with QAOA construct_circuit"""
        adapt_qaoa = AdaptQAOA(quantum_instance=self.statevector_simulator)
        adapt_qaoa.compute_minimum_eigenvalue(operator = I ^ Z) # compute optimal mixer via adapt algorithm
        res_circ = adapt_qaoa.ansatz     # set resultant ansatz 
        recon = adapt_qaoa.construct_circuit(operator = I ^ Z)    # construct the circuit with no mixer_operator input
        self.assertEqual(res_circ, recon)     # check both ansatz are the same (i.e. optimal mixer was used)

        # modify operator
        ref = adapt_qaoa.construct_circuit(
            parameter = [0, 0], operator = Z ^ I) 
        circ2 = adapt_qaoa.construct_circuit(
            parameter = [0, 0], operator = Z ^ I)
        self.assertEqual(circ2, ref)
        circ3 = adapt_qaoa.construct_circuit(
            parameter = [0, 0], operator = I ^ Z)
        self.assertNotEqual(circ3, ref)
        circ4 = adapt_qaoa.construct_circuit(
            parameter = [0, 0], operator = Z ^ I)
        self.assertEqual(circ4, ref)

        # include custom mixers
        circ5 = adapt_qaoa.construct_circuit(
            parameter = [0, 0], operator = Z ^ I, mixer_operators=X^X)
        self.assertNotEqual(circ5.mixer_operators, ref.mixer_operators)
        self.assertNotEqual(circ5, ref)
        circ5_mixers = circ5.mixer_operators
        circ6 = adapt_qaoa.construct_circuit(operator = Z ^ I, mixer_operators=[X^X, Z^Z])
        self.assertEqual(len(circ6.mixer_operators)-1, len(circ5_mixers))
        circ7 = circ6.assign_parameters([0, 0, 0, 0], inplace=True)
        self.assertNotEqual(circ7, ref)
        self.assertNotEqual(circ7, circ5)
        self.assertNotEqual(circ7, circ6)
        circ8 = adapt_qaoa.construct_circuit(
            parameter = [0, 0, 0, 0], operator = Z ^ I, mixer_operators=[Z^Z, X^X])
        self.assertNotEqual(circ7, circ8)

    def test_custom_mixer_operators(self):
        """Test specifying custom ansatz mixers for minimisation"""
        qubit_op, _ = self._get_operator(W1)
        n = qubit_op.num_qubits
        custom_mixer = I^n
        adapt_qaoa = AdaptQAOA(
            max_reps=P2,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op, custom_mixers=custom_mixer)
        self.assertEqual(adapt_qaoa.ansatz.mixer_operators[0], custom_mixer)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, S1)

    def test_custom_mixer_pool(self):
        """Test specifying custom mixer pool for changing problem size"""
        qubit_op, _ = self._get_operator(W1)
        n = qubit_op.num_qubits
        mixer_pool = [M1, Z^n, X^n]
        adapt_qaoa = AdaptQAOA(
            mixer_pool = mixer_pool,
            max_reps=P2,
            quantum_instance=self.statevector_simulator,
        )
        result = adapt_qaoa.compute_minimum_eigenvalue(operator=qubit_op)
        self.assertEqual(adapt_qaoa.mixer_pool, mixer_pool)
        x = self._sample_most_likely(result.eigenstate)
        graph_solution = self._get_graph_solution(x)
        self.assertIn(graph_solution, S1)
       
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
        return PauliSumOp.from_list(opflow_list), shift

    def _get_graph_solution(self, x: np.ndarray) -> str:
        """Get graph solution from binary string.

        Args:
            x : binary string as numpy array.

        Returns:
            a graph solution as string.
        """

        return "".join([str(int(i)) for i in 1 - x])

    def _sample_most_likely(self, state_vector):
        """Compute the most likely binary string from state vector.
        Args:
            state_vector (numpy.ndarray or dict): state vector or counts.

        Returns:
            numpy.ndarray: binary string as numpy.ndarray of ints.
        """
        n = int(np.log2(state_vector.shape[0]))
        k = np.argmax(np.abs(state_vector))
        x = np.zeros(n)
        for i in range(n):
            x[i] = k % 2
            k >>= 1
        return x


if __name__ == "__main__":
    unittest.main()


    """ To fix:
        - test_adapt_qaoa_qc_mixer:
            - Doesn't like circuits as mixers
        - test_adapt_qaoa_qc_mixer_many_parameters
    """
