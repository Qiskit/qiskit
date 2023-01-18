# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Grover's algorithm."""

import itertools
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import data, ddt, idata, unpack

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.circuit.library import GroverOperator, PhaseOracle
from qiskit.primitives import Sampler
from qiskit.quantum_info import Operator, Statevector
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.utils.optionals import HAS_TWEEDLEDUM


@ddt
class TestAmplificationProblem(QiskitAlgorithmsTestCase):
    """Test the amplification problem."""

    def setUp(self):
        super().setUp()
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        self._expected_grover_op = GroverOperator(oracle=oracle)

    @data("oracle_only", "oracle_and_stateprep")
    def test_groverop_getter(self, kind):
        """Test the default construction of the Grover operator."""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)

        if kind == "oracle_only":
            problem = AmplificationProblem(oracle, is_good_state=["11"])
            expected = GroverOperator(oracle)
        else:
            stateprep = QuantumCircuit(2)
            stateprep.ry(0.2, [0, 1])
            problem = AmplificationProblem(
                oracle, state_preparation=stateprep, is_good_state=["11"]
            )
            expected = GroverOperator(oracle, stateprep)

        self.assertEqual(Operator(expected), Operator(problem.grover_operator))

    @data("list_str", "list_int", "statevector", "callable")
    def test_is_good_state(self, kind):
        """Test is_good_state works on different input types."""
        if kind == "list_str":
            is_good_state = ["01", "11"]
        elif kind == "list_int":
            is_good_state = [1]  # means bitstr[1] == '1'
        elif kind == "statevector":
            is_good_state = Statevector(np.array([0, 1, 0, 1]) / np.sqrt(2))
        else:

            def is_good_state(bitstr):
                # same as ``bitstr in ['01', '11']``
                return bitstr[1] == "1"

        possible_states = [
            "".join(list(map(str, item))) for item in itertools.product([0, 1], repeat=2)
        ]

        oracle = QuantumCircuit(2)
        problem = AmplificationProblem(oracle, is_good_state=is_good_state)

        expected = [state in ["01", "11"] for state in possible_states]
        # pylint: disable=not-callable
        actual = [problem.is_good_state(state) for state in possible_states]

        self.assertListEqual(expected, actual)


@ddt
class TestGrover(QiskitAlgorithmsTestCase):
    """Test for the functionality of Grover"""

    def setUp(self):
        super().setUp()
        self.statevector = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"), seed_simulator=12, seed_transpiler=32
        )
        self.qasm = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"), seed_simulator=12, seed_transpiler=32
        )
        self._sampler = Sampler()
        self._sampler_with_shots = Sampler(options={"shots": 1024, "seed": 123})
        algorithm_globals.random_seed = 123

    @unittest.skipUnless(HAS_TWEEDLEDUM, "tweedledum required for this test")
    @data("ideal", "shots", False)
    def test_implicit_phase_oracle_is_good_state(self, use_sampler):
        """Test implicit default for is_good_state with PhaseOracle."""
        grover = self._prepare_grover(use_sampler)
        oracle = PhaseOracle("x & y")
        problem = AmplificationProblem(oracle)
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "11")

    @idata(itertools.product(["ideal", "shots", False], [[1, 2, 3], None, 2]))
    @unpack
    def test_iterations_with_good_state(self, use_sampler, iterations):
        """Test the algorithm with different iteration types and with good state"""
        grover = self._prepare_grover(use_sampler, iterations)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    @data("ideal", "shots", False)
    def test_fixed_iterations_without_good_state(self, use_sampler):
        """Test the algorithm with iterations as an int and without good state"""
        grover = self._prepare_grover(use_sampler, iterations=2)
        problem = AmplificationProblem(Statevector.from_label("111"))
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    @idata(itertools.product(["ideal", "shots", False], [[1, 2, 3], None]))
    @unpack
    def test_iterations_without_good_state(self, use_sampler, iterations):
        """Test the correct error is thrown for none/list of iterations and without good state"""
        grover = self._prepare_grover(use_sampler, iterations=iterations)
        problem = AmplificationProblem(Statevector.from_label("111"))
        with self.assertRaisesRegex(
            TypeError, "An is_good_state function is required with the provided oracle"
        ):
            grover.amplify(problem)

    @data("ideal", "shots", False)
    def test_iterator(self, use_sampler):
        """Test running the algorithm on an iterator."""

        # step-function iterator
        def iterator():
            wait, value, count = 3, 1, 0
            while True:
                yield value
                count += 1
                if count % wait == 0:
                    value += 1

        grover = self._prepare_grover(use_sampler, iterations=iterator())
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    @data("ideal", "shots", False)
    def test_growth_rate(self, use_sampler):
        """Test running the algorithm on a growth rate"""
        grover = self._prepare_grover(use_sampler, growth_rate=8 / 7)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(result.top_measurement, "111")

    @data("ideal", "shots", False)
    def test_max_num_iterations(self, use_sampler):
        """Test the iteration stops when the maximum number of iterations is reached."""

        def zero():
            while True:
                yield 0

        grover = self._prepare_grover(use_sampler, iterations=zero())
        n = 5
        problem = AmplificationProblem(Statevector.from_label("1" * n), is_good_state=["1" * n])
        result = grover.amplify(problem)
        self.assertEqual(len(result.iterations), 2**n)

    @data("ideal", "shots", False)
    def test_max_power(self, use_sampler):
        """Test the iteration stops when the maximum power is reached."""
        lam = 10.0
        grover = self._prepare_grover(use_sampler, growth_rate=lam)
        problem = AmplificationProblem(Statevector.from_label("111"), is_good_state=["111"])
        result = grover.amplify(problem)
        self.assertEqual(len(result.iterations), 0)

    @data("ideal", "shots", False)
    def test_run_circuit_oracle(self, use_sampler):
        """Test execution with a quantum circuit oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        grover = self._prepare_grover(use_sampler)
        result = grover.amplify(problem)
        self.assertIn(result.top_measurement, ["11"])

    @data("ideal", "shots", False)
    def test_run_state_vector_oracle(self, use_sampler):
        """Test execution with a state vector oracle"""
        mark_state = Statevector.from_label("11")
        problem = AmplificationProblem(mark_state, is_good_state=["11"])
        grover = self._prepare_grover(use_sampler)
        result = grover.amplify(problem)
        self.assertIn(result.top_measurement, ["11"])

    @data("ideal", "shots", False)
    def test_run_custom_grover_operator(self, use_sampler):
        """Test execution with a grover operator oracle"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        grover_op = GroverOperator(oracle)
        problem = AmplificationProblem(
            oracle=oracle, grover_operator=grover_op, is_good_state=["11"]
        )
        grover = self._prepare_grover(use_sampler)
        ret = grover.amplify(problem)
        self.assertIn(ret.top_measurement, ["11"])

    def test_optimal_num_iterations(self):
        """Test optimal_num_iterations"""
        num_qubits = 7
        for num_solutions in range(1, 2**num_qubits):
            amplitude = np.sqrt(num_solutions / 2**num_qubits)
            expected = round(np.arccos(amplitude) / (2 * np.arcsin(amplitude)))
            actual = Grover.optimal_num_iterations(num_solutions, num_qubits)
        self.assertEqual(actual, expected)

    def test_construct_circuit(self):
        """Test construct_circuit"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        grover = Grover()
        constructed = grover.construct_circuit(problem, 2, measurement=False)

        grover_op = GroverOperator(oracle)
        expected = QuantumCircuit(2)
        expected.h([0, 1])
        expected.compose(grover_op.power(2), inplace=True)

        self.assertTrue(Operator(constructed).equiv(Operator(expected)))

    @data("ideal", "shots", False)
    def test_circuit_result(self, use_sampler):
        """Test circuit_result"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        # is_good_state=['00'] is intentionally selected to obtain a list of results
        problem = AmplificationProblem(oracle, is_good_state=["00"])
        grover = self._prepare_grover(use_sampler, iterations=[1, 2, 3, 4])
        result = grover.amplify(problem)
        if use_sampler:
            for i, dist in enumerate(result.circuit_results):
                keys, values = zip(*sorted(dist.items()))
                self.assertTupleEqual(keys, ("00", "01", "10", "11"))
                if i in (0, 3):
                    np.testing.assert_allclose(values, [0, 0, 0, 1], atol=0.2)
                else:
                    np.testing.assert_allclose(values, [0.25, 0.25, 0.25, 0.25], atol=0.2)
        else:
            expected_results = [
                {"11": 1024},
                {"00": 238, "01": 253, "10": 263, "11": 270},
                {"00": 238, "01": 253, "10": 263, "11": 270},
                {"11": 1024},
            ]
            self.assertEqual(result.circuit_results, expected_results)

    @data("ideal", "shots", False)
    def test_max_probability(self, use_sampler):
        """Test max_probability"""
        oracle = QuantumCircuit(2)
        oracle.cz(0, 1)
        problem = AmplificationProblem(oracle, is_good_state=["11"])
        grover = self._prepare_grover(use_sampler)
        result = grover.amplify(problem)
        self.assertAlmostEqual(result.max_probability, 1.0)

    @unittest.skipUnless(HAS_TWEEDLEDUM, "tweedledum required for this test")
    @data("ideal", "shots", False)
    def test_oracle_evaluation(self, use_sampler):
        """Test oracle_evaluation for PhaseOracle"""
        oracle = PhaseOracle("x1 & x2 & (not x3)")
        problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)
        grover = self._prepare_grover(use_sampler)
        result = grover.amplify(problem)
        self.assertTrue(result.oracle_evaluation)
        self.assertEqual("011", result.top_measurement)

    def test_sampler_setter(self):
        """Test sampler setter"""
        grover = Grover()
        grover.sampler = self._sampler
        self.assertEqual(grover.sampler, self._sampler)

    def _prepare_grover(self, use_sampler, iterations=None, growth_rate=None):
        """Prepare Grover instance for test"""
        if use_sampler == "ideal":
            grover = Grover(sampler=self._sampler, iterations=iterations, growth_rate=growth_rate)
        elif use_sampler == "shots":
            grover = Grover(
                sampler=self._sampler_with_shots, iterations=iterations, growth_rate=growth_rate
            )
        else:
            with self.assertWarns(PendingDeprecationWarning):
                grover = Grover(
                    quantum_instance=self.qasm, iterations=iterations, growth_rate=growth_rate
                )
        return grover


if __name__ == "__main__":
    unittest.main()
