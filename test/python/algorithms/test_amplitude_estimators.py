# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the quantum amplitude estimation algorithm."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt, idata, data, unpack
from qiskit import QuantumRegister, QuantumCircuit, BasicAer
from qiskit.circuit.library import QFT, GroverOperator
from qiskit.utils import QuantumInstance
from qiskit.algorithms import (
    AmplitudeEstimation,
    MaximumLikelihoodAmplitudeEstimation,
    IterativeAmplitudeEstimation,
    FasterAmplitudeEstimation,
    EstimationProblem,
)
from qiskit.quantum_info import Operator, Statevector


class BernoulliStateIn(QuantumCircuit):
    """A circuit preparing sqrt(1 - p)|0> + sqrt(p)|1>."""

    def __init__(self, probability):
        super().__init__(1)
        angle = 2 * np.arcsin(np.sqrt(probability))
        self.ry(angle, 0)


class BernoulliGrover(QuantumCircuit):
    """The Grover operator corresponding to the Bernoulli A operator."""

    def __init__(self, probability):
        super().__init__(1, global_phase=np.pi)
        self.angle = 2 * np.arcsin(np.sqrt(probability))
        self.ry(2 * self.angle, 0)

    def power(self, power, matrix_power=False):
        if matrix_power:
            return super().power(power, True)

        powered = QuantumCircuit(1)
        powered.ry(power * 2 * self.angle, 0)
        return powered


class SineIntegral(QuantumCircuit):
    r"""Construct the A operator to approximate the integral

        \int_0^1 \sin^2(x) d x

    with a specified number of qubits.
    """

    def __init__(self, num_qubits):
        qr_state = QuantumRegister(num_qubits, "state")
        qr_objective = QuantumRegister(1, "obj")
        super().__init__(qr_state, qr_objective)

        # prepare 1/sqrt{2^n} sum_x |x>_n
        self.h(qr_state)

        # apply the sine/cosine term
        self.ry(2 * 1 / 2 / 2 ** num_qubits, qr_objective[0])
        for i, qubit in enumerate(qr_state):
            self.cry(2 * 2 ** i / 2 ** num_qubits, qubit, qr_objective[0])


@ddt
class TestBernoulli(QiskitAlgorithmsTestCase):
    """Tests based on the Bernoulli A operator.

    This class tests
        * the estimation result
        * the constructed circuits
    """

    def setUp(self):
        super().setUp()

        self._statevector = QuantumInstance(
            backend=BasicAer.get_backend("statevector_simulator"),
            seed_simulator=2,
            seed_transpiler=2,
        )
        self._unitary = QuantumInstance(
            backend=BasicAer.get_backend("unitary_simulator"),
            shots=1,
            seed_simulator=42,
            seed_transpiler=91,
        )

        def qasm(shots=100):
            return QuantumInstance(
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=shots,
                seed_simulator=2,
                seed_transpiler=2,
            )

        self._qasm = qasm

    @idata(
        [
            [0.2, AmplitudeEstimation(2), {"estimation": 0.5, "mle": 0.2}],
            [0.49, AmplitudeEstimation(3), {"estimation": 0.5, "mle": 0.49}],
            [0.2, MaximumLikelihoodAmplitudeEstimation([0, 1, 2]), {"estimation": 0.2}],
            [0.49, MaximumLikelihoodAmplitudeEstimation(3), {"estimation": 0.49}],
            [0.2, IterativeAmplitudeEstimation(0.1, 0.1), {"estimation": 0.2}],
            [0.49, IterativeAmplitudeEstimation(0.001, 0.01), {"estimation": 0.49}],
            [0.2, FasterAmplitudeEstimation(0.1, 3, rescale=False), {"estimation": 0.2}],
            [0.12, FasterAmplitudeEstimation(0.1, 2, rescale=False), {"estimation": 0.12}],
        ]
    )
    @unpack
    def test_statevector(self, prob, qae, expect):
        """statevector test"""
        qae.quantum_instance = self._statevector
        problem = EstimationProblem(BernoulliStateIn(prob), 0, BernoulliGrover(prob))

        result = qae.estimate(problem)
        self.assertGreaterEqual(self._statevector.time_taken, 0.0)
        self._statevector.reset_execution_results()
        for key, value in expect.items():
            self.assertAlmostEqual(
                value, getattr(result, key), places=3, msg=f"estimate `{key}` failed"
            )

    @idata(
        [
            [0.2, 100, AmplitudeEstimation(4), {"estimation": 0.14644, "mle": 0.193888}],
            [0.0, 1000, AmplitudeEstimation(2), {"estimation": 0.0, "mle": 0.0}],
            [
                0.2,
                100,
                MaximumLikelihoodAmplitudeEstimation([0, 1, 2, 4, 8]),
                {"estimation": 0.199606},
            ],
            [0.8, 10, IterativeAmplitudeEstimation(0.1, 0.05), {"estimation": 0.811711}],
            [0.2, 1000, FasterAmplitudeEstimation(0.1, 3, rescale=False), {"estimation": 0.198640}],
            [
                0.12,
                100,
                FasterAmplitudeEstimation(0.01, 3, rescale=False),
                {"estimation": 0.119037},
            ],
        ]
    )
    @unpack
    def test_qasm(self, prob, shots, qae, expect):
        """qasm test"""
        qae.quantum_instance = self._qasm(shots)
        problem = EstimationProblem(BernoulliStateIn(prob), [0], BernoulliGrover(prob))

        result = qae.estimate(problem)
        for key, value in expect.items():
            self.assertAlmostEqual(
                value, getattr(result, key), places=3, msg=f"estimate `{key}` failed"
            )

    @data(True, False)
    def test_qae_circuit(self, efficient_circuit):
        """Test circuits resulting from canonical amplitude estimation.

        Build the circuit manually and from the algorithm and compare the resulting unitaries.
        """
        prob = 0.5

        problem = EstimationProblem(BernoulliStateIn(prob), objective_qubits=[0])
        for m in [2, 5]:
            qae = AmplitudeEstimation(m)
            angle = 2 * np.arcsin(np.sqrt(prob))

            # manually set up the inefficient AE circuit
            qr_eval = QuantumRegister(m, "a")
            qr_objective = QuantumRegister(1, "q")
            circuit = QuantumCircuit(qr_eval, qr_objective)

            # initial Hadamard gates
            for i in range(m):
                circuit.h(qr_eval[i])

            # A operator
            circuit.ry(angle, qr_objective)

            if efficient_circuit:
                qae.grover_operator = BernoulliGrover(prob)
                for power in range(m):
                    circuit.cry(2 * 2 ** power * angle, qr_eval[power], qr_objective[0])
            else:
                oracle = QuantumCircuit(1)
                oracle.z(0)

                state_preparation = QuantumCircuit(1)
                state_preparation.ry(angle, 0)
                grover_op = GroverOperator(oracle, state_preparation)
                for power in range(m):
                    circuit.compose(
                        grover_op.power(2 ** power).control(),
                        qubits=[qr_eval[power], qr_objective[0]],
                        inplace=True,
                    )

            # fourier transform
            iqft = QFT(m, do_swaps=False).inverse().reverse_bits()
            circuit.append(iqft.to_instruction(), qr_eval)

            actual_circuit = qae.construct_circuit(problem, measurement=False)

            self.assertEqual(Operator(circuit), Operator(actual_circuit))

    @data(True, False)
    def test_iqae_circuits(self, efficient_circuit):
        """Test circuits resulting from iterative amplitude estimation.

        Build the circuit manually and from the algorithm and compare the resulting unitaries.
        """
        prob = 0.5
        problem = EstimationProblem(BernoulliStateIn(prob), objective_qubits=[0])

        for k in [2, 5]:
            qae = IterativeAmplitudeEstimation(0.01, 0.05)
            angle = 2 * np.arcsin(np.sqrt(prob))

            # manually set up the inefficient AE circuit
            q_objective = QuantumRegister(1, "q")
            circuit = QuantumCircuit(q_objective)

            # A operator
            circuit.ry(angle, q_objective)

            if efficient_circuit:
                qae.grover_operator = BernoulliGrover(prob)
                circuit.ry(2 * k * angle, q_objective[0])

            else:
                oracle = QuantumCircuit(1)
                oracle.z(0)
                state_preparation = QuantumCircuit(1)
                state_preparation.ry(angle, 0)
                grover_op = GroverOperator(oracle, state_preparation)
                for _ in range(k):
                    circuit.compose(grover_op, inplace=True)

            actual_circuit = qae.construct_circuit(problem, k, measurement=False)
            self.assertEqual(Operator(circuit), Operator(actual_circuit))

    @data(True, False)
    def test_mlae_circuits(self, efficient_circuit):
        """Test the circuits constructed for MLAE"""
        prob = 0.5
        problem = EstimationProblem(BernoulliStateIn(prob), objective_qubits=[0])

        for k in [2, 5]:
            qae = MaximumLikelihoodAmplitudeEstimation(k)
            angle = 2 * np.arcsin(np.sqrt(prob))

            # compute all the circuits used for MLAE
            circuits = []

            # 0th power
            q_objective = QuantumRegister(1, "q")
            circuit = QuantumCircuit(q_objective)
            circuit.ry(angle, q_objective)
            circuits += [circuit]

            # powers of 2
            for power in range(k):
                q_objective = QuantumRegister(1, "q")
                circuit = QuantumCircuit(q_objective)

                # A operator
                circuit.ry(angle, q_objective)

                # Q^(2^j) operator
                if efficient_circuit:
                    qae.grover_operator = BernoulliGrover(prob)
                    circuit.ry(2 * 2 ** power * angle, q_objective[0])

                else:
                    oracle = QuantumCircuit(1)
                    oracle.z(0)
                    state_preparation = QuantumCircuit(1)
                    state_preparation.ry(angle, 0)
                    grover_op = GroverOperator(oracle, state_preparation)
                    for _ in range(2 ** power):
                        circuit.compose(grover_op, inplace=True)
                circuits += [circuit]

            actual_circuits = qae.construct_circuits(problem, measurement=False)

            for actual, expected in zip(actual_circuits, circuits):
                self.assertEqual(Operator(actual), Operator(expected))


@ddt
class TestSineIntegral(QiskitAlgorithmsTestCase):
    """Tests based on the A operator to integrate sin^2(x).

    This class tests
        * the estimation result
        * the confidence intervals
    """

    def setUp(self):
        super().setUp()

        self._statevector = QuantumInstance(
            backend=BasicAer.get_backend("statevector_simulator"),
            seed_simulator=123,
            seed_transpiler=41,
        )

        def qasm(shots=100):
            return QuantumInstance(
                backend=BasicAer.get_backend("qasm_simulator"),
                shots=shots,
                seed_simulator=7192,
                seed_transpiler=90000,
            )

        self._qasm = qasm

    @idata(
        [
            [2, AmplitudeEstimation(2), {"estimation": 0.5, "mle": 0.270290}],
            [4, MaximumLikelihoodAmplitudeEstimation(4), {"estimation": 0.272675}],
            [3, IterativeAmplitudeEstimation(0.1, 0.1), {"estimation": 0.272082}],
            [3, FasterAmplitudeEstimation(0.01, 1), {"estimation": 0.272082}],
        ]
    )
    @unpack
    def test_statevector(self, n, qae, expect):
        """Statevector end-to-end test"""
        # construct factories for A and Q
        # qae.state_preparation = SineIntegral(n)
        qae.quantum_instance = self._statevector
        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])

        # result = qae.run(self._statevector)
        result = qae.estimate(estimation_problem)
        self.assertGreaterEqual(self._statevector.time_taken, 0.0)
        self._statevector.reset_execution_results()
        for key, value in expect.items():
            self.assertAlmostEqual(
                value, getattr(result, key), places=3, msg=f"estimate `{key}` failed"
            )

    @idata(
        [
            [4, 10, AmplitudeEstimation(2), {"estimation": 0.5, "mle": 0.333333}],
            [3, 10, MaximumLikelihoodAmplitudeEstimation(2), {"estimation": 0.256878}],
            [3, 1000, IterativeAmplitudeEstimation(0.01, 0.01), {"estimation": 0.271790}],
            [3, 1000, FasterAmplitudeEstimation(0.1, 4), {"estimation": 0.274168}],
        ]
    )
    @unpack
    def test_qasm(self, n, shots, qae, expect):
        """QASM simulator end-to-end test."""
        # construct factories for A and Q
        qae.quantum_instance = self._qasm(shots)
        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])

        result = qae.estimate(estimation_problem)
        for key, value in expect.items():
            self.assertAlmostEqual(
                value, getattr(result, key), places=3, msg=f"estimate `{key}` failed"
            )

    @idata(
        [
            [
                AmplitudeEstimation(3),
                "mle",
                {
                    "likelihood_ratio": (0.2494734, 0.3003771),
                    "fisher": (0.2486176, 0.2999286),
                    "observed_fisher": (0.2484562, 0.3000900),
                },
            ],
            [
                MaximumLikelihoodAmplitudeEstimation(3),
                "estimation",
                {
                    "likelihood_ratio": (0.2598794, 0.2798536),
                    "fisher": (0.2584889, 0.2797018),
                    "observed_fisher": (0.2659279, 0.2722627),
                },
            ],
        ]
    )
    @unpack
    def test_confidence_intervals(self, qae, key, expect):
        """End-to-end test for all confidence intervals."""
        n = 3
        qae.quantum_instance = self._statevector
        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])

        # statevector simulator
        result = qae.estimate(estimation_problem)
        self.assertGreater(self._statevector.time_taken, 0.0)
        self._statevector.reset_execution_results()
        methods = ["lr", "fi", "oi"]  # short for likelihood_ratio, fisher, observed_fisher
        alphas = [0.1, 0.00001, 0.9]  # alpha shouldn't matter in statevector
        for alpha, method in zip(alphas, methods):
            confint = qae.compute_confidence_interval(result, alpha, method)
            # confidence interval based on statevector should be empty, as we are sure of the result
            self.assertAlmostEqual(confint[1] - confint[0], 0.0)
            self.assertAlmostEqual(confint[0], getattr(result, key))

        # qasm simulator
        shots = 100
        alpha = 0.01
        qae.quantum_instance = self._qasm(shots)
        result = qae.estimate(estimation_problem)
        for method, expected_confint in expect.items():
            confint = qae.compute_confidence_interval(result, alpha, method)
            np.testing.assert_array_almost_equal(confint, expected_confint)
            self.assertTrue(confint[0] <= getattr(result, key) <= confint[1])

    def test_iqae_confidence_intervals(self):
        """End-to-end test for the IQAE confidence interval."""
        n = 3
        qae = IterativeAmplitudeEstimation(0.1, 0.01, quantum_instance=self._statevector)
        expected_confint = (0.1984050, 0.3511015)
        estimation_problem = EstimationProblem(SineIntegral(n), objective_qubits=[n])

        # statevector simulator
        result = qae.estimate(estimation_problem)
        self.assertGreaterEqual(self._statevector.time_taken, 0.0)
        self._statevector.reset_execution_results()
        confint = result.confidence_interval
        # confidence interval based on statevector should be empty, as we are sure of the result
        self.assertAlmostEqual(confint[1] - confint[0], 0.0)
        self.assertAlmostEqual(confint[0], result.estimation)

        # qasm simulator
        shots = 100
        qae.quantum_instance = self._qasm(shots)
        result = qae.estimate(estimation_problem)
        confint = result.confidence_interval
        np.testing.assert_array_almost_equal(confint, expected_confint)
        self.assertTrue(confint[0] <= result.estimation <= confint[1])


@ddt
class TestFasterAmplitudeEstimation(QiskitAlgorithmsTestCase):
    """Specific tests for Faster AE."""

    def test_rescaling(self):
        """Test the rescaling."""
        amplitude = 0.8
        scaling = 0.25
        circuit = QuantumCircuit(1)
        circuit.ry(2 * np.arcsin(amplitude), 0)
        problem = EstimationProblem(circuit, objective_qubits=[0])

        rescaled = problem.rescale(scaling)
        rescaled_amplitude = Statevector.from_instruction(rescaled.state_preparation).data[3]

        self.assertAlmostEqual(scaling * amplitude, rescaled_amplitude)

    def test_run_without_rescaling(self):
        """Run Faster AE without rescaling if the amplitude is in [0, 1/4]."""
        # construct estimation problem
        prob = 0.11
        a_op = QuantumCircuit(1)
        a_op.ry(2 * np.arcsin(np.sqrt(prob)), 0)
        problem = EstimationProblem(a_op, objective_qubits=[0])

        # construct algo without rescaling
        backend = BasicAer.get_backend("statevector_simulator")
        fae = FasterAmplitudeEstimation(0.1, 1, rescale=False, quantum_instance=backend)

        # run the algo
        result = fae.estimate(problem)

        # assert the result is correct
        self.assertAlmostEqual(result.estimation, prob)

        # assert no rescaling was used
        theta = np.mean(result.theta_intervals[-1])
        value_without_scaling = np.sin(theta) ** 2
        self.assertAlmostEqual(result.estimation, value_without_scaling)

    def test_rescaling_with_custom_grover_raises(self):
        """Test that the rescaling option fails if a custom Grover operator is used."""
        prob = 0.8
        a_op = BernoulliStateIn(prob)
        q_op = BernoulliGrover(prob)
        problem = EstimationProblem(a_op, objective_qubits=[0], grover_operator=q_op)

        # construct algo without rescaling
        backend = BasicAer.get_backend("statevector_simulator")
        fae = FasterAmplitudeEstimation(0.1, 1, quantum_instance=backend)

        # run the algo
        with self.assertWarns(Warning):
            _ = fae.estimate(problem)

    @data(("statevector_simulator", 0.2), ("qasm_simulator", 0.199440))
    @unpack
    def test_good_state(self, backend_str, expect):
        """Test with a good state function."""

        def is_good_state(bitstr):
            return bitstr[1] == "1"

        # construct the estimation problem where the second qubit is ignored
        a_op = QuantumCircuit(2)
        a_op.ry(2 * np.arcsin(np.sqrt(0.2)), 0)

        # oracle only affects first qubit
        oracle = QuantumCircuit(2)
        oracle.z(0)

        # reflect only on first qubit
        q_op = GroverOperator(oracle, a_op, reflection_qubits=[0])

        # but we measure both qubits (hence both are objective qubits)
        problem = EstimationProblem(
            a_op, objective_qubits=[0, 1], grover_operator=q_op, is_good_state=is_good_state
        )

        # construct algo
        backend = QuantumInstance(
            BasicAer.get_backend(backend_str), seed_simulator=2, seed_transpiler=2
        )
        # cannot use rescaling with a custom grover operator
        fae = FasterAmplitudeEstimation(0.01, 5, rescale=False, quantum_instance=backend)

        # run the algo
        result = fae.estimate(problem)

        # assert the result is correct
        self.assertAlmostEqual(result.estimation, expect, places=5)


if __name__ == "__main__":
    unittest.main()
