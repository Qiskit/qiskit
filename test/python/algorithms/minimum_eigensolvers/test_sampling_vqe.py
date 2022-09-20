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

"""Test the Sampler VQE."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from functools import partial
import numpy as np
from scipy.optimize import minimize as scipy_minimize

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.algorithms import AlgorithmError
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE
from qiskit.algorithms.optimizers import L_BFGS_B, QNSPSA, OptimizerResult, SLSQP
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute


# pylint: disable=invalid-name, unused-argument
def _mock_optimizer(fun, x0, jac=None, bounds=None, inputs=None):
    """A mock of a callable that can be used as minimizer in the VQE.

    If ``inputs`` is given as a dictionary, stores the inputs in that dictionary.
    """
    result = OptimizerResult()
    result.x = np.zeros_like(x0)
    result.fun = fun(result.x)
    result.nit = 0

    if inputs is not None:
        inputs.update({"fun": fun, "x0": x0, "jac": jac, "bounds": bounds})
    return result


class TestSamplerVQE(QiskitAlgorithmsTestCase):
    """Test VQE"""

    def setUp(self):
        super().setUp()
        self.op = PauliSumOp(SparsePauliOp(["ZZ", "IZ", "II"], coeffs=[1, -0.5, 0.12]))
        self.optimal_value = -1.38
        self.optimal_bitstring = "10"

    def test_exact_sampler(self):
        """Test the VQE on BasicAer's statevector simulator."""
        thetas = ParameterVector("th", 4)
        ansatz = QuantumCircuit(2)
        ansatz.rx(thetas[0], 0)
        ansatz.rx(thetas[1], 1)
        ansatz.cz(0, 1)
        ansatz.ry(thetas[2], 0)
        ansatz.ry(thetas[3], 1)

        optimizer = L_BFGS_B()

        # start in maximal superposition
        initial_point = np.zeros(ansatz.num_parameters)
        initial_point[-ansatz.num_qubits :] = np.pi / 2

        vqe = SamplingVQE(Sampler(), ansatz, optimizer, initial_point=initial_point)
        result = vqe.compute_minimum_eigenvalue(operator=self.op)

        with self.subTest(msg="test eigenvalue"):
            self.assertAlmostEqual(result.eigenvalue, self.optimal_value, places=5)

        with self.subTest(msg="test optimal_value"):
            self.assertAlmostEqual(result.optimal_value, self.optimal_value, places=5)

        with self.subTest(msg="test dimension of optimal point"):
            self.assertEqual(len(result.optimal_point), ansatz.num_parameters)

        with self.subTest(msg="assert cost_function_evals is set"):
            self.assertIsNotNone(result.cost_function_evals)

        with self.subTest(msg="assert optimizer_time is set"):
            self.assertIsNotNone(result.optimizer_time)

        with self.subTest(msg="check best measurement"):
            self.assertEqual(result.best_measurement["bitstring"], self.optimal_bitstring)
            self.assertEqual(result.best_measurement["value"], self.optimal_value)

    def test_invalid_initial_point(self):
        """Test the proper error is raised when the initial point has the wrong size."""
        ansatz = RealAmplitudes(2, reps=1)
        initial_point = np.array([1])

        vqe = SamplingVQE(Sampler(), ansatz, SLSQP(), initial_point=initial_point)

        with self.assertRaises(ValueError):
            _ = vqe.compute_minimum_eigenvalue(operator=self.op)

    def test_ansatz_resize(self):
        """Test the ansatz is properly resized if it's a blueprint circuit."""
        ansatz = RealAmplitudes(1, reps=1)
        vqe = SamplingVQE(Sampler(), ansatz, SLSQP())
        result = vqe.compute_minimum_eigenvalue(operator=self.op)
        self.assertAlmostEqual(result.eigenvalue, self.optimal_value, places=5)

    def test_invalid_ansatz_size(self):
        """Test an error is raised if the ansatz has the wrong number of qubits."""
        ansatz = QuantumCircuit(1)
        ansatz.compose(RealAmplitudes(1, reps=2))
        vqe = SamplingVQE(Sampler(), ansatz, SLSQP())

        with self.assertRaises(AlgorithmError):
            _ = vqe.compute_minimum_eigenvalue(operator=self.op)

    def test_missing_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.op.num_qubits)
        vqe = SamplingVQE(Sampler(), circuit, SLSQP())
        with self.assertRaises(RuntimeError):
            vqe.compute_minimum_eigenvalue(operator=self.op)

    def test_batch_evaluate_slsqp(self):
        """Test batching with SLSQP (as representative of SciPyOptimizer)."""
        optimizer = SLSQP(max_evals_grouped=10)
        vqe = SamplingVQE(Sampler(), RealAmplitudes(), optimizer)
        result = vqe.compute_minimum_eigenvalue(operator=self.op)
        self.assertAlmostEqual(result.eigenvalue, self.optimal_value, places=5)

    def test_batch_evaluate_with_qnspsa(self):
        """Test batch evaluating with QNSPSA works."""
        ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz")

        wrapped_sampler = Sampler()
        inner_sampler = Sampler()

        callcount = {"count": 0}

        def wrapped_run(*args, **kwargs):
            kwargs["callcount"]["count"] += 1
            return inner_sampler.run(*args, **kwargs)

        wrapped_sampler.run = partial(wrapped_run, callcount=callcount)

        fidelity = ComputeUncompute(wrapped_sampler)

        def fidelity_callable(left, right):
            batchsize = np.asarray(left).shape[0]
            job = fidelity.run(batchsize * [ansatz], batchsize * [ansatz], left, right)
            return job.result().fidelities

        qnspsa = QNSPSA(fidelity_callable, maxiter=5)
        qnspsa.set_max_evals_grouped(100)

        vqe = SamplingVQE(wrapped_sampler, ansatz, qnspsa)
        _ = vqe.compute_minimum_eigenvalue(Pauli("ZZ"))

        # 1 calibration + 1 stddev estimation + 1 initial blocking
        # + 5 (1 loss + 1 fidelity + 1 blocking) + 1 return loss + 1 VQE eval
        expected = 1 + 1 + 1 + 5 * 3 + 1 + 1

        self.assertEqual(callcount["count"], expected)

    def test_optimizer_scipy_callable(self):
        """Test passing a SciPy optimizer directly as callable."""
        vqe = SamplingVQE(
            Sampler(),
            RealAmplitudes(),
            partial(scipy_minimize, method="COBYLA", options={"maxiter": 2}),
        )
        result = vqe.compute_minimum_eigenvalue(Pauli("Z"))
        self.assertEqual(result.cost_function_evals, 2)

    def test_optimizer_callable(self):
        """Test passing a optimizer directly as callable."""
        ansatz = RealAmplitudes(1, reps=1)
        vqe = SamplingVQE(Sampler(), ansatz, _mock_optimizer)
        result = vqe.compute_minimum_eigenvalue(Pauli("Z"))
        self.assertTrue(np.all(result.optimal_point == np.zeros(ansatz.num_parameters)))

    def test_auxops(self):
        """Test passing auxiliary operators."""
        ansatz = RealAmplitudes(2, reps=1)
        vqe = SamplingVQE(Sampler(), ansatz, SLSQP())

        as_list = [Pauli("ZZ"), Pauli("II")]
        with self.subTest(auxops=as_list):
            result = vqe.compute_minimum_eigenvalue(self.op, aux_operators=as_list)
            self.assertIsInstance(result.aux_operator_values, list)
            self.assertEqual(len(result.aux_operator_values), 2)
            self.assertAlmostEqual(result.aux_operator_values[0], -1 + 0j, places=5)
            self.assertAlmostEqual(result.aux_operator_values[1], 1 + 0j, places=5)

        as_dict = {"magnetization": SparsePauliOp(["ZI", "IZ"])}
        with self.subTest(auxops=as_dict):
            result = vqe.compute_minimum_eigenvalue(self.op, aux_operators=as_dict)
            self.assertIsInstance(result.aux_operator_values, dict)
            self.assertEqual(len(result.aux_operator_values.keys()), 1)
            self.assertAlmostEqual(result.aux_operator_values["magnetization"], 0j, places=5)

    def test_nondiag_observable_raises(self):
        """Test passing a non-diagonal observable raises an error."""
        vqe = SamplingVQE(Sampler(), RealAmplitudes(), SLSQP())

        with self.assertRaises(ValueError):
            _ = vqe.compute_minimum_eigenvalue(Pauli("X"))


if __name__ == "__main__":
    unittest.main()
