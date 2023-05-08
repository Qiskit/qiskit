# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test VQE"""

import logging
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from test.python.transpiler._dummy_passes import DummyAP

from functools import partial
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from ddt import data, ddt, unpack

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import VQE, AlgorithmError
from qiskit.algorithms.optimizers import (
    CG,
    COBYLA,
    L_BFGS_B,
    P_BFGS,
    QNSPSA,
    SLSQP,
    SPSA,
    TNC,
    OptimizerResult,
)
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import (
    AerPauliExpectation,
    Gradient,
    I,
    MatrixExpectation,
    PauliExpectation,
    PauliSumOp,
    PrimitiveOp,
    TwoQubitReduction,
    X,
    Z,
)
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from qiskit.utils import QuantumInstance, algorithm_globals, optionals

logger = "LocalLogger"


class LogPass(DummyAP):
    """A dummy analysis pass that logs when executed"""

    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self, dag):
        logging.getLogger(logger).info(self.message)


# pylint: disable=invalid-name, unused-argument
def _mock_optimizer(fun, x0, jac=None, bounds=None) -> OptimizerResult:
    """A mock of a callable that can be used as minimizer in the VQE."""
    result = OptimizerResult()
    result.x = np.zeros_like(x0)
    result.fun = fun(result.x)
    result.nit = 0
    return result


@ddt
class TestVQE(QiskitAlgorithmsTestCase):
    """Test VQE"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.h2_energy = -1.85727503

        self.ryrz_wavefunction = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        self.ry_wavefunction = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        with self.assertWarns(DeprecationWarning):
            self.h2_op = (
                -1.052373245772859 * (I ^ I)
                + 0.39793742484318045 * (I ^ Z)
                - 0.39793742484318045 * (Z ^ I)
                - 0.01128010425623538 * (Z ^ Z)
                + 0.18093119978423156 * (X ^ X)
            )
            self.qasm_simulator = QuantumInstance(
                BasicAer.get_backend("qasm_simulator"),
                shots=1024,
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
            )
            self.statevector_simulator = QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                shots=1,
                seed_simulator=self.seed,
                seed_transpiler=self.seed,
            )

    def test_basic_aer_statevector(self):
        """Test the VQE on BasicAer's statevector simulator."""
        wavefunction = self.ryrz_wavefunction
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=L_BFGS_B(),
                quantum_instance=QuantumInstance(
                    BasicAer.get_backend("statevector_simulator"),
                    basis_gates=["u1", "u2", "u3", "cx", "id"],
                    coupling_map=[[0, 1]],
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                ),
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        with self.subTest(msg="test eigenvalue"):
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy)

        with self.subTest(msg="test dimension of optimal point"):
            self.assertEqual(len(result.optimal_point), 16)

        with self.subTest(msg="assert cost_function_evals is set"):
            self.assertIsNotNone(result.cost_function_evals)

        with self.subTest(msg="assert optimizer_time is set"):
            self.assertIsNotNone(result.optimizer_time)

    def test_circuit_input(self):
        """Test running the VQE on a plain QuantumCircuit object."""
        wavefunction = QuantumCircuit(2).compose(EfficientSU2(2))
        optimizer = SLSQP(maxiter=50)

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                quantum_instance=self.statevector_simulator,
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

    @data(
        (MatrixExpectation(), 1),
        (AerPauliExpectation(), 1),
        (PauliExpectation(), 2),
    )
    @unpack
    def test_construct_circuit(self, expectation, num_circuits):
        """Test construct circuits returns QuantumCircuits and the right number of them."""
        try:
            wavefunction = EfficientSU2(2, reps=1)

            with self.assertWarns(DeprecationWarning):
                vqe = VQE(ansatz=wavefunction, expectation=expectation)
                params = [0] * wavefunction.num_parameters
                circuits = vqe.construct_circuit(parameter=params, operator=self.h2_op)

            self.assertEqual(len(circuits), num_circuits)
            for circuit in circuits:
                self.assertIsInstance(circuit, QuantumCircuit)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
            return

    def test_missing_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.h2_op.num_qubits)

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=circuit, quantum_instance=BasicAer.get_backend("statevector_simulator")
            )
            with self.assertRaises(RuntimeError):
                vqe.compute_minimum_eigenvalue(operator=self.h2_op)

    @data(
        (SLSQP(maxiter=50), 5, 4),
        (SPSA(maxiter=150), 2, 2),  # max_evals_grouped=n or =2 if n>2
    )
    @unpack
    def test_max_evals_grouped(self, optimizer, places, max_evals_grouped):
        """VQE Optimizers test"""
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=self.ryrz_wavefunction,
                optimizer=optimizer,
                max_evals_grouped=max_evals_grouped,
                quantum_instance=self.statevector_simulator,
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=places)

    def test_basic_aer_qasm(self):
        """Test the VQE on BasicAer's QASM simulator."""
        optimizer = SPSA(maxiter=300, last_avg=5)
        wavefunction = self.ry_wavefunction

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                max_evals_grouped=1,
                quantum_instance=self.qasm_simulator,
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.86823, places=2)

    def test_qasm_eigenvector_normalized(self):
        """Test VQE with qasm_simulator returns normalized eigenvector."""
        wavefunction = self.ry_wavefunction
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(ansatz=wavefunction, quantum_instance=self.qasm_simulator)
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        amplitudes = list(result.eigenstate.values())
        self.assertAlmostEqual(np.linalg.norm(amplitudes), 1.0, places=4)

    @unittest.skipUnless(optionals.HAS_AER, "Qiskit aer is required to run these tests")
    def test_with_aer_statevector(self):
        """Test VQE with Aer's statevector_simulator."""
        from qiskit_aer import AerSimulator

        backend = AerSimulator(method="statevector")
        wavefunction = self.ry_wavefunction
        optimizer = L_BFGS_B()

        with self.assertWarns(DeprecationWarning):
            quantum_instance = QuantumInstance(
                backend,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                max_evals_grouped=1,
                quantum_instance=quantum_instance,
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)

    @unittest.skipUnless(optionals.HAS_AER, "Qiskit aer is required to run these tests")
    def test_with_aer_qasm(self):
        """Test VQE with Aer's qasm_simulator."""
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        optimizer = SPSA(maxiter=200, last_avg=5)
        wavefunction = self.ry_wavefunction

        with self.assertWarns(DeprecationWarning):
            quantum_instance = QuantumInstance(
                backend,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                expectation=PauliExpectation(),
                quantum_instance=quantum_instance,
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.86305, places=2)

    @unittest.skipUnless(optionals.HAS_AER, "Qiskit aer is required to run these tests")
    def test_with_aer_qasm_snapshot_mode(self):
        """Test the VQE using Aer's qasm_simulator snapshot mode."""
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        optimizer = L_BFGS_B()
        wavefunction = self.ry_wavefunction

        with self.assertWarns(DeprecationWarning):
            quantum_instance = QuantumInstance(
                backend,
                shots=1,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                expectation=AerPauliExpectation(),
                quantum_instance=quantum_instance,
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)

    @unittest.skipUnless(optionals.HAS_AER, "Qiskit aer is required to run these tests")
    @data(
        CG(maxiter=1),
        L_BFGS_B(maxfun=1),
        P_BFGS(maxfun=1, max_processes=0),
        SLSQP(maxiter=1),
        TNC(maxiter=1),
    )
    def test_with_gradient(self, optimizer):
        """Test VQE using Gradient()."""
        from qiskit_aer import AerSimulator

        with self.assertWarns(DeprecationWarning):
            quantum_instance = QuantumInstance(
                backend=AerSimulator(),
                shots=1,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            )
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=self.ry_wavefunction,
                optimizer=optimizer,
                gradient=Gradient(),
                expectation=AerPauliExpectation(),
                quantum_instance=quantum_instance,
                max_evals_grouped=1000,
            )
            vqe.compute_minimum_eigenvalue(operator=self.h2_op)

    def test_with_two_qubit_reduction(self):
        """Test the VQE using TwoQubitReduction."""

        with self.assertWarns(DeprecationWarning):
            qubit_op = PauliSumOp.from_list(
                [
                    ("IIII", -0.8105479805373266),
                    ("IIIZ", 0.17218393261915552),
                    ("IIZZ", -0.22575349222402472),
                    ("IZZI", 0.1721839326191556),
                    ("ZZII", -0.22575349222402466),
                    ("IIZI", 0.1209126326177663),
                    ("IZZZ", 0.16892753870087912),
                    ("IXZX", -0.045232799946057854),
                    ("ZXIX", 0.045232799946057854),
                    ("IXIX", 0.045232799946057854),
                    ("ZXZX", -0.045232799946057854),
                    ("ZZIZ", 0.16614543256382414),
                    ("IZIZ", 0.16614543256382414),
                    ("ZZZZ", 0.17464343068300453),
                    ("ZIZI", 0.1209126326177663),
                ]
            )
            tapered_qubit_op = TwoQubitReduction(num_particles=2).convert(qubit_op)

        for simulator in [self.qasm_simulator, self.statevector_simulator]:
            with self.subTest(f"Test for {simulator}."), self.assertWarns(DeprecationWarning):
                vqe = VQE(
                    self.ry_wavefunction,
                    SPSA(maxiter=300, last_avg=5),
                    quantum_instance=simulator,
                )
                result = vqe.compute_minimum_eigenvalue(tapered_qubit_op)
                energy = -1.868 if simulator == self.qasm_simulator else self.h2_energy
                self.assertAlmostEqual(result.eigenvalue.real, energy, places=2)

    def test_callback(self):
        """Test the callback on VQE."""
        history = {"eval_count": [], "parameters": [], "mean": [], "std": []}

        def store_intermediate_result(eval_count, parameters, mean, std):
            history["eval_count"].append(eval_count)
            history["parameters"].append(parameters)
            history["mean"].append(mean)
            history["std"].append(std)

        optimizer = COBYLA(maxiter=3)
        wavefunction = self.ry_wavefunction

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                optimizer=optimizer,
                callback=store_intermediate_result,
                quantum_instance=self.qasm_simulator,
            )
            vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertTrue(all(isinstance(count, int) for count in history["eval_count"]))
        self.assertTrue(all(isinstance(mean, float) for mean in history["mean"]))
        self.assertTrue(all(isinstance(std, float) for std in history["std"]))
        for params in history["parameters"]:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_reuse(self):
        """Test re-using a VQE algorithm instance."""

        with self.assertWarns(DeprecationWarning):
            vqe = VQE()
        with self.subTest(msg="assert running empty raises AlgorithmError"):
            with self.assertWarns(DeprecationWarning), self.assertRaises(AlgorithmError):
                _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        ansatz = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        vqe.ansatz = ansatz
        with self.subTest(msg="assert missing operator raises AlgorithmError"):
            with self.assertWarns(DeprecationWarning), self.assertRaises(AlgorithmError):
                _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        with self.assertWarns(DeprecationWarning):
            vqe.expectation = MatrixExpectation()
            vqe.quantum_instance = self.statevector_simulator

        with self.subTest(msg="assert VQE works once all info is available"), self.assertWarns(
            DeprecationWarning
        ):
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

        with self.assertWarns(DeprecationWarning):
            operator = PrimitiveOp(
                np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]])
            )

        with self.subTest(msg="assert minimum eigensolver interface works"), self.assertWarns(
            DeprecationWarning
        ):
            result = vqe.compute_minimum_eigenvalue(operator=operator)
            self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_optimizer(self):
        """Test running same VQE twice to re-use optimizer, then switch optimizer"""
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                optimizer=SLSQP(),
                quantum_instance=QuantumInstance(BasicAer.get_backend("statevector_simulator")),
            )

        def run_check():
            with self.assertWarns(DeprecationWarning):
                result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

            self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

        run_check()

        with self.subTest("Optimizer re-use"):
            run_check()

        with self.subTest("Optimizer replace"):
            vqe.optimizer = L_BFGS_B()
            run_check()

    @data(MatrixExpectation(), None)
    def test_backend_change(self, user_expectation):
        """Test that VQE works when backend changes."""

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz"),
                optimizer=SLSQP(maxiter=2),
                expectation=user_expectation,
                quantum_instance=BasicAer.get_backend("statevector_simulator"),
            )
            result0 = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        if user_expectation is not None:
            with self.subTest("User expectation kept."):
                self.assertEqual(vqe.expectation, user_expectation)

        # works also if no expectation is set, since it will be determined automatically
        with self.assertWarns(DeprecationWarning):
            vqe.quantum_instance = BasicAer.get_backend("qasm_simulator")
            result1 = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        if user_expectation is not None:
            with self.subTest("Change backend with user expectation, it is kept."):
                self.assertEqual(vqe.expectation, user_expectation)

        with self.subTest("Check results."):
            self.assertEqual(len(result0.optimal_point), len(result1.optimal_point))

    def test_batch_evaluate_with_qnspsa(self):
        """Test batch evaluating with QNSPSA works."""
        ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz")

        wrapped_backend = BasicAer.get_backend("qasm_simulator")
        inner_backend = BasicAer.get_backend("statevector_simulator")

        callcount = {"count": 0}

        def wrapped_run(circuits, **kwargs):
            kwargs["callcount"]["count"] += 1
            return inner_backend.run(circuits)

        wrapped_backend.run = partial(wrapped_run, callcount=callcount)

        with self.assertWarns(DeprecationWarning):
            fidelity = QNSPSA.get_fidelity(ansatz, backend=wrapped_backend)
        qnspsa = QNSPSA(fidelity, maxiter=5)

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=ansatz,
                optimizer=qnspsa,
                max_evals_grouped=100,
                quantum_instance=wrapped_backend,
            )
            _ = vqe.compute_minimum_eigenvalue(Z ^ Z)

        # 1 calibration + 1 stddev estimation + 1 initial blocking
        # + 5 (1 loss + 1 fidelity + 1 blocking) + 1 return loss + 1 VQE eval
        expected = 1 + 1 + 1 + 5 * 3 + 1 + 1

        self.assertEqual(callcount["count"], expected)

    def test_set_ansatz_to_none(self):
        """Tests that setting the ansatz to None results in the default behavior"""

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=self.ryrz_wavefunction,
                optimizer=L_BFGS_B(),
                quantum_instance=self.statevector_simulator,
            )

        vqe.ansatz = None
        self.assertIsInstance(vqe.ansatz, RealAmplitudes)

    def test_set_optimizer_to_none(self):
        """Tests that setting the optimizer to None results in the default behavior"""

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=self.ryrz_wavefunction,
                optimizer=L_BFGS_B(),
                quantum_instance=self.statevector_simulator,
            )

        vqe.optimizer = None
        self.assertIsInstance(vqe.optimizer, SLSQP)

    def test_optimizer_scipy_callable(self):
        """Test passing a SciPy optimizer directly as callable."""

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                optimizer=partial(scipy_minimize, method="L-BFGS-B", options={"maxiter": 2}),
                quantum_instance=self.statevector_simulator,
            )
            result = vqe.compute_minimum_eigenvalue(Z)

        self.assertEqual(result.cost_function_evals, 20)

    def test_optimizer_callable(self):
        """Test passing a optimizer directly as callable."""
        ansatz = RealAmplitudes(1, reps=1)
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=ansatz,
                optimizer=_mock_optimizer,
                quantum_instance=self.statevector_simulator,
            )
            result = vqe.compute_minimum_eigenvalue(Z)
        self.assertTrue(np.all(result.optimal_point == np.zeros(ansatz.num_parameters)))

    def test_aux_operators_list(self):
        """Test list-based aux_operators."""
        wavefunction = self.ry_wavefunction

        # Start with an empty list
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(ansatz=wavefunction, quantum_instance=self.statevector_simulator)

            # Start with an empty list
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=[])

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Go again with two auxiliary operators
        with self.assertWarns(DeprecationWarning):
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
            aux_ops = [aux_op1, aux_op2]

        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = [*aux_ops, None, 0]
        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=extra_ops)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
        self.assertEqual(len(result.aux_operator_eigenvalues), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0, places=6)
        self.assertEqual(result.aux_operator_eigenvalues[2][0], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[3][0], 0.0)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[2][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[3][1], 0.0)

    def test_aux_operators_dict(self):
        """Test dictionary compatibility of aux_operators"""
        wavefunction = self.ry_wavefunction

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(ansatz=wavefunction, quantum_instance=self.statevector_simulator)

        # Start with an empty dictionary
        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators={})

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Go again with two auxiliary operators
        with self.assertWarns(DeprecationWarning):
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
            aux_ops = {"aux_op1": aux_op1, "aux_op2": aux_op2}

        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = {**aux_ops, "None_operator": None, "zero_operator": 0}
        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=extra_ops)

        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)
        self.assertEqual(len(result.aux_operator_eigenvalues), 3)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][0], 0, places=6)
        self.assertEqual(result.aux_operator_eigenvalues["zero_operator"][0], 0.0)
        self.assertTrue("None_operator" not in result.aux_operator_eigenvalues.keys())
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["zero_operator"][1], 0.0)

    def test_aux_operator_std_dev_pauli(self):
        """Test non-zero standard deviations of aux operators with PauliExpectation."""
        wavefunction = self.ry_wavefunction
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                expectation=PauliExpectation(),
                optimizer=COBYLA(maxiter=0),
                quantum_instance=self.qasm_simulator,
            )

        with self.assertWarns(DeprecationWarning):
            # Go again with two auxiliary operators
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
            aux_ops = [aux_op1, aux_op2]

        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.6796875, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.02534712219145965, places=6)

        # Go again with additional None and zero operators
        aux_ops = [*aux_ops, None, 0]

        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

        self.assertEqual(len(result.aux_operator_eigenvalues), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.57421875, places=6)
        self.assertEqual(result.aux_operator_eigenvalues[2][0], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[3][0], 0.0)
        # # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(
            result.aux_operator_eigenvalues[1][1], 0.026562146577166837, places=6
        )
        self.assertAlmostEqual(result.aux_operator_eigenvalues[2][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[3][1], 0.0)

    @unittest.skipUnless(optionals.HAS_AER, "Qiskit aer is required to run these tests")
    def test_aux_operator_std_dev_aer_pauli(self):
        """Test non-zero standard deviations of aux operators with AerPauliExpectation."""
        from qiskit_aer import AerSimulator

        wavefunction = self.ry_wavefunction
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=wavefunction,
                expectation=AerPauliExpectation(),
                optimizer=COBYLA(maxiter=0),
                quantum_instance=QuantumInstance(
                    backend=AerSimulator(),
                    shots=1,
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                ),
            )
        with self.assertWarns(DeprecationWarning):
            # Go again with two auxiliary operators
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
            aux_ops = [aux_op1, aux_op2]

        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.6698863565455391, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0, places=6)

        # Go again with additional None and zero operators
        aux_ops = [*aux_ops, None, 0]
        with self.assertWarns(DeprecationWarning):
            result = vqe.compute_minimum_eigenvalue(self.h2_op, aux_operators=aux_ops)

        self.assertEqual(len(result.aux_operator_eigenvalues), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.6036400943063891, places=6)
        self.assertEqual(result.aux_operator_eigenvalues[2][0], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[3][0], 0.0)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[2][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[3][1], 0.0)

    def test_2step_transpile(self):
        """Test the two-step transpiler pass."""
        # count how often the pass for parameterized circuits is called
        pre_counter = LogPass("pre_passmanager")
        pre_pass = PassManager(pre_counter)
        config = PassManagerConfig(basis_gates=["u3", "cx"])
        pre_pass += level_1_pass_manager(config)

        # ... and the pass for bound circuits
        bound_counter = LogPass("bound_pass_manager")
        bound_pass = PassManager(bound_counter)

        optimizer = SPSA(maxiter=5, learning_rate=0.01, perturbation=0.01)

        with self.assertWarns(DeprecationWarning):
            quantum_instance = QuantumInstance(
                backend=BasicAer.get_backend("statevector_simulator"),
                basis_gates=["u3", "cx"],
                pass_manager=pre_pass,
                bound_pass_manager=bound_pass,
            )

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(optimizer=optimizer, quantum_instance=quantum_instance)
            with self.assertLogs(logger, level="INFO") as cm:
                _ = vqe.compute_minimum_eigenvalue(Z)

        expected = [
            "pre_passmanager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "bound_pass_manager",
            "pre_passmanager",
            "bound_pass_manager",
        ]
        self.assertEqual([record.message for record in cm.records], expected)

    def test_construct_eigenstate_from_optpoint(self):
        """Test constructing the eigenstate from the optimal point, if the default ansatz is used."""

        # use Hamiltonian yielding more than 11 parameters in the default ansatz
        with self.assertWarns(DeprecationWarning):
            hamiltonian = Z ^ Z ^ Z

        optimizer = SPSA(maxiter=1, learning_rate=0.01, perturbation=0.01)

        with self.assertWarns(DeprecationWarning):
            quantum_instance = QuantumInstance(
                backend=BasicAer.get_backend("statevector_simulator"), basis_gates=["u3", "cx"]
            )

        with self.assertWarns(DeprecationWarning):
            vqe = VQE(optimizer=optimizer, quantum_instance=quantum_instance)
            result = vqe.compute_minimum_eigenvalue(hamiltonian)

        optimal_circuit = vqe.ansatz.bind_parameters(result.optimal_point)
        self.assertTrue(Statevector(result.eigenstate).equiv(optimal_circuit))


if __name__ == "__main__":
    unittest.main()
