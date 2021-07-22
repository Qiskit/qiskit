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

# pylint: disable=no-name-in-module,import-error

""" Test VQE """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import data, ddt, unpack

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import VQE, AlgorithmError
from qiskit.algorithms.optimizers import (
    CG,
    COBYLA,
    L_BFGS_B,
    P_BFGS,
    SLSQP,
    SPSA,
    TNC,
)
from qiskit.circuit.library import EfficientSU2, TwoLocal
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
from qiskit.utils import QuantumInstance, algorithm_globals, has_aer

if has_aer():
    from qiskit import Aer


@ddt
class TestVQE(QiskitAlgorithmsTestCase):
    """Test VQE"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.h2_op = (
            -1.052373245772859 * (I ^ I)
            + 0.39793742484318045 * (I ^ Z)
            - 0.39793742484318045 * (Z ^ I)
            - 0.01128010425623538 * (Z ^ Z)
            + 0.18093119978423156 * (X ^ X)
        )
        self.h2_energy = -1.85727503

        self.ryrz_wavefunction = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        self.ry_wavefunction = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

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
        vqe = VQE(
            ansatz=wavefunction, optimizer=optimizer, quantum_instance=self.statevector_simulator
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
        vqe = VQE(ansatz=circuit, quantum_instance=BasicAer.get_backend("statevector_simulator"))
        with self.assertRaises(RuntimeError):
            vqe.compute_minimum_eigenvalue(operator=self.h2_op)

    @data(
        (SLSQP(maxiter=50), 5, 4),
        (SPSA(maxiter=150), 3, 2),  # max_evals_grouped=n or =2 if n>2
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
                sort_parameters_by_name=True,
            )
        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=places)

    def test_basic_aer_qasm(self):
        """Test the VQE on BasicAer's QASM simulator."""
        optimizer = SPSA(maxiter=300, last_avg=5)
        wavefunction = self.ry_wavefunction

        vqe = VQE(
            ansatz=wavefunction,
            optimizer=optimizer,
            max_evals_grouped=1,
            quantum_instance=self.qasm_simulator,
        )

        # TODO benchmark this later.
        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.86823, places=2)

    def test_qasm_aux_operators_normalized(self):
        """Test VQE with qasm_simulator returns normalized aux_operator eigenvalues."""
        wavefunction = self.ry_wavefunction
        vqe = VQE(ansatz=wavefunction, quantum_instance=self.qasm_simulator)
        _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        opt_params = [
            3.50437328,
            3.87415376,
            0.93684363,
            5.92219622,
            -1.53527887,
            1.87941418,
            -4.5708326,
            0.70187027,
        ]

        vqe._ret.optimal_point = opt_params
        vqe._ret.optimal_parameters = dict(
            zip(sorted(wavefunction.parameters, key=lambda p: p.name), opt_params)
        )

        with self.assertWarns(DeprecationWarning):
            optimal_vector = vqe.get_optimal_vector()

        self.assertAlmostEqual(sum(v ** 2 for v in optimal_vector.values()), 1.0, places=4)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_with_aer_statevector(self):
        """Test VQE with Aer's statevector_simulator."""
        backend = Aer.get_backend("aer_simulator_statevector")
        wavefunction = self.ry_wavefunction
        optimizer = L_BFGS_B()

        quantum_instance = QuantumInstance(
            backend,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        vqe = VQE(
            ansatz=wavefunction,
            optimizer=optimizer,
            max_evals_grouped=1,
            quantum_instance=quantum_instance,
        )

        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_with_aer_qasm(self):
        """Test VQE with Aer's qasm_simulator."""
        backend = Aer.get_backend("aer_simulator")
        optimizer = SPSA(maxiter=200, last_avg=5)
        wavefunction = self.ry_wavefunction

        quantum_instance = QuantumInstance(
            backend,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        vqe = VQE(
            ansatz=wavefunction,
            optimizer=optimizer,
            expectation=PauliExpectation(),
            quantum_instance=quantum_instance,
        )

        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.86305, places=2)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_with_aer_qasm_snapshot_mode(self):
        """Test the VQE using Aer's qasm_simulator snapshot mode."""

        backend = Aer.get_backend("aer_simulator")
        optimizer = L_BFGS_B()
        wavefunction = self.ry_wavefunction

        quantum_instance = QuantumInstance(
            backend,
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        vqe = VQE(
            ansatz=wavefunction,
            optimizer=optimizer,
            expectation=AerPauliExpectation(),
            quantum_instance=quantum_instance,
        )

        result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    @data(
        CG(maxiter=1),
        L_BFGS_B(maxfun=1),
        P_BFGS(maxfun=1, max_processes=0),
        SLSQP(maxiter=1),
        TNC(maxiter=1),
    )
    def test_with_gradient(self, optimizer):
        """Test VQE using Gradient()."""
        quantum_instance = QuantumInstance(
            backend=Aer.get_backend("qasm_simulator"),
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
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
            with self.subTest(f"Test for {simulator}."):
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
        vqe = VQE()
        with self.subTest(msg="assert running empty raises AlgorithmError"):
            with self.assertRaises(AlgorithmError):
                _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        ansatz = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        vqe.ansatz = ansatz
        with self.subTest(msg="assert missing operator raises AlgorithmError"):
            with self.assertRaises(AlgorithmError):
                _ = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        vqe.expectation = MatrixExpectation()
        vqe.quantum_instance = self.statevector_simulator
        with self.subTest(msg="assert VQE works once all info is available"):
            result = vqe.compute_minimum_eigenvalue(operator=self.h2_op)
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

        operator = PrimitiveOp(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]]))

        with self.subTest(msg="assert minimum eigensolver interface works"):
            result = vqe.compute_minimum_eigenvalue(operator=operator)
            self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_optimizer(self):
        """Test running same VQE twice to re-use optimizer, then switch optimizer"""
        vqe = VQE(
            optimizer=SLSQP(),
            quantum_instance=QuantumInstance(BasicAer.get_backend("statevector_simulator")),
        )

        def run_check():
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

        vqe.quantum_instance = BasicAer.get_backend("qasm_simulator")

        # works also if no expectation is set, since it will be determined automatically
        result1 = vqe.compute_minimum_eigenvalue(operator=self.h2_op)

        if user_expectation is not None:
            with self.subTest("Change backend with user expectation, it is kept."):
                self.assertEqual(vqe.expectation, user_expectation)

        with self.subTest("Check results."):
            self.assertEqual(len(result0.optimal_point), len(result1.optimal_point))


if __name__ == "__main__":
    unittest.main()
