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

""" Test VQD """

import logging
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from functools import partial
import numpy as np
from ddt import data, ddt, unpack

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import VQD, AlgorithmError
from qiskit.algorithms.optimizers import (
    CG,
    COBYLA,
    L_BFGS_B,
    P_BFGS,
    QNSPSA,
    SLSQP,
    SPSA,
    TNC,
)
from qiskit.circuit.library import EfficientSU2, RealAmplitudes, TwoLocal
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import (
    AerPauliExpectation,
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
from qiskit.utils import QuantumInstance, algorithm_globals, has_aer
from ..transpiler._dummy_passes import DummyAP

if has_aer():
    from qiskit import Aer

logger = "LocalLogger"


class LogPass(DummyAP):
    """A dummy analysis pass that logs when executed"""

    def __init__(self, message):
        super().__init__()
        self.message = message

    def run(self, dag):
        logging.getLogger(logger).info(self.message)


@ddt
class TestVQD(QiskitAlgorithmsTestCase):
    """Test VQD"""

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
        self.h2_energy_excited = [-1.85727503, -1.24458455]
        
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
        """Test the VQD on BasicAer's statevector simulator."""
        wavefunction = self.ryrz_wavefunction
        vqd = VQD(
            k=2,
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

        result = vqd.compute_eigenvalues(operator=self.h2_op)

        with self.subTest(msg="test eigenvalue"):
            np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 2)

        with self.subTest(msg="test dimension of optimal point"):
            self.assertEqual(len(result.optimal_point[-1]), 16)

        with self.subTest(msg="assert cost_function_evals is set"):
            self.assertIsNotNone(result.cost_function_evals)

        with self.subTest(msg="assert optimizer_time is set"):
            self.assertIsNotNone(result.optimizer_time)

    def test_mismatching_num_qubits(self):
        """Ensuring circuit and operator mismatch is caught"""
        wavefunction = QuantumCircuit(1)
        optimizer = SLSQP(maxiter=50)
        vqd = VQD(
            k=1,ansatz=wavefunction, optimizer=optimizer, quantum_instance=self.statevector_simulator
        )
        with self.assertRaises(AlgorithmError):
            result = vqd.compute_eigenvalues(operator=self.h2_op)

         

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
            vqd = VQD(k = 2, ansatz=wavefunction, expectation=expectation)
            params = [0] * wavefunction.num_parameters
            circuits = vqd.construct_circuit(parameter=params, operator=self.h2_op)

            self.assertEqual(len(circuits), num_circuits)
            for circuit in circuits:
                self.assertIsInstance(circuit, QuantumCircuit)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
            return

    def test_missing_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.h2_op.num_qubits)
        vqd = VQD(k=1,ansatz=circuit, quantum_instance=BasicAer.get_backend("statevector_simulator"))
        with self.assertRaises(RuntimeError):
            vqd.compute_eigenvalues(operator=self.h2_op)

    @data(
        (SLSQP(maxiter=50), 5, 4) # max_evals_grouped=n or =2 if n>2
    )
    @unpack
    def test_max_evals_grouped(self, optimizer, places, max_evals_grouped):
        """VQE Optimizers test"""
        vqd = VQD(
            k=2,
            ansatz=self.ryrz_wavefunction,
            optimizer=optimizer,
            max_evals_grouped=max_evals_grouped,
            quantum_instance=self.statevector_simulator,
        )
        result = vqd.compute_eigenvalues(operator=self.h2_op)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 2)

    def test_basic_aer_qasm(self):
        """Test the VQD on BasicAer's QASM simulator."""
        optimizer = COBYLA(maxiter=1000)
        wavefunction = self.ry_wavefunction

        vqd = VQD(

            ansatz=wavefunction,
            optimizer=optimizer,
            max_evals_grouped=1,
            quantum_instance=self.qasm_simulator,
        )

        # TODO benchmark this later.
        result = vqd.compute_eigenvalues(operator=self.h2_op)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 2)



    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_with_aer_statevector(self):
        """Test VQD with Aer's statevector_simulator."""
        backend = Aer.get_backend("aer_simulator_statevector")
        wavefunction = self.ry_wavefunction
        optimizer = L_BFGS_B()

        quantum_instance = QuantumInstance(
            backend,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        vqd = VQD(
            k=2,
            ansatz=wavefunction,
            optimizer=optimizer,
            max_evals_grouped=1,
            quantum_instance=quantum_instance,
        )

        result = vqd.compute_eigenvalues(operator=self.h2_op)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 2)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_with_aer_qasm(self):
        """Test VQD with Aer's qasm_simulator."""
        backend = Aer.get_backend("aer_simulator")
        optimizer = SPSA(maxiter=200, last_avg=5)
        wavefunction = self.ry_wavefunction

        quantum_instance = QuantumInstance(
            backend,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        vqd = VQD(
            k=2,
            ansatz=wavefunction,
            optimizer=optimizer,
            expectation=PauliExpectation(),
            quantum_instance=quantum_instance,
        )

        result = vqd.compute_eigenvalues(operator=self.h2_op)

        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 2)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_with_aer_qasm_snapshot_mode(self):
        """Test the VQD using Aer's qasm_simulator snapshot mode."""

        backend = Aer.get_backend("aer_simulator")
        optimizer = L_BFGS_B()
        wavefunction = self.ry_wavefunction

        quantum_instance = QuantumInstance(
            backend,
            shots=1,
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        vqd = VQD(
            k=2,
            ansatz=wavefunction,
            optimizer=optimizer,
            expectation=AerPauliExpectation(),
            quantum_instance=quantum_instance,
        )

        result = vqd.compute_eigenvalues(operator=self.h2_op)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 2)


    def test_callback(self):
        """Test the callback on VQD."""
        history = {"eval_count": [], "parameters": [], "mean": [], "std": []}

        def store_intermediate_result(eval_count, parameters, mean, std):
            history["eval_count"].append(eval_count)
            history["parameters"].append(parameters)
            history["mean"].append(mean)
            history["std"].append(std)

        optimizer = COBYLA(maxiter=3)
        wavefunction = self.ry_wavefunction

        vqd = VQD(
            ansatz=wavefunction,
            optimizer=optimizer,
            callback=store_intermediate_result,
            quantum_instance=self.qasm_simulator,
        )
        vqd.compute_eigenvalues(operator=self.h2_op)

        self.assertTrue(all(isinstance(count, int) for count in history["eval_count"]))
        self.assertTrue(all(isinstance(mean, float) for mean in history["mean"]))
        self.assertTrue(all(isinstance(std, float) for std in history["std"]))
        for params in history["parameters"]:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_reuse(self):
        """Test re-using a VQD algorithm instance."""
        vqd = VQD(k = 1)
        with self.subTest(msg="assert running empty raises AlgorithmError"):
            with self.assertRaises(AlgorithmError):
                _ = vqd.compute_eigenvalues(operator=self.h2_op)

        ansatz = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        vqd.ansatz = ansatz
        with self.subTest(msg="assert missing operator raises AlgorithmError"):
            with self.assertRaises(AlgorithmError):
                _ = vqd.compute_eigenvalues(operator=self.h2_op)

        vqd.expectation = MatrixExpectation()
        vqd.quantum_instance = self.statevector_simulator
        with self.subTest(msg="assert VQE works once all info is available"):
            result = vqd.compute_eigenvalues(operator=self.h2_op)
            np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy, decimal = 2)

        operator = PrimitiveOp(np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]]))

        with self.subTest(msg="assert minimum eigensolver interface works"):
            result = vqd.compute_eigenvalues(operator=operator)
            self.assertAlmostEqual(result.eigenvalues.real[0], -1.0, places=5)

    def test_vqd_optimizer(self):
        """Test running same VQD twice to re-use optimizer, then switch optimizer"""
        vqd = VQD(
            k =2,
            optimizer=SLSQP(),
            quantum_instance=QuantumInstance(BasicAer.get_backend("statevector_simulator")),
        )

        def run_check():
            result = vqd.compute_eigenvalues(operator=self.h2_op)
            np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal = 3)

        run_check()

        with self.subTest("Optimizer re-use"):
            run_check()

        with self.subTest("Optimizer replace"):
            vqd.optimizer = L_BFGS_B()
            run_check()

    @data(MatrixExpectation(), None)
    def test_backend_change(self, user_expectation):
        """Test that VQE works when backend changes."""
        vqd = VQD(
            k = 1,
            ansatz=TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz"),
            optimizer=SLSQP(maxiter=2),
            expectation=user_expectation,
            quantum_instance=BasicAer.get_backend("statevector_simulator"),
        )
        result0 = vqd.compute_eigenvalues(operator=self.h2_op)
        if user_expectation is not None:
            with self.subTest("User expectation kept."):
                self.assertEqual(vqd.expectation, user_expectation)

        vqd.quantum_instance = BasicAer.get_backend("qasm_simulator")

        # works also if no expectation is set, since it will be determined automatically
        result1 = vqd.compute_eigenvalues(operator=self.h2_op)

        if user_expectation is not None:
            with self.subTest("Change backend with user expectation, it is kept."):
                self.assertEqual(vqd.expectation, user_expectation)

        with self.subTest("Check results."):
            self.assertEqual(len(result0.optimal_point), len(result1.optimal_point))


    def test_set_ansatz_to_none(self):
        """Tests that setting the ansatz to None results in the default behavior"""
        vqd = VQD(
            k=1,
            ansatz=self.ryrz_wavefunction,
            optimizer=L_BFGS_B(),
            quantum_instance=self.statevector_simulator,
        )
        vqd.ansatz = None
        self.assertIsInstance(vqd.ansatz, RealAmplitudes)

    def test_set_optimizer_to_none(self):
        """Tests that setting the optimizer to None results in the default behavior"""
        vqd = VQD(
            k=1,
            ansatz=self.ryrz_wavefunction,
            optimizer=L_BFGS_B(),
            quantum_instance=self.statevector_simulator,
        )
        vqd.optimizer = None
        self.assertIsInstance(vqd.optimizer, SLSQP)

    def test_aux_operators_list(self):
        """Test list-based aux_operators."""
        wavefunction = self.ry_wavefunction
        vqd = VQD(k = 2, ansatz=wavefunction, quantum_instance=self.statevector_simulator)

        # Start with an empty list
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=[])
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal=2)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=aux_ops)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal=2)
        self.assertEqual(len(result.aux_operator_eigenvalues[-1]), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][0], 2, places=2)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][0], 0, places=2)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = [*aux_ops, None, 0]
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=extra_ops)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal=2)
        self.assertEqual(len(result.aux_operator_eigenvalues[-1]), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][0], 2, places=2)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][0], 0, places=2)
        self.assertEqual(result.aux_operator_eigenvalues[0][2], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[0][3][0], 0.0)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[0][3][1], 0.0)

    def test_aux_operators_dict(self):
        """Test dictionary compatibility of aux_operators"""
        wavefunction = self.ry_wavefunction
        vqd = VQD(ansatz=wavefunction, quantum_instance=self.statevector_simulator)

        # Start with an empty dictionary
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators={})
        np.testing.assert_array_almost_equal(result.eigenvalues.real, self.h2_energy_excited, decimal=2)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = {"aux_op1": aux_op1, "aux_op2": aux_op2}
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=aux_ops)
        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(len(result.eigenstates), 2)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        self.assertEqual(len(result.aux_operator_eigenvalues[0]), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = {**aux_ops, "None_operator": None, "zero_operator": 0}
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=extra_ops)
        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)
        self.assertEqual(len(result.aux_operator_eigenvalues), 1)
        self.assertEqual(len(result.aux_operator_eigenvalues[0]), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][0], 0, places=6)
        self.assertIsNone(result.aux_operator_eigenvalues[0][2], None)
        self.assertEqual(result.aux_operator_eigenvalues[0][3][0], 0.0)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[0][3][1], 0.0)

    def test_aux_operator_std_dev_pauli(self):
        """Test non-zero standard deviations of aux operators with PauliExpectation."""
        wavefunction = self.ry_wavefunction
        vqd = VQD(
            ansatz=wavefunction,
            expectation=PauliExpectation(),
            optimizer=COBYLA(maxiter=0),
            quantum_instance=self.qasm_simulator,
        )

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=aux_ops)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.5784419552370315, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(
            result.aux_operator_eigenvalues[1][1], 0.015183867579396111, places=6
        )

        # Go again with additional None and zero operators
        aux_ops = [*aux_ops, None, 0]
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=aux_ops)
        self.assertEqual(len(result.aux_operator_eigenvalues), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.56640625, places=6)
        self.assertEqual(result.aux_operator_eigenvalues[2][0], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[3][0], 0.0)
        # # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.01548658094658011, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[2][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[3][1], 0.0)

    @unittest.skipUnless(has_aer(), "qiskit-aer doesn't appear to be installed.")
    def test_aux_operator_std_dev_aer_pauli(self):
        """Test non-zero standard deviations of aux operators with AerPauliExpectation."""
        wavefunction = self.ry_wavefunction
        vqd = VQD(
            ansatz=wavefunction,
            expectation=AerPauliExpectation(),
            optimizer=COBYLA(maxiter=0),
            quantum_instance=QuantumInstance(
                backend=Aer.get_backend("qasm_simulator"),
                shots=1,
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )

        # Go again with two auxiliary operators
        aux_op1 = PauliSumOp.from_list([("II", 2.0)])
        aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=aux_ops) 
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2.0, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0.6698863565455391, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0, places=6)

        # Go again with additional None and zero operators
        aux_ops = [*aux_ops, None, 0]
        result = vqd.compute_eigenvalues(self.h2_op, aux_operators=aux_ops)
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


    def test_construct_eigenstate_from_optpoint(self):
        """Test constructing the eigenstate from the optimal point, if the default ansatz is used."""

        # use Hamiltonian yielding more than 11 parameters in the default ansatz
        hamiltonian = Z ^ Z ^ Z
        optimizer = SPSA(maxiter=1, learning_rate=0.01, perturbation=0.01)
        quantum_instance = QuantumInstance(
            backend=BasicAer.get_backend("statevector_simulator"), basis_gates=["u3", "cx"]
        )
        vqe = VQD(optimizer=optimizer, quantum_instance=quantum_instance)
        result = vqe.compute_eigenvalues(hamiltonian)

        optimal_circuit = vqe.ansatz.bind_parameters(result.optimal_point[-1])
        self.assertTrue(Statevector(result.eigenstates[-1]).equiv(optimal_circuit))


if __name__ == "__main__":
    unittest.main()
