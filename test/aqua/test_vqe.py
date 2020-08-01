# -*- coding: utf-8 -*-

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

""" Test VQE """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, unpack, data
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import TwoLocal, EfficientSU2

from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.operators import (WeightedPauliOperator, PrimitiveOp, X, Z, I,
                                   AerPauliExpectation, PauliExpectation,
                                   MatrixExpectation, ExpectationBase)
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA, SPSA, SLSQP
from qiskit.aqua.algorithms import VQE


@ddt
class TestVQE(QiskitAquaTestCase):
    """ Test VQE """

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        self.h2_op = -1.052373245772859 * (I ^ I) \
            + 0.39793742484318045 * (I ^ Z) \
            - 0.39793742484318045 * (Z ^ I) \
            - 0.01128010425623538 * (Z ^ Z) \
            + 0.18093119978423156 * (X ^ X)
        self.h2_energy = -1.85727503

        self.ryrz_wavefunction = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')
        self.ry_wavefunction = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')

        self.qasm_simulator = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                              shots=1024,
                                              seed_simulator=self.seed,
                                              seed_transpiler=self.seed)
        self.statevector_simulator = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                     shots=1,
                                                     seed_simulator=self.seed,
                                                     seed_transpiler=self.seed)

    def test_basic_aer_statevector(self):
        """Test the VQE on BasicAer's statevector simulator."""
        wavefunction = self.ryrz_wavefunction
        vqe = VQE(self.h2_op, wavefunction, L_BFGS_B())

        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                         coupling_map=[[0, 1]],
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))

        with self.subTest(msg='test eigenvalue'):
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy)

        with self.subTest(msg='test dimension of optimal point'):
            self.assertEqual(len(result.optimal_point), 16)

        with self.subTest(msg='assert cost_function_evals is set'):
            self.assertIsNotNone(result.cost_function_evals)

        with self.subTest(msg='assert optimizer_time is set'):
            self.assertIsNotNone(result.optimizer_time)

    def test_deprecated_variational_forms(self):
        """Test running the VQE on a deprecated VariationalForm object."""
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        wavefunction = RYRZ(2)
        vqe = VQE(self.h2_op, wavefunction, L_BFGS_B())
        warnings.filterwarnings('always', category=DeprecationWarning)
        result = vqe.run(self.statevector_simulator)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy)

    def test_circuit_input(self):
        """Test running the VQE on a plain QuantumCircuit object."""
        wavefunction = QuantumCircuit(2).compose(EfficientSU2(2))
        optimizer = SLSQP(maxiter=50)
        vqe = VQE(self.h2_op, wavefunction, optimizer=optimizer)
        result = vqe.run(self.statevector_simulator)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

    @data(
        (MatrixExpectation(), 1),
        (AerPauliExpectation(), 1),
        (PauliExpectation(), 2),
    )
    @unpack
    def test_construct_circuit(self, expectation, num_circuits):
        """Test construct circuits returns QuantumCircuits and the right number of them."""
        wavefunction = EfficientSU2(2, reps=1)
        vqe = VQE(self.h2_op, wavefunction, expectation=expectation)
        params = [0] * wavefunction.num_parameters
        circuits = vqe.construct_circuit(params)

        self.assertEqual(len(circuits), num_circuits)
        for circuit in circuits:
            self.assertIsInstance(circuit, QuantumCircuit)

    def test_legacy_operator(self):
        """Test the VQE accepts and converts the legacy WeightedPauliOperator."""
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        h2_op = WeightedPauliOperator.from_dict(pauli_dict)
        vqe = VQE(h2_op)
        self.assertEqual(vqe.operator, self.h2_op)

    def test_missing_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.h2_op.num_qubits)
        vqe = VQE(self.h2_op, circuit)
        with self.assertRaises(RuntimeError):
            vqe.run(BasicAer.get_backend('statevector_simulator'))

    @data(
        (SLSQP(maxiter=50), 5, 4),
        (SPSA(maxiter=150), 3, 2),  # max_evals_grouped=n or =2 if n>2
    )
    @unpack
    def test_max_evals_grouped(self, optimizer, places, max_evals_grouped):
        """ VQE Optimizers test """
        vqe = VQE(self.h2_op, self.ryrz_wavefunction, optimizer,
                  max_evals_grouped=max_evals_grouped,
                  quantum_instance=self.statevector_simulator)
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=places)

    def test_basic_aer_qasm(self):
        """Test the VQE on BasicAer's QASM simulator."""
        optimizer = SPSA(maxiter=300, last_avg=5)
        wavefunction = self.ry_wavefunction

        vqe = VQE(self.h2_op, wavefunction, optimizer, max_evals_grouped=1)

        # TODO benchmark this later.
        result = vqe.run(self.qasm_simulator)
        self.assertAlmostEqual(result.eigenvalue.real, -1.86823, places=2)

    def test_statevector_snapshot_mode(self):
        """Test the VQE using Aer's statevector_simulator snapshot mode."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('statevector_simulator')
        wavefunction = self.ry_wavefunction
        optimizer = L_BFGS_B()

        vqe = VQE(self.h2_op, wavefunction, optimizer, max_evals_grouped=1)

        quantum_instance = QuantumInstance(backend,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)

    def test_qasm_snapshot_mode(self):
        """Test the VQE using Aer's qasm_simulator snapshot mode."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')
        optimizer = L_BFGS_B()
        wavefunction = self.ry_wavefunction

        vqe = VQE(self.h2_op, wavefunction, optimizer,
                  expectation=AerPauliExpectation(), max_evals_grouped=1)

        quantum_instance = QuantumInstance(backend, shots=1,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=6)

    def test_callback(self):
        """Test the callback on VQE."""
        history = {'eval_count': [], 'parameters': [], 'mean': [], 'std': []}

        def store_intermediate_result(eval_count, parameters, mean, std):
            history['eval_count'].append(eval_count)
            history['parameters'].append(parameters)
            history['mean'].append(mean)
            history['std'].append(std)

        optimizer = COBYLA(maxiter=3)
        wavefunction = self.ry_wavefunction

        vqe = VQE(self.h2_op, wavefunction, optimizer, callback=store_intermediate_result)
        vqe.run(self.qasm_simulator)

        self.assertTrue(all(isinstance(count, int) for count in history['eval_count']))
        self.assertTrue(all(isinstance(mean, float) for mean in history['mean']))
        self.assertTrue(all(isinstance(std, float) for std in history['std']))
        for params in history['parameters']:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_reuse(self):
        """Test re-using a VQE algorithm instance."""
        vqe = VQE()
        with self.subTest(msg='assert running empty raises AquaError'):
            with self.assertRaises(AquaError):
                _ = vqe.run()

        var_form = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')
        vqe.var_form = var_form
        with self.subTest(msg='assert missing operator raises AquaError'):
            with self.assertRaises(AquaError):
                _ = vqe.run()

        vqe.operator = self.h2_op
        with self.subTest(msg='assert missing backend raises AquaError'):
            with self.assertRaises(AquaError):
                _ = vqe.run()

        vqe.quantum_instance = self.statevector_simulator
        with self.subTest(msg='assert VQE works once all info is available'):
            result = vqe.run()
            self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy, places=5)

        operator = PrimitiveOp(np.array([[1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 2, 0],
                                         [0, 0, 0, 3]]))

        with self.subTest(msg='assert minimum eigensolver interface works'):
            result = vqe.compute_minimum_eigenvalue(operator)
            self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_optimizer(self):
        """ Test running same VQE twice to re-use optimizer, then switch optimizer """
        vqe = VQE(self.h2_op, optimizer=SLSQP(),
                  quantum_instance=QuantumInstance(BasicAer.get_backend('statevector_simulator')))

        def run_check():
            result = vqe.compute_minimum_eigenvalue()
            self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

        run_check()

        with self.subTest('Optimizer re-use'):
            run_check()

        with self.subTest('Optimizer replace'):
            vqe.optimizer = L_BFGS_B()
            run_check()

    def test_vqe_expectation_select(self):
        """Test expectation selection with Aer's qasm_simulator."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')

        with self.subTest('Defaults'):
            vqe = VQE(self.h2_op, quantum_instance=backend)
            self.assertIsInstance(vqe.expectation, PauliExpectation)

        with self.subTest('Include custom'):
            vqe = VQE(self.h2_op, include_custom=True, quantum_instance=backend)
            self.assertIsInstance(vqe.expectation, AerPauliExpectation)

        with self.subTest('Set explicitly'):
            vqe = VQE(self.h2_op, expectation=AerPauliExpectation(), quantum_instance=backend)
            self.assertIsInstance(vqe.expectation, AerPauliExpectation)

    @unittest.skip(reason="IBMQ testing not available in general.")
    def test_ibmq(self):
        """ IBMQ VQE Test """
        from qiskit import IBMQ
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_qasm_simulator')
        ansatz = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')

        opt = SLSQP(maxiter=1)
        opt.set_max_evals_grouped(100)
        vqe = VQE(self.h2_op, ansatz, SLSQP(maxiter=2))

        result = vqe.run(backend)
        print(result)
        self.assertAlmostEqual(result.eigenvalue.real, self.h2_energy)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, self.h2_energy, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)

    @data(MatrixExpectation(), None)
    def test_backend_change(self, user_expectation):
        """Test that VQE works when backend changes."""
        vqe = VQE(operator=self.h2_op,
                  var_form=TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz'),
                  optimizer=SLSQP(maxiter=2),
                  expectation=user_expectation,
                  quantum_instance=BasicAer.get_backend('statevector_simulator'))
        result0 = vqe.run()
        if user_expectation is not None:
            with self.subTest('User expectation kept.'):
                self.assertEqual(vqe.expectation, user_expectation)
        else:
            with self.subTest('Expectation created.'):
                self.assertIsInstance(vqe.expectation, ExpectationBase)
        try:
            vqe.set_backend(BasicAer.get_backend('qasm_simulator'))
        except Exception as ex:  # pylint: disable=broad-except
            self.fail("Failed to change backend. Error: '{}'".format(str(ex)))
            return

        result1 = vqe.run()
        if user_expectation is not None:
            with self.subTest('Change backend with user expectation, it is kept.'):
                self.assertEqual(vqe.expectation, user_expectation)
        else:
            with self.subTest('Change backend without user expectation, one created.'):
                self.assertIsInstance(vqe.expectation, ExpectationBase)

        with self.subTest('Check results.'):
            self.assertEqual(len(result0.optimal_point), len(result1.optimal_point))


if __name__ == '__main__':
    unittest.main()
