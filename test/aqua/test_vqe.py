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
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, unpack, data
from qiskit import BasicAer, QuantumCircuit, IBMQ
from qiskit.circuit import ParameterVector

from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.operators import WeightedPauliOperator, PrimitiveOp
from qiskit.aqua.components.variational_forms import RY, RYRZ, VariationalForm
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA, SPSA, SLSQP
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.algorithms import VQE


@ddt
class TestVQE(QiskitAquaTestCase):
    """ Test VQE """

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict).to_opflow()

    @data(VariationalForm, QuantumCircuit)
    def test_vqe(self, var_form_type):
        """ VQE test """
        var_form = RYRZ(self.qubit_op.num_qubits)
        if var_form_type is QuantumCircuit:
            params = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(params)

        vqe = VQE(self.qubit_op, var_form, L_BFGS_B())
        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                         coupling_map=[[0, 1]],
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)

    def test_vqe_no_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.qubit_op.num_qubits)
        for i in range(circuit.num_qubits):
            circuit.h(i)
            circuit.cx(i, (i + 1) % circuit.num_qubits)
            circuit.rx(0.2, i)

        with self.assertRaises(AquaError):
            _ = VQE(self.qubit_op, circuit)

    @data(
        (SLSQP, 5, 4),
        (SLSQP, 5, 1),
        (SPSA, 3, 2),  # max_evals_grouped=n is considered as max_evals_grouped=2 if n>2
        (SPSA, 3, 1)
    )
    @unpack
    def test_vqe_optimizers(self, optimizer_cls, places, max_evals_grouped):
        """ VQE Optimizers test """
        result = VQE(self.qubit_op,
                     RYRZ(self.qubit_op.num_qubits),
                     optimizer_cls(),
                     max_evals_grouped=max_evals_grouped).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'), shots=1,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))

        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=places)

    @data(
        (RY, 5, VariationalForm),
        (RYRZ, 5, VariationalForm),
        (RY, 5, QuantumCircuit),
        (RYRZ, 5, QuantumCircuit),
    )
    @unpack
    def test_vqe_var_forms(self, var_form_cls, places, var_form_type):
        """ VQE Var Forms test """
        var_form = var_form_cls(self.qubit_op.num_qubits)
        if var_form_type is QuantumCircuit:
            params = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(params)

        vqe = VQE(self.qubit_op, var_form, L_BFGS_B())
        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'), shots=1,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=places)

    @data(VariationalForm, QuantumCircuit)
    def test_vqe_qasm(self, var_form_type):
        """ VQE QASM test """
        backend = BasicAer.get_backend('qasm_simulator')
        num_qubits = self.qubit_op.num_qubits
        var_form = RY(num_qubits, depth=3)
        if var_form_type is QuantumCircuit:
            params = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(params)

        optimizer = SPSA(max_trials=300, last_avg=5)
        algo = VQE(self.qubit_op, var_form, optimizer)
        # TODO benchmark this later.
        quantum_instance = QuantumInstance(backend, shots=1000,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.86823, places=2)

    @data(VariationalForm, QuantumCircuit)
    def test_vqe_statevector_snapshot_mode(self, var_form_type):
        """ VQE Aer statevector_simulator snapshot mode test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('statevector_simulator')
        num_qubits = self.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, depth=3, initial_state=init_state)
        if var_form_type is QuantumCircuit:
            params = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(params)

        optimizer = L_BFGS_B()
        algo = VQE(self.qubit_op, var_form, optimizer, max_evals_grouped=1)
        quantum_instance = QuantumInstance(backend,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=6)

    @data(VariationalForm, QuantumCircuit)
    def test_vqe_qasm_snapshot_mode(self, var_form_type):
        """ VQE Aer qasm_simulator snapshot mode test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')
        num_qubits = self.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, depth=3, initial_state=init_state)
        if var_form_type is QuantumCircuit:
            params = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(params)

        optimizer = L_BFGS_B()
        algo = VQE(self.qubit_op, var_form, optimizer, max_evals_grouped=1)
        quantum_instance = QuantumInstance(backend, shots=1,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = algo.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=6)

    @data(VariationalForm, QuantumCircuit)
    def test_vqe_callback(self, var_form_type):
        """ VQE Callback test """
        history = {'eval_count': [], 'parameters': [], 'mean': [], 'std': []}

        def store_intermediate_result(eval_count, parameters, mean, std):
            history['eval_count'].append(eval_count)
            history['parameters'].append(parameters)
            history['mean'].append(mean)
            history['std'].append(std)

        backend = BasicAer.get_backend('qasm_simulator')
        num_qubits = self.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, depth=1, initial_state=init_state)
        if var_form_type is QuantumCircuit:
            params = ParameterVector('θ', var_form.num_parameters)
            var_form = var_form.construct_circuit(params)
        optimizer = COBYLA(maxiter=3)
        algo = VQE(self.qubit_op, var_form, optimizer,
                   callback=store_intermediate_result)
        aqua_globals.random_seed = 50
        quantum_instance = QuantumInstance(backend,
                                           seed_transpiler=50,
                                           shots=1024,
                                           seed_simulator=50)
        algo.run(quantum_instance)

        self.assertTrue(all(isinstance(count, int) for count in history['eval_count']))
        self.assertTrue(all(isinstance(mean, float) for mean in history['mean']))
        self.assertTrue(all(isinstance(std, float) for std in history['std']))
        for params in history['parameters']:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_vqe_reuse(self):
        """ Test vqe reuse """
        vqe = VQE()
        with self.assertRaises(AquaError):
            _ = vqe.run()

        num_qubits = self.qubit_op.num_qubits
        var_form = RY(num_qubits, depth=3)
        vqe.var_form = var_form
        with self.assertRaises(AquaError):
            _ = vqe.run()

        vqe.operator = self.qubit_op
        with self.assertRaises(AquaError):
            _ = vqe.run()

        qinst = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        vqe.quantum_instance = qinst
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

        operator = PrimitiveOp(np.array([[1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 2, 0],
                                         [0, 0, 0, 3]]))
        vqe.operator = operator
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_mes(self):
        """ Test vqe minimum eigen solver interface """
        vqe = VQE(var_form=RY(self.qubit_op.num_qubits, depth=3), optimizer=COBYLA())
        vqe.set_backend(BasicAer.get_backend('statevector_simulator'))
        result = vqe.compute_minimum_eigenvalue(self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

    @unittest.skip(reason="IBMQ testing not available in general.")
    def test_ibmq_vqe(self):
        """ IBMQ VQE Test """
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_qasm_simulator')
        var_form = RYRZ(self.qubit_op.num_qubits)

        opt = SLSQP(maxiter=1)
        opt.set_max_evals_grouped(100)
        vqe = VQE(self.qubit_op, var_form, SLSQP(maxiter=2))

        result = vqe.run(backend)
        print(result)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)


if __name__ == '__main__':
    unittest.main()
