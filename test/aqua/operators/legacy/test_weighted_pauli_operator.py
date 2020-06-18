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

""" Test WeightedPauliOperator """

import unittest
import itertools
import warnings
import os
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Pauli, state_fidelity
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.initial_states import Custom


@ddt
class TestWeightedPauliOperator(QiskitAquaTestCase):
    """WeightedPauliOperator tests."""

    def setUp(self):
        super().setUp()
        seed = 0
        aqua_globals.random_seed = seed

        self.num_qubits = 3
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=self.num_qubits)]
        weights = aqua_globals.random.random(len(paulis))
        self.qubit_op = WeightedPauliOperator.from_list(paulis, weights)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        self.var_form = RYRZ(self.qubit_op.num_qubits, 1)
        warnings.filterwarnings('always', category=DeprecationWarning)

        qasm_simulator = BasicAer.get_backend('qasm_simulator')
        self.quantum_instance_qasm = QuantumInstance(qasm_simulator, shots=65536,
                                                     seed_simulator=seed, seed_transpiler=seed)
        statevector_simulator = BasicAer.get_backend('statevector_simulator')
        self.quantum_instance_statevector = \
            QuantumInstance(statevector_simulator, shots=1,
                            seed_simulator=seed, seed_transpiler=seed)

    def test_from_to_file(self):
        """ from to file test """
        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        weights = [0.2 + -1j * 0.8, 0.6 + -1j * 0.6, 0.8 + -1j * 0.2,
                   -0.2 + -1j * 0.8, -0.6 - -1j * 0.6, -0.8 - -1j * 0.2]
        op = WeightedPauliOperator.from_list(paulis, weights)
        file_path = self.get_resource_path('temp_op.json')
        op.to_file(file_path)
        self.assertTrue(os.path.exists(file_path))

        load_op = WeightedPauliOperator.from_file(file_path)
        self.assertEqual(op, load_op)
        os.remove(file_path)

    def test_num_qubits(self):
        """ num qubits test """
        op = WeightedPauliOperator(paulis=[])
        self.assertEqual(op.num_qubits, 0)
        self.assertEqual(self.qubit_op.num_qubits, self.num_qubits)

    def test_is_empty(self):
        """ is empty test """
        op = WeightedPauliOperator(paulis=[])
        self.assertTrue(op.is_empty())
        self.assertFalse(self.qubit_op.is_empty())

    def test_str(self):
        """ str test """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        op_a += op_b

        self.assertEqual("Representation: paulis, qubits: 4, size: 2", str(op_a))

        op_a = WeightedPauliOperator(paulis=[pauli_term_a], name='ABC')
        self.assertEqual("ABC: Representation: paulis, qubits: 4, size: 1", str(op_a))

    def test_multiplication(self):
        """ multiplication test """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        new_op = op_a * op_b

        self.assertEqual(1, len(new_op.paulis))
        self.assertEqual(-0.25, new_op.paulis[0][0])
        self.assertEqual('ZZYY', new_op.paulis[0][1].to_label())

        new_op = -2j * new_op
        self.assertEqual(0.5j, new_op.paulis[0][0])

        new_op = new_op * 0.3j
        self.assertEqual(-0.15, new_op.paulis[0][0])

    def test_iadd(self):
        """ iadd test """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        ori_op_a = op_a.copy()
        ori_op_b = op_b.copy()
        op_a += op_b

        self.assertNotEqual(op_a, ori_op_a)
        self.assertEqual(op_b, ori_op_b)
        self.assertEqual(2, len(op_a.paulis))

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        op_a += WeightedPauliOperator(paulis=[pauli_term_c])

        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.75, op_a.paulis[0][0])

    def test_add(self):
        """ add test """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        ori_op_a = op_a.copy()
        ori_op_b = op_b.copy()
        new_op = op_a + op_b

        self.assertEqual(op_a, ori_op_a)
        self.assertEqual(op_b, ori_op_b)
        self.assertEqual(1, len(op_a.paulis))
        self.assertEqual(2, len(new_op.paulis))

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        new_op = new_op + WeightedPauliOperator(paulis=[pauli_term_c])

        self.assertEqual(2, len(new_op.paulis))
        self.assertEqual(0.75, new_op.paulis[0][0])

    def test_sub(self):
        """ sub test """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        ori_op_a = op_a.copy()
        ori_op_b = op_b.copy()
        new_op = op_a - op_b

        self.assertEqual(op_a, ori_op_a)
        self.assertEqual(op_b, ori_op_b)
        self.assertEqual(1, len(op_a.paulis))
        self.assertEqual(2, len(new_op.paulis))
        self.assertEqual(0.5, new_op.paulis[0][0])
        self.assertEqual(-0.5, new_op.paulis[1][0])

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        new_op = new_op - WeightedPauliOperator(paulis=[pauli_term_c])

        self.assertEqual(2, len(new_op.paulis))
        self.assertEqual(0.25, new_op.paulis[0][0])

    def test_isub(self):
        """ isub test """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        ori_op_a = op_a.copy()
        ori_op_b = op_b.copy()
        op_a -= op_b

        self.assertNotEqual(op_a, ori_op_a)
        self.assertEqual(op_b, ori_op_b)
        self.assertEqual(2, len(op_a.paulis))

        pauli_c = 'IXYZ'
        coeff_c = 0.5
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        op_a -= WeightedPauliOperator(paulis=[pauli_term_c])
        # sub does not remove zero weights.
        self.assertEqual(2, len(op_a.paulis))

    def test_equal_operator(self):
        """ equal operator test """
        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op1 = WeightedPauliOperator.from_list(paulis, coeffs)

        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op2 = WeightedPauliOperator.from_list(paulis, coeffs)

        coeffs = [-0.2, -0.6, -0.8, 0.2, 0.6, 0.8]
        op3 = WeightedPauliOperator.from_list(paulis, coeffs)

        coeffs = [-0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op4 = WeightedPauliOperator.from_list(paulis, coeffs)

        self.assertEqual(op1, op2)
        self.assertNotEqual(op1, op3)
        self.assertNotEqual(op1, op4)
        self.assertNotEqual(op3, op4)

    def test_negation_operator(self):
        """ negation operator test """
        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op1 = WeightedPauliOperator.from_list(paulis, coeffs)
        coeffs = [-0.2, -0.6, -0.8, 0.2, 0.6, 0.8]
        op2 = WeightedPauliOperator.from_list(paulis, coeffs)

        self.assertNotEqual(op1, op2)
        self.assertEqual(op1, -op2)
        self.assertEqual(-op1, op2)
        op1 = op1 * -1.0
        self.assertEqual(op1, op2)

    def test_simplify(self):
        """ simplify test """
        pauli_a = 'IXYZ'
        pauli_b = 'IXYZ'
        coeff_a = 0.5
        coeff_b = -0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a])
        op_b = WeightedPauliOperator(paulis=[pauli_term_b])
        new_op = op_a + op_b
        new_op.simplify()

        self.assertEqual(0, len(new_op.paulis), "{}".format(new_op.print_details()))
        self.assertTrue(new_op.is_empty())

        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op1 = WeightedPauliOperator.from_list(paulis, coeffs)

        for i, pauli in enumerate(paulis):
            tmp_op = WeightedPauliOperator(paulis=[[-coeffs[i], pauli]])
            op1 += tmp_op
            op1.simplify()
            self.assertEqual(len(paulis) - (i + 1), len(op1.paulis))

    def test_simplify_same_paulis(self):
        """ simplify same paulis test """
        pauli_a = 'IXYZ'
        pauli_b = 'IXYZ'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = WeightedPauliOperator(paulis=[pauli_term_a, pauli_term_b])

        self.assertEqual(1, len(op_a.paulis), "{}".format(op_a.print_details()))
        self.assertEqual(1, len(op_a.basis))
        self.assertEqual(0, op_a.basis[0][1][0])

    def test_chop_real(self):
        """ chop real test """
        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op = WeightedPauliOperator.from_list(paulis, coeffs)
        ori_op = op.copy()

        for threshold, num_paulis in zip([0.4, 0.7, 0.9], [4, 2, 0]):
            op = ori_op.copy()
            op1 = op.chop(threshold=threshold, copy=True)
            self.assertEqual(len(op.paulis), 6, "\n{}".format(op.print_details()))
            self.assertEqual(len(op1.paulis), num_paulis, "\n{}".format(op1.print_details()))

            op1 = op.chop(threshold=threshold, copy=False)
            self.assertEqual(len(op.paulis), num_paulis, "\n{}".format(op.print_details()))
            self.assertEqual(len(op1.paulis), num_paulis, "\n{}".format(op1.print_details()))

    def test_chop_complex(self):
        """ chop complex test """
        paulis = [Pauli.from_label(x) for x in ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']]
        coeffs = [0.2 + -0.5j, 0.6 - 0.3j, 0.8 - 0.6j,
                  -0.5 + -0.2j, -0.3 + 0.6j, -0.6 + 0.8j]
        op = WeightedPauliOperator.from_list(paulis, coeffs)
        ori_op = op.copy()
        for threshold, num_paulis in zip([0.4, 0.7, 0.9], [6, 2, 0]):
            op = ori_op.copy()
            op1 = op.chop(threshold=threshold, copy=True)
            self.assertEqual(len(op.paulis), 6, "\n{}".format(op.print_details()))
            self.assertEqual(len(op1.paulis), num_paulis, "\n{}".format(op1.print_details()))

            op1 = op.chop(threshold=threshold, copy=False)
            self.assertEqual(len(op.paulis), num_paulis, "\n{}".format(op.print_details()))
            self.assertEqual(len(op1.paulis), num_paulis, "\n{}".format(op1.print_details()))

    def test_evaluate_single_pauli_qasm(self):
        """ evaluate single pauli qasm test """
        # X
        op = WeightedPauliOperator.from_list([Pauli.from_label('X')])
        qr = QuantumRegister(1, name='q')
        wave_function = QuantumCircuit(qr)
        # + 1 eigenstate
        wave_function.h(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=False)
        self.assertAlmostEqual(1.0, actual_value[0].real, places=5)
        # - 1 eigenstate
        wave_function = QuantumCircuit(qr)
        wave_function.x(qr[0])
        wave_function.h(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=False)
        self.assertAlmostEqual(-1.0, actual_value[0].real, places=5)

        # Y
        op = WeightedPauliOperator.from_list([Pauli.from_label('Y')])
        qr = QuantumRegister(1, name='q')
        wave_function = QuantumCircuit(qr)
        # + 1 eigenstate
        wave_function.h(qr[0])
        wave_function.s(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=False)
        self.assertAlmostEqual(1.0, actual_value[0].real, places=5)
        # - 1 eigenstate
        wave_function = QuantumCircuit(qr)
        wave_function.x(qr[0])
        wave_function.h(qr[0])
        wave_function.s(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=False)
        self.assertAlmostEqual(-1.0, actual_value[0].real, places=5)

        # Z
        op = WeightedPauliOperator.from_list([Pauli.from_label('Z')])
        qr = QuantumRegister(1, name='q')
        wave_function = QuantumCircuit(qr)
        # + 1 eigenstate
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=False)
        self.assertAlmostEqual(1.0, actual_value[0].real, places=5)
        # - 1 eigenstate
        wave_function = QuantumCircuit(qr)
        wave_function.x(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=False)
        self.assertAlmostEqual(-1.0, actual_value[0].real, places=5)

    def test_evaluate_single_pauli_statevector(self):
        """ evaluate single pauli statevector test """
        # X
        op = WeightedPauliOperator.from_list([Pauli.from_label('X')])
        qr = QuantumRegister(1, name='q')
        wave_function = QuantumCircuit(qr)
        # + 1 eigenstate
        wave_function.h(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=True)
        result = self.quantum_instance_statevector.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=True)
        self.assertAlmostEqual(1.0, actual_value[0].real, places=5)
        # - 1 eigenstate
        wave_function = QuantumCircuit(qr)
        wave_function.x(qr[0])
        wave_function.h(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=True)
        result = self.quantum_instance_statevector.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=True)
        self.assertAlmostEqual(-1.0, actual_value[0].real, places=5)

        # Y
        op = WeightedPauliOperator.from_list([Pauli.from_label('Y')])
        qr = QuantumRegister(1, name='q')
        wave_function = QuantumCircuit(qr)
        # + 1 eigenstate
        wave_function.h(qr[0])
        wave_function.s(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=True)
        result = self.quantum_instance_statevector.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=True)
        self.assertAlmostEqual(1.0, actual_value[0].real, places=5)
        # - 1 eigenstate
        wave_function = QuantumCircuit(qr)
        wave_function.x(qr[0])
        wave_function.h(qr[0])
        wave_function.s(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=True)
        result = self.quantum_instance_statevector.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=True)
        self.assertAlmostEqual(-1.0, actual_value[0].real, places=5)

        # Z
        op = WeightedPauliOperator.from_list([Pauli.from_label('Z')])
        qr = QuantumRegister(1, name='q')
        wave_function = QuantumCircuit(qr)
        # + 1 eigenstate
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=True)
        result = self.quantum_instance_statevector.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=True)
        self.assertAlmostEqual(1.0, actual_value[0].real, places=5)
        # - 1 eigenstate
        wave_function = QuantumCircuit(qr)
        wave_function.x(qr[0])
        circuits = op.construct_evaluation_circuit(wave_function=wave_function,
                                                   statevector_mode=True)
        result = self.quantum_instance_statevector.execute(circuits)
        actual_value = op.evaluate_with_result(result=result, statevector_mode=True)
        self.assertAlmostEqual(-1.0, actual_value[0].real, places=5)

    def test_evaluate_qasm_mode(self):
        """ evaluate qasm mode test """
        wave_function = self.var_form.construct_circuit(
            np.array(aqua_globals.random.standard_normal(self.var_form.num_parameters)))

        circuits = self.qubit_op.construct_evaluation_circuit(
            wave_function=wave_function, statevector_mode=True)
        reference = self.qubit_op.evaluate_with_result(
            result=self.quantum_instance_statevector.execute(circuits), statevector_mode=True)
        circuits = self.qubit_op.construct_evaluation_circuit(
            wave_function=wave_function, statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        actual_value = self.qubit_op.evaluate_with_result(result=result,
                                                          statevector_mode=False)

        self.assertGreaterEqual(reference[0].real, actual_value[0].real - 3 * actual_value[1].real)
        self.assertLessEqual(reference[0].real, actual_value[0].real + 3 * actual_value[1].real)

    def test_evaluate_statevector_mode(self):
        """ evaluate statevector mode test """
        wave_function = self.var_form.construct_circuit(
            np.array(aqua_globals.random.standard_normal(self.var_form.num_parameters)))
        wave_fn_statevector = \
            self.quantum_instance_statevector.execute(wave_function).get_statevector(wave_function)
        # use matrix operator as reference:
        reference = self.qubit_op.evaluate_with_statevector(wave_fn_statevector)

        circuits = self.qubit_op.construct_evaluation_circuit(wave_function=wave_function,
                                                              statevector_mode=True)
        actual_value = self.qubit_op.evaluate_with_result(
            result=self.quantum_instance_statevector.execute(circuits), statevector_mode=True)
        self.assertAlmostEqual(reference[0], actual_value[0], places=10)

    def test_evaluate_with_aer_mode(self):
        """ evaluate with aer mode test """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

        statevector_simulator = Aer.get_backend('statevector_simulator')
        quantum_instance_statevector = QuantumInstance(statevector_simulator, shots=1)

        wave_function = self.var_form.construct_circuit(
            np.array(aqua_globals.random.standard_normal(self.var_form.num_parameters)))

        circuits = self.qubit_op.construct_evaluation_circuit(wave_function=wave_function,
                                                              statevector_mode=True)
        reference = self.qubit_op.evaluate_with_result(
            result=quantum_instance_statevector.execute(circuits), statevector_mode=True)

        circuits = self.qubit_op.construct_evaluation_circuit(
            wave_function=wave_function, statevector_mode=True, use_simulator_snapshot_mode=True)
        actual_value = self.qubit_op.evaluate_with_result(
            result=quantum_instance_statevector.execute(circuits),
            statevector_mode=True,
            use_simulator_snapshot_mode=True)
        self.assertAlmostEqual(reference[0], actual_value[0], places=10)

    @idata([
        ['trotter', 1, 3],
        ['suzuki', 1, 3]
    ])
    @unpack
    def test_evolve(self, expansion_mode, evo_time, num_time_slices):
        """ evolve test """
        expansion_orders = [1, 2, 3, 4] if expansion_mode == 'suzuki' else [1]
        num_qubits = 2
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=num_qubits)]
        weights = aqua_globals.random.random(len(paulis))
        pauli_op = WeightedPauliOperator.from_list(paulis, weights)
        matrix_op = op_converter.to_matrix_operator(pauli_op)
        state_in = Custom(num_qubits, state='random')

        # get the exact state_out from raw matrix multiplication
        state_out_exact = matrix_op.evolve(
            state_in=state_in.construct_circuit('vector'),
            evo_time=evo_time,
            num_time_slices=0
        )
        # self.log.debug('exact:\n%s', state_out_exact)
        self.log.debug('Under %s expansion mode:', expansion_mode)
        for expansion_order in expansion_orders:
            # assure every time the operator from the original one
            if expansion_mode == 'suzuki':
                self.log.debug('With expansion order %s:', expansion_order)
            state_out_matrix = matrix_op.evolve(
                state_in=state_in.construct_circuit('vector'),
                evo_time=evo_time,
                num_time_slices=num_time_slices,
                expansion_mode=expansion_mode,
                expansion_order=expansion_order
            )
            quantum_registers = QuantumRegister(pauli_op.num_qubits, name='q')
            qc = QuantumCircuit(quantum_registers)
            qc += state_in.construct_circuit('circuit', quantum_registers)
            qc += pauli_op.copy().evolve(
                evo_time=evo_time,
                num_time_slices=num_time_slices,
                quantum_registers=quantum_registers,
                expansion_mode=expansion_mode,
                expansion_order=expansion_order,
            )
            state_out_circuit = self.quantum_instance_statevector.execute(qc).get_statevector(qc)

            self.log.debug('The fidelity between exact and matrix:   %s',
                           state_fidelity(state_out_exact, state_out_matrix))
            self.log.debug('The fidelity between exact and circuit:  %s',
                           state_fidelity(state_out_exact, state_out_circuit))
            f_mc = state_fidelity(state_out_matrix, state_out_circuit)
            self.log.debug('The fidelity between matrix and circuit: %s', f_mc)
            self.assertAlmostEqual(f_mc, 1)

    def test_simplification(self):
        """ Test Hamiltonians produce same result after simplification by constructor """
        q = QuantumRegister(2, name='q')
        qc = QuantumCircuit(q)
        qc.rx(10.9891251356965, 0)
        qc.rx(6.286692023269373, 1)
        qc.rz(7.848801398269382, 0)
        qc.rz(9.42477796076938, 1)
        qc.cx(0, 1)

        def eval_op(op):
            from qiskit import execute
            backend = BasicAer.get_backend('qasm_simulator')
            evaluation_circuits = op.construct_evaluation_circuit(qc, False)
            job = execute(evaluation_circuits, backend, shots=1024)
            return op.evaluate_with_result(job.result(), False)

        pauli_string = [[1.0, Pauli.from_label('XX')],
                        [-1.0, Pauli.from_label('YY')],
                        [-1.0, Pauli.from_label('ZZ')]]
        wpo = WeightedPauliOperator(pauli_string)
        expectation_value, _ = eval_op(wpo)
        self.assertAlmostEqual(expectation_value, -3.0, places=2)

        # Half each coefficient value but double up (6 Paulis total)
        pauli_string = [[0.5, Pauli.from_label('XX')],
                        [-0.5, Pauli.from_label('YY')],
                        [-0.5, Pauli.from_label('ZZ')]]
        pauli_string *= 2
        wpo2 = WeightedPauliOperator(pauli_string)
        expectation_value, _ = eval_op(wpo2)
        self.assertAlmostEqual(expectation_value, -3.0, places=2)


if __name__ == '__main__':
    unittest.main()
