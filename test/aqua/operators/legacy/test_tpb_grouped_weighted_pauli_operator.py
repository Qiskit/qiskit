# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test TPB Grouped WeightedPauliOperator """

import unittest
import itertools
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit import BasicAer
from qiskit.circuit.library import EfficientSU2
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.operators import (WeightedPauliOperator,
                                   TPBGroupedWeightedPauliOperator)
from qiskit.aqua.operators.legacy import op_converter


class TestTPBGroupedWeightedPauliOperator(QiskitAquaTestCase):
    """TPBGroupedWeightedPauliOperator tests."""

    def setUp(self):
        super().setUp()
        seed = 0
        aqua_globals.random_seed = seed

        self.num_qubits = 3
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=self.num_qubits)]
        weights = aqua_globals.random.random(len(paulis))
        self.qubit_op = WeightedPauliOperator.from_list(paulis, weights)
        self.var_form = EfficientSU2(self.qubit_op.num_qubits, reps=1)

        qasm_simulator = BasicAer.get_backend('qasm_simulator')
        self.quantum_instance_qasm = QuantumInstance(qasm_simulator, shots=65536,
                                                     seed_simulator=seed, seed_transpiler=seed)

        statevector_simulator = BasicAer.get_backend('statevector_simulator')
        self.quantum_instance_statevector = \
            QuantumInstance(statevector_simulator, shots=1,
                            seed_simulator=seed, seed_transpiler=seed)

    def test_sorted_grouping(self):
        """Test with color grouping approach."""
        num_qubits = 2
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=num_qubits)]
        weights = aqua_globals.random.random(len(paulis))
        op = WeightedPauliOperator.from_list(paulis, weights)
        grouped_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
            op, TPBGroupedWeightedPauliOperator.sorted_grouping)

        # check all paulis are still existed.
        for g_p in grouped_op.paulis:
            passed = False
            for pauli in op.paulis:
                if pauli[1] == g_p[1]:
                    passed = pauli[0] == g_p[0]
                    break
            self.assertTrue(passed,
                            "non-existed paulis in grouped_paulis: {}".format(g_p[1].to_label()))

        # check the number of basis of grouped
        # one should be less than and equal to the original one.
        self.assertGreaterEqual(len(op.basis), len(grouped_op.basis))

    def test_unsorted_grouping(self):
        """Test with normal grouping approach."""

        num_qubits = 4
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=num_qubits)]
        weights = aqua_globals.random.random(len(paulis))
        op = WeightedPauliOperator.from_list(paulis, weights)
        grouped_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
            op, TPBGroupedWeightedPauliOperator.unsorted_grouping)

        for g_p in grouped_op.paulis:
            passed = False
            for pauli in op.paulis:
                if pauli[1] == g_p[1]:
                    passed = pauli[0] == g_p[0]
                    break
            self.assertTrue(passed,
                            "non-existed paulis in grouped_paulis: {}".format(g_p[1].to_label()))

        self.assertGreaterEqual(len(op.basis), len(grouped_op.basis))

    def test_chop(self):
        """ chop test """
        paulis = [Pauli.from_label(x) for x in ['IIXX', 'ZZXX', 'ZZZZ', 'XXZZ', 'XXXX', 'IXXX']]
        coeffs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        op = WeightedPauliOperator.from_list(paulis, coeffs)
        grouped_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
            op, TPBGroupedWeightedPauliOperator.sorted_grouping)

        original_num_basis = len(grouped_op.basis)
        chopped_grouped_op = grouped_op.chop(0.35, copy=True)
        self.assertLessEqual(len(chopped_grouped_op.basis), 3)
        self.assertLessEqual(len(chopped_grouped_op.basis), original_num_basis)
        # ZZXX group is remove
        for b, _ in chopped_grouped_op.basis:
            self.assertFalse(b.to_label() == 'ZZXX')

        chopped_grouped_op = grouped_op.chop(0.55, copy=True)
        self.assertLessEqual(len(chopped_grouped_op.basis), 1)
        self.assertLessEqual(len(chopped_grouped_op.basis), original_num_basis)

        for b, _ in chopped_grouped_op.basis:
            self.assertFalse(b.to_label() == 'ZZXX')
            self.assertFalse(b.to_label() == 'ZZZZ')
            self.assertFalse(b.to_label() == 'XXZZ')

    def test_evaluate_qasm_mode(self):
        """ evaluate qasm mode test """
        wave_function = self.var_form.assign_parameters(
            np.array(aqua_globals.random.standard_normal(self.var_form.num_parameters)))
        wave_fn_statevector = \
            self.quantum_instance_statevector.execute(wave_function).get_statevector(wave_function)
        reference = self.qubit_op.copy().evaluate_with_statevector(wave_fn_statevector)

        shots = 65536 // len(self.qubit_op.paulis)
        self.quantum_instance_qasm.set_config(shots=shots)
        circuits = self.qubit_op.construct_evaluation_circuit(wave_function=wave_function,
                                                              statevector_mode=False)
        result = self.quantum_instance_qasm.execute(circuits)
        pauli_value = self.qubit_op.evaluate_with_result(result=result, statevector_mode=False)
        grouped_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
            self.qubit_op, TPBGroupedWeightedPauliOperator.sorted_grouping)
        shots = 65536 // grouped_op.num_groups
        self.quantum_instance_qasm.set_config(shots=shots)
        circuits = grouped_op.construct_evaluation_circuit(wave_function=wave_function,
                                                           statevector_mode=False)
        grouped_pauli_value = grouped_op.evaluate_with_result(
            result=self.quantum_instance_qasm.execute(circuits), statevector_mode=False)

        self.assertGreaterEqual(reference[0].real,
                                grouped_pauli_value[0].real - 3 * grouped_pauli_value[1].real)
        self.assertLessEqual(reference[0].real,
                             grouped_pauli_value[0].real + 3 * grouped_pauli_value[1].real)
        # this check assure the std of grouped pauli is
        # less than pauli mode under a fixed amount of total shots
        self.assertLessEqual(grouped_pauli_value[1].real, pauli_value[1].real)

    def test_equal(self):
        """ equal test """
        gop_1 = op_converter.to_tpb_grouped_weighted_pauli_operator(
            self.qubit_op, TPBGroupedWeightedPauliOperator.sorted_grouping)
        gop_2 = op_converter.to_tpb_grouped_weighted_pauli_operator(
            self.qubit_op, TPBGroupedWeightedPauliOperator.unsorted_grouping)

        self.assertEqual(gop_1, gop_2)


if __name__ == '__main__':
    unittest.main()
