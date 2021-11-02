# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Tests for error mitigation routines."""

import unittest
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase
from qiskit.result import Counts
from test.python.mitigation.generate_data import test_data
from qiskit.mitigation import CompleteReadoutMitigator
from qiskit.mitigation import TensoredReadoutMitigator
from qiskit.test.mock import FakeYorktown
from qiskit.quantum_info.operators.predicates import matrix_equal


@ddt
class TestReadoutMitigation(QiskitTestCase):
    def compare_results(self, res1, res2):
        """Compare the results between two runs"""
        res1_total_shots = sum(res1.values())
        res2_total_shots = sum(res2.values())
        keys = set(res1.keys()).union(set(res2.keys()))
        total = 0
        for key in keys:
            val1 = res1.get(key, 0) / res1_total_shots
            val2 = res2.get(key, 0) / res2_total_shots
            total += abs(val1 - val2) ** 2
        return total

    @data([test_data['test_1']])
    @unpack
    def test_mitigation_improvement(self, data):
        """Test whether readout mitigation led to more accurate results"""
        CRM = CompleteReadoutMitigator(data['complete_method_matrix'])
        TRM = TensoredReadoutMitigator(data['tensor_method_matrices'])
        mitigators = [CRM, TRM]
        for circuit_name, circuit_data in data['circuits'].items():
            counts_ideal = Counts(circuit_data['counts_ideal'])
            counts_noise = Counts(circuit_data['counts_noise'])
            unmitigated_error = self.compare_results(counts_ideal, counts_noise)
            for mitigator in mitigators:
                mitigated_probs = (
                    mitigator.quasi_probabilities(counts_noise)[0]
                    .nearest_probability_distribution()
                    .binary_probabilities()
                )
                mitigated_error = self.compare_results(counts_ideal, mitigated_probs)
                self.assertTrue(mitigated_error < unmitigated_error * 0.1, "Mitigator {} did not improve circuit {} measurements".format(mitigator, circuit_name))

    @data([test_data['test_1']])
    @unpack
    def test_clbits_parameter(self, data):
        """Test whether the clbits parameter is handled correctly"""
        # counts_ideal is {'000': 5000, '001': 5000}
        counts_ideal_12 = Counts({'00': 10000})
        counts_ideal_02 = Counts({'00': 5000, '01': 5000})
        counts_noise = Counts({'000': 4844, '001': 4962, '100': 56, '101': 65, '011': 37, '010': 35, '110': 1})
        CRM = CompleteReadoutMitigator(data['complete_method_matrix'])
        TRM = TensoredReadoutMitigator(data['tensor_method_matrices'])
        mitigators = [CRM, TRM]
        for mitigator in mitigators:
            mitigated_probs_12 = (
                mitigator.quasi_probabilities(counts_noise, qubits = [1,2], clbits=[1,2])[0]
                    .nearest_probability_distribution()
                    .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_12, mitigated_probs_12)
            self.assertTrue(mitigated_error < 0.001, "Mitigator {} did not correctly marganalize for qubits 1,2".format(mitigator))

            mitigated_probs_02 = (
                mitigator.quasi_probabilities(counts_noise, qubits = [0,2], clbits=[0, 2])[0]
                    .nearest_probability_distribution()
                    .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_02, mitigated_probs_02)
            self.assertTrue(mitigated_error < 0.001,
                            "Mitigator {} did not correctly marganalize for qubits 0,2".format(mitigator))

    @data([test_data['test_1']])
    @unpack
    def test_qubits_parameter(self, data):
        """Test whether the qubits parameter is handled correctly"""
        counts_ideal_012 = Counts({'000': 5000, '001': 5000})
        counts_ideal_210 = Counts({'000': 5000, '100': 5000})
        counts_ideal_102 = Counts({'000': 5000, '010': 5000})
        counts_noise = Counts({'000': 4844, '001': 4962, '100': 56, '101': 65, '011': 37, '010': 35, '110': 1})
        CRM = CompleteReadoutMitigator(data['complete_method_matrix'])
        TRM = TensoredReadoutMitigator(data['tensor_method_matrices'])
        mitigators = [CRM, TRM]
        for mitigator in mitigators:
            for mitigator in mitigators:
                mitigated_probs_012 = (
                    mitigator.quasi_probabilities(counts_noise, qubits=[0, 1, 2])[0]
                        .nearest_probability_distribution()
                        .binary_probabilities()
                )
                mitigated_error = self.compare_results(counts_ideal_012, mitigated_probs_012)
                self.assertTrue(mitigated_error < 0.001,
                                "Mitigator {} did not correctly handle qubit order 0, 1, 2".format(mitigator))

                mitigated_probs_210 = (
                    mitigator.quasi_probabilities(counts_noise, qubits=[2, 1, 0])[0]
                        .nearest_probability_distribution()
                        .binary_probabilities()
                )
                mitigated_error = self.compare_results(counts_ideal_210, mitigated_probs_210)
                self.assertTrue(mitigated_error < 0.001,
                                "Mitigator {} did not correctly handle qubit order 2, 1, 0".format(mitigator))

                mitigated_probs_102 = (
                    mitigator.quasi_probabilities(counts_noise, qubits=[1, 0, 2])[0]
                        .nearest_probability_distribution()
                        .binary_probabilities()
                )
                mitigated_error = self.compare_results(counts_ideal_102, mitigated_probs_102)
                self.assertTrue(mitigated_error < 0.001,
                                "Mitigator {} did not correctly handle qubit order 1, 0, 2".format(mitigator))
    def test_from_backend(self):
        """Test whether a tensored mitigator can be created directly from backend properties"""
        backend = FakeYorktown()
        num_qubits = len(backend.properties().qubits)
        rng = np.random.default_rng(42)
        probs = rng.random((num_qubits, 2))
        for qubit_idx, qubit_prop in enumerate(backend.properties().qubits):
            for prop in qubit_prop:
                if prop.name == "prob_meas1_prep0":
                    prop.value = probs[qubit_idx][0]
                if prop.name == "prob_meas0_prep1":
                    prop.value = probs[qubit_idx][1]
        TRM_from_backend = TensoredReadoutMitigator(backend=backend)

        mats = []
        for qubit_idx in range(num_qubits):
            mat = np.array([[1-probs[qubit_idx][0], probs[qubit_idx][1]],
                            [probs[qubit_idx][0], 1-probs[qubit_idx][1]]])
            mats.append(mat)
        TRM_from_matrices = TensoredReadoutMitigator(amats=mats)
        self.assertTrue(matrix_equal(TRM_from_backend.assignment_matrix(), TRM_from_matrices.assignment_matrix()))

if __name__ == "__main__":
    unittest.main()