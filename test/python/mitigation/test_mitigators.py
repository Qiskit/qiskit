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
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase
from qiskit.result import Counts
from test.python.mitigation.generate_data import test_data
from qiskit.mitigation import CompleteReadoutMitigator
from qiskit.mitigation import TensoredReadoutMitigator

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
        for circuit in data['circuits']:
            counts_ideal = Counts(circuit['counts_ideal'])
            counts_noise = Counts(circuit['counts_noise'])
            unmitigated_error = self.compare_results(counts_ideal, counts_noise)
            for mitigator in mitigators:
                mitigated_probs = (
                    mitigator.quasi_probabilities(counts_noise)[0]
                    .nearest_probability_distribution()
                    .binary_probabilities()
                )
                mitigated_error = self.compare_results(counts_ideal, mitigated_probs)
                self.assertTrue(mitigated_error < unmitigated_error * 0.1)

if __name__ == "__main__":
    unittest.main()
