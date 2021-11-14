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
from test.python.mitigation.generate_data import test_data
import numpy as np
from ddt import ddt, data, unpack
from qiskit.test import QiskitTestCase
from qiskit.result import Counts
from qiskit.mitigation import CorrelatedReadoutMitigator
from qiskit.mitigation import LocalReadoutMitigator
from qiskit.mitigation.utils import (
    z_diagonal,
    counts_probability_vector,
    str2diag,
    expval_with_stddev,
)
from qiskit.test.mock import FakeYorktown
from qiskit.quantum_info.operators.predicates import matrix_equal


@ddt
class TestReadoutMitigation(QiskitTestCase):
    """Tests for correlated and local readout mitigation."""

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

    @data([test_data["test_1"]])
    @unpack
    def test_mitigation_improvement(self, circuits_data):
        """Test whether readout mitigation led to more accurate results"""
        CRM = CorrelatedReadoutMitigator(circuits_data["correlated_method_matrix"])
        LRM = LocalReadoutMitigator(circuits_data["local_method_matrices"])
        mitigators = [CRM, LRM]
        for circuit_name, circuit_data in circuits_data["circuits"].items():
            counts_ideal = Counts(circuit_data["counts_ideal"])
            counts_noise = Counts(circuit_data["counts_noise"])
            unmitigated_error = self.compare_results(counts_ideal, counts_noise)
            # TODO: verify mitigated stddev is larger
            # unmitigated_stddev =
            for mitigator in mitigators:
                mitigated_quasi_probs = mitigator.quasi_probabilities(counts_noise)
                mitigated_probs = (
                    mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities()
                )
                mitigated_error = self.compare_results(counts_ideal, mitigated_probs)
                self.assertTrue(
                    mitigated_error < unmitigated_error * 0.8,
                    "Mitigator {} did not improve circuit {} measurements".format(
                        mitigator, circuit_name
                    ),
                )

    @data([test_data["test_1"]])
    @unpack
    def test_expectation_improvement(self, circuits_data):
        """Test whether readout mitigation led to more accurate results
        and that its standard deviation is increased"""
        CRM = CorrelatedReadoutMitigator(circuits_data["correlated_method_matrix"])
        LRM = LocalReadoutMitigator(circuits_data["local_method_matrices"])
        num_qubits = circuits_data["num_qubits"]
        diagonals = []
        diagonals.append(z_diagonal(2 ** num_qubits))
        diagonals.append("IZ0")
        diagonals.append("ZZZ")
        diagonals.append("101")
        diagonals.append("IZI")
        mitigators = [CRM, LRM]
        for circuit_name, circuit_data in circuits_data["circuits"].items():
            counts_ideal = Counts(circuit_data["counts_ideal"])
            counts_noise = Counts(circuit_data["counts_noise"])
            probs_ideal = counts_probability_vector(counts_ideal)
            probs_noise = counts_probability_vector(counts_noise)
            for diagonal in diagonals:
                if isinstance(diagonal, str):
                    diagonal = str2diag(diagonal)
                unmitigated_expectation, unmitigated_stddev = expval_with_stddev(
                    diagonal, probs_noise, shots=counts_noise.shots()
                )  # np.dot(probs_noise, diagonal)
                ideal_expectation = np.dot(probs_ideal, diagonal)
                unmitigated_error = np.abs(ideal_expectation - unmitigated_expectation)
                for mitigator in mitigators:
                    mitigated_expectation, mitigated_stddev = mitigator.expectation_value(
                        counts_noise, diagonal
                    )
                    mitigated_error = np.abs(ideal_expectation - mitigated_expectation)
                    self.assertTrue(
                        mitigated_error < unmitigated_error,
                        "Mitigator {} did not improve circuit {} measurements".format(
                            mitigator, circuit_name
                        ),
                    )
                    self.assertTrue(
                        mitigated_stddev >= unmitigated_stddev,
                        "Mitigator {} did not increase circuit {} the standard deviation".format(
                            mitigator, circuit_name
                        ),
                    )

    @data([test_data["test_1"]])
    @unpack
    def test_clbits_parameter(self, circuits_data):
        """Test whether the clbits parameter is handled correctly"""
        # counts_ideal is {'000': 5000, '001': 5000}
        counts_ideal_12 = Counts({"00": 10000})
        counts_ideal_02 = Counts({"00": 5000, "01": 5000})
        counts_noise = Counts(
            {"000": 4844, "001": 4962, "100": 56, "101": 65, "011": 37, "010": 35, "110": 1}
        )
        CRM = CorrelatedReadoutMitigator(circuits_data["correlated_method_matrix"])
        LRM = LocalReadoutMitigator(circuits_data["local_method_matrices"])
        mitigators = [CRM, LRM]
        for mitigator in mitigators:
            mitigated_probs_12 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[1, 2], clbits=[1, 2])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_12, mitigated_probs_12)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly marganalize for qubits 1,2".format(mitigator),
            )

            mitigated_probs_02 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[0, 2], clbits=[0, 2])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_02, mitigated_probs_02)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly marganalize for qubits 0,2".format(mitigator),
            )

    @data([test_data["test_1"]])
    @unpack
    def test_qubits_parameter(self, circuits_data):
        """Test whether the qubits parameter is handled correctly"""
        counts_ideal_012 = Counts({"000": 5000, "001": 5000})
        counts_ideal_210 = Counts({"000": 5000, "100": 5000})
        counts_ideal_102 = Counts({"000": 5000, "010": 5000})
        counts_noise = Counts(
            {"000": 4844, "001": 4962, "100": 56, "101": 65, "011": 37, "010": 35, "110": 1}
        )
        CRM = CorrelatedReadoutMitigator(circuits_data["correlated_method_matrix"])
        LRM = LocalReadoutMitigator(circuits_data["local_method_matrices"])
        mitigators = [CRM, LRM]
        for mitigator in mitigators:
            mitigated_probs_012 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[0, 1, 2])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_012, mitigated_probs_012)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 0, 1, 2".format(mitigator),
            )

            mitigated_probs_210 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[2, 1, 0])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_210, mitigated_probs_210)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 2, 1, 0".format(mitigator),
            )

            mitigated_probs_102 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[1, 0, 2])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_102, mitigated_probs_102)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 1, 0, 2".format(mitigator),
            )

    @data([test_data["test_1"]])
    @unpack
    def test_repeated_qubits_parameter(self, circuits_data):
        counts_ideal_012 = Counts({"000": 5000, "001": 5000})
        counts_ideal_210 = Counts({"000": 5000, "100": 5000})
        counts_noise = Counts(
            {"000": 4844, "001": 4962, "100": 56, "101": 65, "011": 37, "010": 35, "110": 1}
        )
        CRM = CorrelatedReadoutMitigator(circuits_data["correlated_method_matrix"], qubits=[0, 1, 2])
        LRM = LocalReadoutMitigator(circuits_data["local_method_matrices"], qubits=[0, 1, 2])
        mitigators = [CRM, LRM]
        for mitigator in mitigators:
            mitigated_probs_210 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[2, 1, 0])
                    .nearest_probability_distribution()
                    .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_210, mitigated_probs_210)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 2,1,0".format(mitigator),
            )

            # checking qubit order 2,1,0 should not "overwrite" the default 0,1,2
            mitigated_probs_012 = (
                mitigator.quasi_probabilities(counts_noise)
                    .nearest_probability_distribution()
                    .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_012, mitigated_probs_012)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 0,1,2 (the expected default)".format(mitigator),
            )

    def test_from_backend(self):
        """Test whether a local mitigator can be created directly from backend properties"""
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
        LRM_from_backend = LocalReadoutMitigator(backend=backend)

        mats = []
        for qubit_idx in range(num_qubits):
            mat = np.array(
                [
                    [1 - probs[qubit_idx][0], probs[qubit_idx][1]],
                    [probs[qubit_idx][0], 1 - probs[qubit_idx][1]],
                ]
            )
            mats.append(mat)
        LRM_from_matrices = LocalReadoutMitigator(amats=mats)
        self.assertTrue(
            matrix_equal(
                LRM_from_backend.assignment_matrix(), LRM_from_matrices.assignment_matrix()
            )
        )


if __name__ == "__main__":
    unittest.main()
