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
from numpy import array
from ddt import ddt, data, unpack
from qiskit import QiskitError
from qiskit.test import QiskitTestCase
from qiskit.result import Counts
from qiskit.result import CorrelatedReadoutMitigator
from qiskit.result import LocalReadoutMitigator
from qiskit.result.mitigation.utils import (
    z_diagonal,
    counts_probability_vector,
    str2diag,
    expval_with_stddev,
    stddev,
)
from qiskit.test.mock import FakeYorktown
from qiskit.quantum_info.operators.predicates import matrix_equal


@ddt
class TestReadoutMitigation(QiskitTestCase):
    """Tests for correlated and local readout mitigation."""

    test_data = {
        "test_1": {
            "local_method_matrices": [
                array([[0.996525, 0.002], [0.003475, 0.998]]),
                array([[0.991175, 0.00415], [0.008825, 0.99585]]),
                array([[0.9886, 0.00565], [0.0114, 0.99435]]),
            ],
            "correlated_method_matrix": array(
                [
                    [
                        9.771e-01,
                        1.800e-03,
                        4.600e-03,
                        0.000e00,
                        5.600e-03,
                        0.000e00,
                        0.000e00,
                        0.000e00,
                    ],
                    [
                        3.200e-03,
                        9.799e-01,
                        0.000e00,
                        3.400e-03,
                        0.000e00,
                        5.800e-03,
                        0.000e00,
                        1.000e-04,
                    ],
                    [
                        8.000e-03,
                        0.000e00,
                        9.791e-01,
                        2.400e-03,
                        1.000e-04,
                        0.000e00,
                        5.700e-03,
                        0.000e00,
                    ],
                    [
                        0.000e00,
                        8.300e-03,
                        3.200e-03,
                        9.834e-01,
                        0.000e00,
                        0.000e00,
                        0.000e00,
                        5.300e-03,
                    ],
                    [
                        1.170e-02,
                        0.000e00,
                        0.000e00,
                        0.000e00,
                        9.810e-01,
                        2.500e-03,
                        5.000e-03,
                        0.000e00,
                    ],
                    [
                        0.000e00,
                        9.900e-03,
                        0.000e00,
                        0.000e00,
                        3.900e-03,
                        9.823e-01,
                        0.000e00,
                        3.500e-03,
                    ],
                    [
                        0.000e00,
                        0.000e00,
                        1.310e-02,
                        0.000e00,
                        9.400e-03,
                        1.000e-04,
                        9.857e-01,
                        1.200e-03,
                    ],
                    [
                        0.000e00,
                        1.000e-04,
                        0.000e00,
                        1.080e-02,
                        0.000e00,
                        9.300e-03,
                        3.600e-03,
                        9.899e-01,
                    ],
                ]
            ),
            "num_qubits": 3,
            "shots": 10000,
            "circuits": {
                "ghz_3_qubits": {
                    "counts_ideal": {"111": 5000, "000": 5000},
                    "counts_noise": {
                        "111": 4955,
                        "000": 4886,
                        "001": 16,
                        "100": 46,
                        "010": 36,
                        "101": 23,
                        "011": 29,
                        "110": 9,
                    },
                },
                "first_qubit_h_3_qubits": {
                    "counts_ideal": {"000": 5000, "001": 5000},
                    "counts_noise": {
                        "000": 4844,
                        "001": 4962,
                        "100": 56,
                        "101": 65,
                        "011": 37,
                        "010": 35,
                        "110": 1,
                    },
                },
            },
        }
    }

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
            probs_noise = {
                key: value / circuits_data["shots"] for key, value in counts_noise.items()
            }
            unmitigated_error = self.compare_results(counts_ideal, counts_noise)
            # TODO: verify mitigated stddev is larger
            unmitigated_stddev = stddev(probs_noise, circuits_data["shots"])
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
                mitigated_stddev_upper_bound = mitigated_quasi_probs._stddev_upper_bound
                max_unmitigated_stddev = max(unmitigated_stddev.values())
                self.assertTrue(
                    mitigated_stddev_upper_bound >= max_unmitigated_stddev,
                    "Mitigator {} on circuit {} gave stddev upper bound {} "
                    "while unmitigated stddev maximum is {}".format(
                        mitigator,
                        circuit_name,
                        mitigated_stddev_upper_bound,
                        max_unmitigated_stddev,
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
        qubit_index = {i: i for i in range(num_qubits)}
        for circuit_name, circuit_data in circuits_data["circuits"].items():
            counts_ideal = Counts(circuit_data["counts_ideal"])
            counts_noise = Counts(circuit_data["counts_noise"])
            probs_ideal, _ = counts_probability_vector(counts_ideal, qubit_index=qubit_index)
            probs_noise, _ = counts_probability_vector(counts_noise, qubit_index=qubit_index)
            for diagonal in diagonals:
                if isinstance(diagonal, str):
                    diagonal = str2diag(diagonal)
                unmitigated_expectation, unmitigated_stddev = expval_with_stddev(
                    diagonal, probs_noise, shots=counts_noise.shots()
                )
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
        """Tests the order of mitigated qubits."""
        counts_ideal_012 = Counts({"000": 5000, "001": 5000})
        counts_ideal_210 = Counts({"000": 5000, "100": 5000})
        counts_noise = Counts(
            {"000": 4844, "001": 4962, "100": 56, "101": 65, "011": 37, "010": 35, "110": 1}
        )
        CRM = CorrelatedReadoutMitigator(
            circuits_data["correlated_method_matrix"], qubits=[0, 1, 2]
        )
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
                "Mitigator {} did not correctly handle qubit order 0,1,2 (the expected default)".format(
                    mitigator
                ),
            )

    @data([test_data["test_1"]])
    @unpack
    def test_qubits_subset_parameter(self, circuits_data):
        """Tests mitigation on a subset of the initial set of qubits."""
        counts_ideal_2 = Counts({"0": 5000, "1": 5000})
        counts_ideal_6 = Counts({"0": 10000})

        counts_noise = Counts(
            {"000": 4844, "001": 4962, "100": 56, "101": 65, "011": 37, "010": 35, "110": 1}
        )
        CRM = CorrelatedReadoutMitigator(
            circuits_data["correlated_method_matrix"], qubits=[2, 4, 6]
        )
        LRM = LocalReadoutMitigator(circuits_data["local_method_matrices"], qubits=[2, 4, 6])
        mitigators = [CRM, LRM]
        for mitigator in mitigators:
            mitigated_probs_2 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[2])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_2, mitigated_probs_2)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit subset".format(mitigator),
            )

            mitigated_probs_6 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[6])
                .nearest_probability_distribution()
                .binary_probabilities()
            )
            mitigated_error = self.compare_results(counts_ideal_6, mitigated_probs_6)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit subset".format(mitigator),
            )
            diagonal = str2diag("ZZ")
            ideal_expectation = 0
            mitigated_expectation, _ = mitigator.expectation_value(
                counts_noise, diagonal, qubits=[2, 6]
            )
            mitigated_error = np.abs(ideal_expectation - mitigated_expectation)
            self.assertTrue(
                mitigated_error < 0.1,
                "Mitigator {} did not improve circuit expectation".format(mitigator),
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

    def test_error_handling(self):
        """Test that the assignment matrices are valid."""
        bad_matrix_A = np.array([[-0.3, 1], [1.3, 0]])  # negative indices
        bad_matrix_B = np.array([[0.2, 1], [0.7, 0]])  # columns not summing to 1
        good_matrix_A = np.array([[0.2, 1], [0.8, 0]])
        for bad_matrix in [bad_matrix_A, bad_matrix_B]:
            with self.assertRaises(QiskitError) as cm:
                CorrelatedReadoutMitigator(bad_matrix)
            self.assertEqual(
                cm.exception.message,
                "Assignment matrix columns must be valid probability distributions",
            )

        with self.assertRaises(QiskitError) as cm:
            amats = [good_matrix_A, bad_matrix_A]
            LocalReadoutMitigator(amats)
        self.assertEqual(
            cm.exception.message,
            "Assignment matrix columns must be valid probability distributions",
        )

        with self.assertRaises(QiskitError) as cm:
            amats = [bad_matrix_B, good_matrix_A]
            LocalReadoutMitigator(amats)
        self.assertEqual(
            cm.exception.message,
            "Assignment matrix columns must be valid probability distributions",
        )


if __name__ == "__main__":
    unittest.main()
