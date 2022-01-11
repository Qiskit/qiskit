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
from collections import Counter
import numpy as np
from qiskit import QiskitError
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Statevector
from qiskit.result import Counts
from qiskit.result import CorrelatedReadoutMitigator
from qiskit.result import LocalReadoutMitigator
from qiskit.result.utils import marginal_counts
from qiskit.result.mitigation.utils import (
    z_diagonal,
    counts_probability_vector,
    str2diag,
    expval_with_stddev,
    stddev,
)
from qiskit.test.mock import FakeYorktown
from qiskit.quantum_info.operators.predicates import matrix_equal


class TestReadoutMitigation(QiskitTestCase):
    """Tests for correlated and local readout mitigation."""

    # test_data = {
    #     "test_1": {
    #         "local_method_matrices": [
    #             array([[0.996525, 0.002], [0.003475, 0.998]]),
    #             array([[0.991175, 0.00415], [0.008825, 0.99585]]),
    #             array([[0.9886, 0.00565], [0.0114, 0.99435]]),
    #         ],
    #         "correlated_method_matrix": array(
    #             [
    #                 [
    #                     9.771e-01,
    #                     1.800e-03,
    #                     4.600e-03,
    #                     0.000e00,
    #                     5.600e-03,
    #                     0.000e00,
    #                     0.000e00,
    #                     0.000e00,
    #                 ],
    #                 [
    #                     3.200e-03,
    #                     9.799e-01,
    #                     0.000e00,
    #                     3.400e-03,
    #                     0.000e00,
    #                     5.800e-03,
    #                     0.000e00,
    #                     1.000e-04,
    #                 ],
    #                 [
    #                     8.000e-03,
    #                     0.000e00,
    #                     9.791e-01,
    #                     2.400e-03,
    #                     1.000e-04,
    #                     0.000e00,
    #                     5.700e-03,
    #                     0.000e00,
    #                 ],
    #                 [
    #                     0.000e00,
    #                     8.300e-03,
    #                     3.200e-03,
    #                     9.834e-01,
    #                     0.000e00,
    #                     0.000e00,
    #                     0.000e00,
    #                     5.300e-03,
    #                 ],
    #                 [
    #                     1.170e-02,
    #                     0.000e00,
    #                     0.000e00,
    #                     0.000e00,
    #                     9.810e-01,
    #                     2.500e-03,
    #                     5.000e-03,
    #                     0.000e00,
    #                 ],
    #                 [
    #                     0.000e00,
    #                     9.900e-03,
    #                     0.000e00,
    #                     0.000e00,
    #                     3.900e-03,
    #                     9.823e-01,
    #                     0.000e00,
    #                     3.500e-03,
    #                 ],
    #                 [
    #                     0.000e00,
    #                     0.000e00,
    #                     1.310e-02,
    #                     0.000e00,
    #                     9.400e-03,
    #                     1.000e-04,
    #                     9.857e-01,
    #                     1.200e-03,
    #                 ],
    #                 [
    #                     0.000e00,
    #                     1.000e-04,
    #                     0.000e00,
    #                     1.080e-02,
    #                     0.000e00,
    #                     9.300e-03,
    #                     3.600e-03,
    #                     9.899e-01,
    #                 ],
    #             ]
    #         ),
    #         "num_qubits": 3,
    #         "shots": 10000,
    #         "circuits": {
    #             "ghz_3_qubits": {
    #                 "counts_ideal": {"111": 5000, "000": 5000},
    #                 "counts_noise": {
    #                     "111": 4955,
    #                     "000": 4886,
    #                     "001": 16,
    #                     "100": 46,
    #                     "010": 36,
    #                     "101": 23,
    #                     "011": 29,
    #                     "110": 9,
    #                 },
    #             },
    #             "first_qubit_h_3_qubits": {
    #                 "counts_ideal": {"000": 5000, "001": 5000},
    #                 "counts_noise": {
    #                     "000": 4844,
    #                     "001": 4962,
    #                     "100": 56,
    #                     "101": 65,
    #                     "011": 37,
    #                     "010": 35,
    #                     "110": 1,
    #                 },
    #             },
    #         },
    #     }
    # }
    rng = np.random.default_rng(42)

    @staticmethod
    def compare_results(res1, res2):
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

    @staticmethod
    def mitigators(assignment_matrices, qubits=None):
        full_assignment_matrix = assignment_matrices[0]
        for m in assignment_matrices[1:]:
            full_assignment_matrix = np.kron(full_assignment_matrix, m)
        CRM = CorrelatedReadoutMitigator(full_assignment_matrix, qubits)
        LRM = LocalReadoutMitigator(assignment_matrices, qubits)
        mitigators = [CRM, LRM]
        return mitigators

    # @staticmethod
    # def prepare_test_data(assignment_matrices, circuits, shots=1024):
    #     full_assignment_matrix = assignment_matrices[0]
    #     for m in assignment_matrices[1:]:
    #         full_assignment_matrix = full_assignment_matrix.kron(m)
    #     num_qubits = len(assignment_matrices)
    #     ideal_assignment_matrix = np.eye(2**num_qubits)
    #     circuit_results = {}
    #     for (circuit_name, circuit) in circuits.items():
    #         result = {
    #             "counts_ideal": TestReadoutMitigation.simulate_circuit(circuit,
    #                                                   ideal_assignment_matrix,
    #                                                   num_qubits, shots),
    #             "counts_noise": TestReadoutMitigation.simulate_circuit(circuit, full_assignment_matrix, num_qubits, shots)
    #         }
    #         circuit_results[circuit_name] = result
    #
    #     test_data = {
    #         "local_method_matrices": assignment_matrices,
    #         "correlated_method_matrix": full_assignment_matrix,
    #         "num_qubits": num_qubits,
    #         "shots": shots,
    #         "circuits": circuit_results
    #     }
    #     return test_data

    @staticmethod
    def simulate_circuit(circuit, assignment_matrix, num_qubits, shots=1024):
        probs = Statevector.from_instruction(circuit).probabilities()
        noisy_probs = assignment_matrix @ probs
        labels = [bin(a)[2:].zfill(num_qubits) for a in range(2 ** num_qubits)]
        results = TestReadoutMitigation.rng.choice(labels, size=shots, p=noisy_probs)
        return Counts(dict(Counter(results)))

    @staticmethod
    def ghz_3_circuit():
        c = QuantumCircuit(3)
        c.h(0)
        c.cx(0, 1)
        c.cx(1, 2)
        return (c, "ghz_3_ciruit", 3)

    @staticmethod
    def first_qubit_h_3_circuit():
        c = QuantumCircuit(3)
        c.h(0)
        return (c, "first_qubit_h_3_circuit", 3)

    @staticmethod
    def assignment_matrices():
        result = []
        # from FakeYorktown
        assignment_matrices = [
            np.array([[0.98828125, 0.04003906], [0.01171875, 0.95996094]]),
            np.array([[0.99023438, 0.02929688], [0.00976562, 0.97070312]]),
            np.array([[0.984375, 0.02441406], [0.015625, 0.97558594]]),
        ]
        return assignment_matrices
        # result.append((assignment_matrices, 3))

    # @staticmethod
    # def test_data_list():
    #     result = []
    #     for ((c,c_qubits), (m, m_qubits)) in itertools.product(TestReadoutMitigation.circuits(),
    #                                                            TestReadoutMitigation.assignment_matrices()):
    #         if c_qubits == m_qubits:
    #             result.append(TestReadoutMitigation.prepare_test_data(m,c))
    #     return result

    @staticmethod
    def counts_data(circuit, assignment_matrices, shots=1024):
        full_assignment_matrix = assignment_matrices[0]
        for m in assignment_matrices[1:]:
            full_assignment_matrix = np.kron(full_assignment_matrix, m)
        num_qubits = len(assignment_matrices)
        ideal_assignment_matrix = np.eye(2 ** num_qubits)
        counts_ideal = TestReadoutMitigation.simulate_circuit(
            circuit, ideal_assignment_matrix, num_qubits, shots
        )
        counts_noise = TestReadoutMitigation.simulate_circuit(
            circuit, full_assignment_matrix, num_qubits, shots
        )
        probs_noise = {key: value / shots for key, value in counts_noise.items()}
        return counts_ideal, counts_noise, probs_noise

    def test_mitigation_improvement(self):
        """Test whether readout mitigation led to more accurate results"""
        shots = 1024
        assignment_matrices = self.assignment_matrices()
        num_qubits = len(assignment_matrices)
        mitigators = self.mitigators(assignment_matrices)
        circuit, circuit_name, num_qubits = self.ghz_3_circuit()
        counts_ideal, counts_noise, probs_noise = self.counts_data(
            circuit, assignment_matrices, shots
        )
        unmitigated_error = self.compare_results(counts_ideal, counts_noise)
        unmitigated_stddev = stddev(probs_noise, shots)

        for mitigator in mitigators:
            mitigated_quasi_probs = mitigator.quasi_probabilities(counts_noise)
            mitigated_probs = (
                mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities(
                    num_bits=num_qubits
                )
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

    def test_expectation_improvement(self):
        """Test whether readout mitigation led to more accurate results
        and that its standard deviation is increased"""
        shots = 1024
        assignment_matrices = self.assignment_matrices()
        mitigators = self.mitigators(assignment_matrices)
        num_qubits = len(assignment_matrices)
        diagonals = []
        diagonals.append(z_diagonal(2 ** num_qubits))
        diagonals.append("IZ0")
        diagonals.append("ZZZ")
        diagonals.append("101")
        diagonals.append("IZZ")
        qubit_index = {i: i for i in range(num_qubits)}
        circuit, circuit_name, num_qubits = self.ghz_3_circuit()
        counts_ideal, counts_noise, _ = self.counts_data(circuit, assignment_matrices, shots)
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
                    "Mitigator {} did not improve circuit {} expectation computation for diagonal {} "
                    "ideal: {}, unmitigated: {} mitigated: {}".format(
                        mitigator,
                        circuit_name,
                        diagonal,
                        ideal_expectation,
                        unmitigated_expectation,
                        mitigated_expectation,
                    ),
                )
                self.assertTrue(
                    mitigated_stddev >= unmitigated_stddev,
                    "Mitigator {} did not increase circuit {} the standard deviation".format(
                        mitigator, circuit_name
                    ),
                )

    def test_clbits_parameter(self):
        """Test whether the clbits parameter is handled correctly"""
        shots = 10000
        assignment_matrices = self.assignment_matrices()
        mitigators = self.mitigators(assignment_matrices)
        circuit, circuit_name, num_qubits = self.first_qubit_h_3_circuit()
        counts_ideal, counts_noise, _ = self.counts_data(circuit, assignment_matrices, shots)
        counts_ideal_12 = marginal_counts(counts_ideal, [1, 2])
        counts_ideal_02 = marginal_counts(counts_ideal, [0, 2])

        for mitigator in mitigators:
            mitigated_probs_12 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[1, 2], clbits=[1, 2])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=2)
            )
            mitigated_error = self.compare_results(counts_ideal_12, mitigated_probs_12)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly marganalize for qubits 1,2".format(mitigator),
            )

            mitigated_probs_02 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[0, 2], clbits=[0, 2])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=2)
            )
            mitigated_error = self.compare_results(counts_ideal_02, mitigated_probs_02)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly marganalize for qubits 0,2".format(mitigator),
            )

    def test_qubits_parameter(self):
        """Test whether the qubits parameter is handled correctly"""
        shots = 10000
        assignment_matrices = self.assignment_matrices()
        mitigators = self.mitigators(assignment_matrices)
        circuit, circuit_name, num_qubits = self.first_qubit_h_3_circuit()
        counts_ideal, counts_noise, _ = self.counts_data(circuit, assignment_matrices, shots)
        counts_ideal_012 = counts_ideal
        counts_ideal_210 = Counts({"000": counts_ideal["000"], "100": counts_ideal["001"]})
        counts_ideal_102 = Counts({"000": counts_ideal["000"], "010": counts_ideal["001"]})

        for mitigator in mitigators:
            mitigated_probs_012 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[0, 1, 2])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=3)
            )
            mitigated_error = self.compare_results(counts_ideal_012, mitigated_probs_012)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 0, 1, 2".format(mitigator),
            )

            mitigated_probs_210 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[2, 1, 0])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=3)
            )
            mitigated_error = self.compare_results(counts_ideal_210, mitigated_probs_210)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 2, 1, 0".format(mitigator),
            )

            mitigated_probs_102 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[1, 0, 2])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=3)
            )
            mitigated_error = self.compare_results(counts_ideal_102, mitigated_probs_102)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 1, 0, 2".format(mitigator),
            )

    def test_repeated_qubits_parameter(self):
        """Tests the order of mitigated qubits."""
        shots = 10000
        assignment_matrices = self.assignment_matrices()
        mitigators = self.mitigators(assignment_matrices, qubits=[0, 1, 2])
        circuit, circuit_name, num_qubits = self.first_qubit_h_3_circuit()
        counts_ideal, counts_noise, _ = self.counts_data(circuit, assignment_matrices, shots)
        counts_ideal_012 = counts_ideal
        counts_ideal_210 = Counts({"000": counts_ideal["000"], "100": counts_ideal["001"]})

        for mitigator in mitigators:
            mitigated_probs_210 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[2, 1, 0])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=3)
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
                .binary_probabilities(num_bits=3)
            )
            mitigated_error = self.compare_results(counts_ideal_012, mitigated_probs_012)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit order 0,1,2 (the expected default)".format(
                    mitigator
                ),
            )

    def test_qubits_subset_parameter(self):
        """Tests mitigation on a subset of the initial set of qubits."""

        shots = 10000
        assignment_matrices = self.assignment_matrices()
        mitigators = self.mitigators(assignment_matrices, qubits=[2, 4, 6])
        circuit, circuit_name, num_qubits = self.first_qubit_h_3_circuit()
        counts_ideal, counts_noise, _ = self.counts_data(circuit, assignment_matrices, shots)
        counts_ideal_2 = marginal_counts(counts_ideal, [0])
        counts_ideal_6 = marginal_counts(counts_ideal, [2])

        for mitigator in mitigators:
            mitigated_probs_2 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[2])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=1)
            )
            mitigated_error = self.compare_results(counts_ideal_2, mitigated_probs_2)
            self.assertTrue(
                mitigated_error < 0.001,
                "Mitigator {} did not correctly handle qubit subset".format(mitigator),
            )

            mitigated_probs_6 = (
                mitigator.quasi_probabilities(counts_noise, qubits=[6])
                .nearest_probability_distribution()
                .binary_probabilities(num_bits=1)
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


class TestLocalReadoutMitigation(QiskitTestCase):
    def test_assignment_matrix(self):
        qubits = [0, 2, 3]
        assignment_matrices = [
            np.array([[0.98828125, 0.04003906], [0.01171875, 0.95996094]]),
            np.array([[0.99023438, 0.02929688], [0.00976562, 0.97070312]]),
            np.array([[0.984375, 0.02441406], [0.015625, 0.97558594]]),
        ]
        expected_mitigation_matrix = np.array(
            [
                [
                    1.03929190e00,
                    -4.33478790e-02,
                    -3.13669642e-02,
                    1.30828632e-03,
                    -2.60083030e-02,
                    1.08478164e-03,
                    7.84958982e-04,
                    -3.27398944e-05,
                ],
                [
                    -1.26871849e-02,
                    1.06995259e00,
                    3.82913092e-04,
                    -3.22923375e-02,
                    3.17497086e-04,
                    -2.67755876e-02,
                    -9.58240871e-06,
                    8.08116468e-04,
                ],
                [
                    -1.04556476e-02,
                    4.36095141e-04,
                    1.06020322e00,
                    -4.42200702e-02,
                    2.61652816e-04,
                    -1.09132907e-05,
                    -2.65316092e-02,
                    1.10660825e-03,
                ],
                [
                    1.27637610e-04,
                    -1.07641051e-02,
                    -1.29424604e-02,
                    1.09148083e00,
                    -3.19413406e-06,
                    2.69371972e-04,
                    3.23885361e-04,
                    -2.73143321e-02,
                ],
                [
                    -1.66453157e-02,
                    6.94260323e-04,
                    5.02373800e-04,
                    -2.09535346e-05,
                    1.04865489e00,
                    -4.37384003e-02,
                    -3.16495494e-02,
                    1.32007268e-03,
                ],
                [
                    2.03198156e-04,
                    -1.71363778e-02,
                    -6.13274220e-06,
                    5.17194593e-04,
                    -1.28014838e-02,
                    1.07959180e00,
                    3.86362759e-04,
                    -3.25832593e-02,
                ],
                [
                    1.67457819e-04,
                    -6.98450675e-06,
                    -1.69802316e-02,
                    7.08229351e-04,
                    -1.05498426e-02,
                    4.40023925e-04,
                    1.06975459e00,
                    -4.46184491e-02,
                ],
                [
                    -2.04424601e-06,
                    1.72398080e-04,
                    2.07286652e-04,
                    -1.74811743e-02,
                    1.28787498e-04,
                    -1.08610790e-02,
                    -1.30590591e-02,
                    1.10131398e00,
                ],
            ]
        )
        LRM = LocalReadoutMitigator(assignment_matrices, qubits)
        self.assertTrue(matrix_equal(expected_mitigation_matrix, LRM.mitigation_matrix()))


if __name__ == "__main__":
    unittest.main()
