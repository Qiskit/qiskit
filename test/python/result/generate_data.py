# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Data for readout mitigation tests"""
from numpy import array

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
