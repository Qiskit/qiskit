# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for handling linear reversible circuits."""

import copy
import numpy as np
from . import calc_inverse_matrix


def transpose_cx_circ(qc):
    """Transpose all cx gates in a circuit."""
    data = qc.data
    for i, _ in enumerate(data):
        if data[i][0].name == "cx" and data[i].operation.num_qubits == 2:
            data[i][1][0], data[i][1][1] = data[i][1][1], data[i][1][0]


def optimize_cx_4_options(function, mat, optimize_count=True):
    """Get best implementation of CX, implementing M,M^(-1),M^T,M^(-1)^T"""
    qc = function(mat)
    best_qc = qc
    best_depth = qc.depth()
    best_count = qc.count_ops()["cx"]

    for i in range(1, 4):
        mat_cpy = copy.deepcopy(mat)
        # i=1 inverse, i=2 transpose, i=3 transpose and inverse
        if i == 1:
            mat_cpy = calc_inverse_matrix(mat_cpy)
            qc = function(mat_cpy)
            qc = qc.inverse()
        elif i == 2:
            mat_cpy = np.transpose(mat_cpy)
            qc = function(mat_cpy)
            transpose_cx_circ(qc)
            qc = qc.inverse()
        elif i == 3:
            mat_cpy = calc_inverse_matrix(np.transpose(mat_cpy))
            qc = function(mat_cpy)
            transpose_cx_circ(qc)

        new_depth = qc.depth()
        new_count = qc.count_ops()["cx"]
        better_count = (optimize_count and best_count > new_count) or (
            not optimize_count and best_depth == new_depth and best_count > new_count
        )
        better_depth = (not optimize_count and best_depth > new_depth) or (
            optimize_count and best_count == new_count and best_depth > new_depth
        )

        if better_count or better_depth:
            best_count = new_count
            best_depth = new_depth
            best_qc = qc

    return best_qc
