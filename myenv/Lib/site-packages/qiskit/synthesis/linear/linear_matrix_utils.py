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

"""Utility functions for handling binary matrices."""

# pylint: disable=unused-import
from qiskit._accelerate.synthesis.linear import (
    gauss_elimination,
    gauss_elimination_with_perm,
    compute_rank_after_gauss_elim,
    compute_rank,
    calc_inverse_matrix,
    binary_matmul,
    random_invertible_binary_matrix,
    check_invertible_binary_matrix,
    row_op,
    col_op,
)
