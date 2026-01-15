# This code is part of Qiskit.
#
# (C) Copyright IBM 2017 - 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing cnot circuits"""

from .cnot_synth import synth_cnot_count_full_pmh
from .linear_depth_lnn import synth_cnot_depth_line_kms
from .linear_matrix_utils import (
    random_invertible_binary_matrix,
    calc_inverse_matrix,
    check_invertible_binary_matrix,
    binary_matmul,
)

# This is re-import is kept for compatibility with Terra 0.23. Eligible for deprecation in 0.25+.
# pylint: disable=cyclic-import,wrong-import-order
from qiskit.synthesis.linear_phase import synth_cnot_phase_aam as graysynth
