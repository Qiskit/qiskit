# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing cnot circuits and cnot-phase circuit synthesize."""


from .graysynth import graysynth, synth_cnot_count_full_pmh
from .linear_matrix_utils import (
    random_invertible_binary_matrix,
    calc_inverse_matrix,
    check_invertible_binary_matrix,
)
