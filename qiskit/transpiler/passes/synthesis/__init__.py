# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing transpiler synthesis passes."""

from .unitary_synthesis import UnitarySynthesis
from .solovay_kitaev import SolovayKitaevDecomposition
from .solovay_kitaev_utils import (
    GateSequence, compute_frobenius_norm,
    compute_euler_angles_from_s03, 
    compute_su2_from_euler_angles, convert_su2_to_so3, 
    _compute_trace_so3, solve_decomposition_angle, 
    compute_rotation_between, _compute_commutator_so3,
    compute_rotation_from_angle_and_axis,
    compute_rotation_axis, convert_so3_to_su2
) 