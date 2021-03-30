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

"""Utility passes used for other main passes."""

from .check_map import CheckMap
from .check_cx_direction import CheckCXDirection  # Deprecated
from .cx_direction import CXDirection  # Deprecated
from .check_gate_direction import CheckGateDirection
from .gate_direction import GateDirection
from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements
from .remove_final_measurements import RemoveFinalMeasurements
from .merge_adjacent_barriers import MergeAdjacentBarriers
from .dag_fixed_point import DAGFixedPoint
from .fixed_point import FixedPoint
from .error import Error
