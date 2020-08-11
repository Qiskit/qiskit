# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Scheduling utility functions."""

# pylint: disable=invalid-name
from qiskit.util import deprecate_function
from qiskit.pulse import macros, utils

format_meas_map = deprecate_function(
    '"format_meas_map" has been moved to "qiskit.pulse.utils"')(utils.format_meas_map)

measure = deprecate_function(
    '"measure" has been moved to "qiskit.pulse.macros"')(macros.measure)

measure_all = deprecate_function(
    '"measure_all" has been moved to "qiskit.pulse.macros"')(macros.measure_all)
