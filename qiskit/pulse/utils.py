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

"""
Pulse utilities.
"""
import warnings

# pylint: disable=unused-argument


def align_measures(schedules, cmd_def, cal_gate, max_calibration_duration=None, align_time=None):
    """
    This function has been moved!
    """
    warnings.warn("The function `align_measures` has been moved to qiskit.pulse.reschedule. "
                  "It cannot be invoked from `utils` anymore (this call returns None).")


def add_implicit_acquires(schedule, meas_map):
    """
    This function has been moved!
    """
    warnings.warn("The function `add_implicit_acquires` has been moved to qiskit.pulse.reschedule."
                  " It cannot be invoked from `utils` anymore (this call returns None).")


def pad(schedule, channels=None, until=None):
    """
    This function has been moved!
    """
    warnings.warn("The function `pad` has been moved to qiskit.pulse.reschedule. It cannot be "
                  "invoked from `utils` anymore (this call returns None).")
