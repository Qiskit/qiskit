# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities for handling duration of a circuit instruction.
"""
import warnings


def duration_in_dt(duration_in_sec: float, dt_in_sec: float) -> int:
    """
    Return duration in dt.

    Args:
        duration_in_sec: duration [s] to be converted.
        dt_in_sec: duration of dt in seconds used for conversion.

    Returns:
        Duration in dt.
    """
    res = round(duration_in_sec / dt_in_sec)
    rounding_error = abs(duration_in_sec - res * dt_in_sec)
    if rounding_error > 1e-15:
        warnings.warn("Duration is rounded to %d [dt] = %e [s] from %e [s]"
                      % (res, res * dt_in_sec, duration_in_sec),
                      UserWarning)
    return res
