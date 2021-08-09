# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""SI unit utilities"""

import warnings
from typing import Tuple

import numpy as np


def apply_prefix(value: float, unit: str) -> float:
    """
    Given a SI unit prefix and value, apply the prefix to convert to
    standard SI unit.

    Args:
        value: The number to apply prefix to.
        unit: String prefix.

    Returns:
        Converted value.

    Raises:
        Exception: If the units aren't recognized.
    """
    downfactors = {"p": 1e12, "n": 1e9, "u": 1e6, "µ": 1e6, "m": 1e3}
    upfactors = {"k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    if not unit or len(unit) == 1:
        # "m" can represent meter
        return value
    if unit[0] in downfactors:
        return value / downfactors[unit[0]]
    elif unit[0] in upfactors:
        return value * upfactors[unit[0]]
    else:
        raise Exception(f"Could not understand units: {unit}")


def detach_prefix(value: float) -> Tuple[float, str]:
    """
    Given a SI unit value, find the most suitable auxiliary unit to scale the value.

    For example, the ``value = 1.3e8`` will be converted into a tuple of ``(130.0, "M")``,
    which represents a scaled value and auxiliary unit that may be used to display the value.
    In above example, that value might be displayed as ``130 MHz`` (unit is arbitrary here).

    Example:

        >>> value, prefix = detach_prefix(1e4)
        >>> print(f"{value} {prefix}Hz")
        10 kHz

    Args:
        value: The number to find prefix.

    Returns:
        A tuple of scaled value and prefix.
    """
    downfactors = ["p", "n", "μ", "m"]
    upfactors = ["k", "M", "G", "T"]

    if not value:
        return 0.0, ""

    try:
        fixed_point_3n = int(np.floor(np.log10(np.abs(value)) / 3))
        if fixed_point_3n != 0:
            if fixed_point_3n > 0:
                prefix = upfactors[fixed_point_3n - 1]
            else:
                prefix = downfactors[fixed_point_3n]
            scale = 10 ** (-3 * fixed_point_3n)
        else:
            prefix = ""
            scale = 1.0
    except IndexError:
        warnings.warn(f"The value {value} is out of range. Raw value is returned.", UserWarning)
        prefix = ""
        scale = 1.0

    return scale * value, prefix
