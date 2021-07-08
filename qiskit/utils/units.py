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
    upfactors = {"k": 1e3, "M": 1e6, "G": 1e9}
    if not unit:
        return value
    if unit[0] in downfactors:
        return value / downfactors[unit[0]]
    elif unit[0] in upfactors:
        return value * upfactors[unit[0]]
    else:
        raise Exception(f"Could not understand units: {unit}")
