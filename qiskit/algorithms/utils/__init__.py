# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Common Qiskit algorithms utility functions."""

from .validate_initial_point import validate_initial_point
from .validate_bounds import validate_bounds

__all__ = [
    "validate_initial_point",
    "validate_bounds",
]
