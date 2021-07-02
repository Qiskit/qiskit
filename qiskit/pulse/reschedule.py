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

"""Deprecated import path. Use `pulse.transforms` instead."""
import warnings

# pylint: disable=unused-import
from qiskit.pulse.transforms import (
    align_measures,
    add_implicit_acquires,
    pad,
    compress_pulses,
)


warnings.warn(
    "The reschedule module has been renamed to transforms. This import path " "is deprecated."
)
