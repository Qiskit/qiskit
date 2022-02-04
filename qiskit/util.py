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

"""Common utilities for Qiskit."""

import warnings

from qiskit.utils.deprecation import deprecate_arguments
from qiskit.utils.deprecation import deprecate_function
from qiskit.utils.multiprocessing import is_main_process
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit.utils.units import apply_prefix

__all__ = [
    "deprecate_arguments",
    "deprecate_function",
    "is_main_process",
    "local_hardware_info",
    "apply_prefix",
]

warnings.warn(
    "The 'qiskit.util' namespace is deprecated since qiskit-terra 0.17 and will be removed in 0.20."
    " It has been renamed to 'qiskit.utils'.",
    category=DeprecationWarning,
)
