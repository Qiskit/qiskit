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
from qiskit.utils import __all__
from qiskit.utils import *  # pylint: disable=wildcard-import,unused-wildcard-import


warnings.warn(
    "The 'qiskit.util' namespace is deprecated since qiskit-terra 0.17 and will be removed in 0.20."
    " It has been renamed to 'qiskit.utils'.",
    category=DeprecationWarning,
)
