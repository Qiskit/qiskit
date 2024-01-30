# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=unused-wildcard-import,missing-module-docstring,wildcard-import
import warnings

from qiskit.utils.parallel import *

warnings.warn(
    "'qiskit.tools.parallel' is deprecated and is now located at 'qiskit.utils.parallel'. This "
    "path will no longer work in Qiskit 1.0.0",
    DeprecationWarning,
    2,
)
