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
"""
Quantum Shannon Decomposition.
"""

from __future__ import annotations
import warnings

# pylint: disable=wildcard-import,unused-wildcard-import

from qiskit.synthesis.unitary.qsd import *

warnings.warn(
    "The qiskit.quantum_info.synthesis module is deprecated since Qiskit 0.46.0."
    "It will be removed in the Qiskit 1.0 release.",
    stacklevel=2,
    category=DeprecationWarning,
)
