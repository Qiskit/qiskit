# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Functionality and helpers for testing Qiskit."""

import warnings
from .base import QiskitTestCase
from .decorators import requires_aer_provider, online_test, slow_test
from .reference_circuits import ReferenceCircuits
from .utils import Path

warnings.warn(
    "The qiskit.test module is deprecated in Qiskit 0.46 and it going to be removed in Qiskit 1.0.",
    DeprecationWarning,
    stacklevel=2,
)
