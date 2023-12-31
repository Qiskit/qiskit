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

"""State and Unitary synthesis methods."""

from __future__ import annotations
import warnings
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.synthesis.two_qubits.xx_decompose import XXDecomposer
from .two_qubit_decompose import TwoQubitBasisDecomposer, two_qubit_cnot_decompose

warnings.warn(
    "The qiskit.quantum_info.synthesis module is pending deprecation since Qiskit 0.46.0."
    "It will be deprecated in a following release, no sooner than 3 months after the 0.46.0 release.",
    stacklevel=2,
    category=PendingDeprecationWarning,
)
