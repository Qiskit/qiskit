# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Pauli Evolution Gate."""

import warnings
from .evolution import PauliEvolutionGate  # pylint: disable=unused-import

warnings.warn(
    "The PauliEvolutionGate import from qiskit.circuit.library.pauli_evolution is pending "
    "and will be deprecated no sooner than 3 months after the Qiskit Terra 0.24 release. Instead, "
    "the full import path is qiskit.circuit.library.evolution.pauli_evolution.",
    stacklevel=2,
    category=PendingDeprecationWarning,
)
