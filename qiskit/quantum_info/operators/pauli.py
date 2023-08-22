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


"""
DEPRECATED Tools for working with Pauli Operators.
"""

from __future__ import annotations
import warnings


def __getattr__(name):
    if name == "Pauli":
        from qiskit.quantum_info import Pauli

        warnings.warn(
            f"Importing from '{__name__}' is deprecated since Qiskit Terra 0.21 and the module"
            " will be removed in a future release.  Import directly from 'qiskit.quantum_info'.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        return Pauli
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
