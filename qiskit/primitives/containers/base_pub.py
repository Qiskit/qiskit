# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Base Pubs class
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from .dataclasses import frozen_dataclass


@frozen_dataclass
class BasePub:
    """Base class for PUB (Primitive Unified Bloc)"""

    circuit: QuantumCircuit
    """Quantum circuit object for the pubs."""

    def validate(self):
        """Validate the data"""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")
