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


class BasePub:
    """Base class for PUB (Primitive Unified Bloc)"""

    __slots__ = ("_circuit",)

    def __init__(self, circuit: QuantumCircuit, validate: bool = False):
        """
        Initialize a BasePub.

        Args:
            circuit: Quantum circuit object for the pubs.
            validate: if True, the input data is validated during initizlization.
        """
        self._circuit = circuit
        if validate:
            self.validate()

    @property
    def circuit(self) -> QuantumCircuit:
        """A quantum circuit for the pub"""
        return self._circuit

    def validate(self):
        """Validate the data"""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")
