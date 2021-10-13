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

"""Quantum Operation Mixin."""

from abc import ABC, abstractmethod


class Operation(ABC):
    """Quantum Operation Mixin Class."""

    @property
    @abstractmethod
    def name(self):
        """Unique string identifier for operation type."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self):
        """Number of qubits."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_clbits(self):
        """Number of classical bits."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_params(self):
        """Number of parameters."""
        raise NotImplementedError

    @property
    @abstractmethod
    def params(self):
        """List of parameters to specialize a specific CircuitElement instance."""
        raise NotImplementedError
