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

from abc import ABC


class Operation(ABC):
    """Quantum Operation Mixin Class."""

    def __init__(self, name, num_qubits, num_clbits, params):
        self._name = name
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._params = params

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return self._name

    @property
    def num_qubits(self):
        """Number of qubits."""
        return self._num_qubits

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return self._num_clbits

    @property
    def num_params(self):
        """Number of parameters."""
        return len(self._params)

    @property
    def params(self):
        """List of parameters to specialize a specific Operation instance."""
        return self._params
