# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from abc import ABC
from abc import abstractmethod


class Target(ABC):
    """Abstract class for Backend Target

    This abstract class is used to provide target information to the compiler
    about a backend. These are immutable properties about the backend. The
    properties defined here are what information may get consumed by terra,
    however if it doesn't apply to a backend they can return ``None`` to
    indicate this. The version field is specifically around the target format.
    If in the future the fields in a target grow we should create a subclass
    and that subclass should increase the version number. This will let the
    transpiler with extra fields know beforehand which version the target
    is (and ignore fields if they need a newer version). Additionally, in
    the future this will be needed if/when a serialization format is added for
    this.
    """

    version = 1

    @property
    @abstractmethod
    def num_qubits(self):
        """Return the number of qubits for the backend."""
        pass

    @property
    @abstractmethod
    def basis_gates(self):
        """Return the list of basis gates for the backend."""
        pass

    @property
    @abstractmethod
    def supported_instructions(self):
        """Return the list of supported non-gate instructions for the backend."""
        pass

    @property
    @abstractmethod
    def coupling_map(self):
        """Return the qiskit.transpiler.CouplingMap object"""
        pass

    @property
    @abstractmethod
    def conditional(self):
        """Return bool whether the target can execute gates with classical conditions."""
        pass

    @property
    @abstractmethod
    def gates(self):
        """Return a list of dictionaries describing the properties of each gate."""
        pass
