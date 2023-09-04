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
Logical Elements
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from qiskit.pulse.exceptions import PulseError


class LogicalElement(metaclass=ABCMeta):
    """Base class of logical elements.

    :class:`LogicalElement`s are abstraction of the quantum HW component which can be controlled
    by the user ("apply instructions on").
    Every played pulse, and most instructions are associated with a :class:`LogicalElement` on which
    it is being performed.
    Logical elements identified by their index, and a unique name for each class such that the
    objects name is given by ``<class name><index>``.

    To implement a new logical element inherit from :class:`LogicalElement` the ``name`` method needs to
    be overridden with a proper name for the class.
    """

    def __init__(self, index):
        """Create ``LogicalElement``.

        Args:
            index: The index of the logical element.
        """
        self._validate_index(index)
        self._index = index
        self._hash = hash(self.name)

    @property
    def index(self):
        """Return the ``index`` of this logical element."""
        return self._index

    @abstractmethod
    def _validate_index(self, index) -> None:
        """Raise a PulseError if the logical element ``index`` is invalid.

        Raises:
            PulseError: If ``index`` is not valid.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this logical element."""
        pass

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: "LogicalElement") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same ``identifier``.

        Args:
            other: The logical element to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._index == other._index

    def __hash__(self) -> int:
        return self._hash


class Qubit(LogicalElement):
    """Qubit logical element.

    ``Qubit`` represents the different qubits in the system, as identified by
    their (positive integer) index.
    """

    # pylint: disable=useless-parent-delegation
    def __init__(self, index: int):
        """Qubit logical element.

        Args:
            index: Qubit index.
        """
        super().__init__(index)

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise PulseError("Qubit index must be a non-negative integer")

    @property
    def name(self) -> str:
        """Return the name of this qubit"""
        return f"Q{self.index}"


class Coupler(LogicalElement):
    """Coupler logical element.

    :class:`Coupler` represents an element which couples two qubits, and can be controlled on its own.
    It is identified by the tuple of indices of the coupled qubits.
    """

    def __init__(self, qubit_index_1: int, qubit_index_2: int):
        """Coupler logical element.

        The coupler ``index`` is defined as the ``tuple`` (``qubit_index_1``,``qubit_index_2``).

        Args:
            qubit_index_1: Index of the first qubit coupled by the coupler.
            qubit_index_2: Index of the second qubit coupled by the coupler.
        """
        super().__init__((qubit_index_1, qubit_index_2))

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the coupler ``index`` is invalid. Namely,
        check if coupled qubit indices are non-negative integers.

        Raises:
            PulseError: If ``index`` is invalid.
        """
        for qubit_index in index:
            if not isinstance(qubit_index, (int, np.integer)) or qubit_index < 0:
                raise PulseError("Both indices of coupled qubits must be non-negative integers")

    @property
    def name(self) -> str:
        """Return the name of this coupler"""
        return f"Coupler{self.index}"
