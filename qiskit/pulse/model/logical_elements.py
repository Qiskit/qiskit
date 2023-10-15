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
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from qiskit.pulse.exceptions import PulseError


class LogicalElement(ABC):
    """Base class of logical elements.

    A :class:`LogicalElement` is an abstraction of a quantum hardware component which can be controlled
    by the user (instructions can be applied to it).
    Every played pulse and most other instructions are associated with a :class:`LogicalElement` on which
    they are performed.
    A logical element is identified by its type and index.

    """

    def __init__(self, index: Tuple[int, ...]):
        """Create ``LogicalElement``.

        Args:
            index: Tuple of indices of the logical element.
        """
        self._validate_index(index)
        self._index = index
        self._hash = hash((self._index, type(self)))

    @property
    def index(self) -> Tuple[int, ...]:
        """Return the ``index`` of this logical element."""
        return self._index

    @abstractmethod
    def _validate_index(self, index) -> None:
        """Raise a PulseError if the logical element ``index`` is invalid.

        Raises:
            PulseError: If ``index`` is not valid.
        """
        pass

    def __eq__(self, other: "LogicalElement") -> bool:
        """Return True iff self and other are equal, specifically, iff they have the same type
        and the same ``index``.

        Args:
            other: The logical element to compare to this one.

        Returns:
            True iff equal.
        """
        return type(self) is type(other) and self._index == other._index

    def __repr__(self) -> str:
        ind_str = str(self._index) if len(self._index) > 1 else f"({self._index[0]})"
        return type(self).__name__ + ind_str

    def __hash__(self) -> int:
        return self._hash


class Qubit(LogicalElement):
    """Qubit logical element.

    ``Qubit`` represents the different qubits in the system, as identified by
    their (positive integer) index values.
    """

    def __init__(self, index: int):
        """Qubit logical element.

        Args:
            index: Qubit index.
        """
        super().__init__((index,))

    @property
    def qubit_index(self):
        """Index of the Qubit"""
        return self.index[0]

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the qubit index is invalid. Namely, check if the index is a
        non-negative integer.

        Raises:
            PulseError: If ``index`` is a negative integer.
        """
        if not isinstance(index[0], (int, np.integer)) or index[0] < 0:
            raise PulseError("Qubit index must be a non-negative integer")


class Coupler(LogicalElement):
    """Coupler logical element.

    :class:`Coupler` represents an element which couples qubits, and can be controlled on its own.
    It is identified by the tuple of indices of the coupled qubits.
    """

    def __init__(self, *qubits):
        """Coupler logical element.

        The coupler ``index`` is defined as the ``tuple`` (\\*qubits).

        Args:
            *qubits: any number of qubit indices coupled by the coupler.
        """
        super().__init__(tuple(qubits))

    def _validate_index(self, index) -> None:
        """Raise a ``PulseError`` if the coupler ``index`` is invalid. Namely,
        check if coupled qubit indices are non-negative integers, at least two indices were provided,
        and that the indices don't repeat.

        Raises:
            PulseError: If ``index`` is invalid.
        """
        if len(index) < 2:
            raise PulseError("At least two qubit indices are needed for a Coupler")
        for qubit_index in index:
            if not isinstance(qubit_index, (int, np.integer)) or qubit_index < 0:
                raise PulseError("Both indices of coupled qubits must be non-negative integers")
        if len(set(index)) != len(index):
            raise PulseError("Indices of a coupler can not repeat")
