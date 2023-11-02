# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum bit and Classical bit objects.
"""
from qiskit.circuit.exceptions import CircuitError


class Bit:
    """Implement a generic bit.

    .. note::
        This class should not be instantiated directly. This is just a superclass
        for :class:`~.Clbit` and :class:`~.circuit.Qubit`.

    """

    _register = None
    __slots__ = {"_index", "_repr"}

    def __init__(self, register=None, index=None):
        """Create a new generic bit."""
        del register
        if index is not None:
            try:
                index = int(index)
            except Exception as ex:
                raise CircuitError(
                    f"index needs to be castable to an int: type {type(index)} was provided"
                ) from ex

        self._index = index

    def __repr__(self):
        """Return the official string representing the bit."""
        if self._index is None:
            # Similar to __hash__, use default repr method for new-style Bits.
            return object.__repr__(self)
        return f"{self.__class__.__name__}({self._index})"

    def __hash__(self):
        if self._index is None:
            # Similar to __repr__, use default repr method for new-style Bits.
            return object.__hash__(self)
        return hash(self._index)

    def __copy__(self):
        # Bits are immutable.
        return self

    def __deepcopy__(self, memo=None):
        # Bits are immutable.
        return self
