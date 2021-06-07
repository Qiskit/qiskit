# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Serializable iterators for Qiskit's optimization routines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List


class SerializableIterator(ABC):
    """A base class for serializable iterators."""

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Serialize the iterator."""
        raise NotImplementedError

    @abstractmethod
    def get_iterator(self) -> Iterator[float]:
        """Get the iterator."""
        raise NotImplementedError

    @staticmethod
    def deserialize(serialized: Dict[str, Any]) -> "SerializableIterator":
        """Construct the iterator from the serialized data."""

        name = serialized.pop("name")
        classes = {"constant": Constant, "powerlaw": Powerlaw, "concatenated": Concatenated}
        return classes[name](**serialized)


class Constant(SerializableIterator):
    """An iterator yielding constant values."""

    def __init__(self, value: float) -> None:
        """
        Args:
            value: The constant value yielded from this iterator.
        """
        self.value = value

    def get_iterator(self) -> Iterator[float]:
        def constant_it():
            while True:
                yield self.value

        return constant_it

    def serialize(self) -> Dict[str, Any]:
        return {"name": "constant", "value": self.value}


class Powerlaw(SerializableIterator):
    r"""An iterator yielding values from a powerlaw.

    The powerlaw is

    .. math::

        k(n) = c \left(\frac{1}{n + A}\right)^p,

    where :math:`c` is the constant coeffient (``coeff``), :math:`p` is the exponent
    (``exponent``), :math:`A` is a constant offset (``offset``) and :math:`n` is an integer.
    """

    def __init__(self, coeff, exponent, offset, skip=0):
        """
        Args:
            coeff: The coefficient of the powerlaw.
            power: The exponent in the powerlaw.
            offset: The offset.
            skip: How many initial values to skip in the iterator.
        """
        self.coeff = coeff
        self.exponent = exponent
        self.offset = offset
        self.skip = skip

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": "powerlaw",
            "coeff": self.coeff,
            "exponent": self.exponent,
            "offset": self.offset,
            "skip": self.skip,
        }

    def get_iterator(self) -> Iterator[float]:
        def powerlaw_it():
            n = 1
            while True:
                if n > self.skip:
                    yield self.coeff / ((n + self.offset) ** self.exponent)
                n += 1

        return powerlaw_it


class Concatenated(SerializableIterator):
    """An iterator consisting of concatenated other iterators."""

    def __init__(self, iterators: List[SerializableIterator], breakpoints: List[int]) -> None:
        """
        Args:
            iterators: A list of iterators this iterator is made up of.
            breakpoints: A list of integers specifying when to use the next iterator.
        """
        self.iterators = []
        # deserialize if necessary
        for iterator in iterators:
            if isinstance(iterator, (list, tuple)):
                self.iterators.append(self.deserialize(iterator))
            else:
                self.iterators.append(iterator)

        self.breakpoints = breakpoints

    def serialize(self) -> Dict[str, Any]:
        return {
            "name": "concatenated",
            "iterators": [it.serialize() for it in self.iterators],
            "breakpoints": self.breakpoints,
        }

    def get_iterator(self) -> Iterator[float]:
        iterators = [it.get_iterator()() for it in self.iterators]
        breakpoints = self.breakpoints

        def concat():
            i, n = 0, 0  # n counts always up, i is at which iterator/breakpoint pair we are
            while True:
                if i < len(breakpoints) and n >= breakpoints[i]:
                    i += 1
                yield next(iterators[i])  # pylint: disable=stop-iteration-return
                n += 1

        return concat
