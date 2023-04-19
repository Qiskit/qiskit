# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class to represent the Learning Rate."""
from __future__ import annotations

from collections.abc import Generator, Callable
from itertools import tee
import numpy as np


class LearningRate(Generator):
    """Represents a Learning Rate.
    Will be an attribute of :class:`~.GradientDescentState`. Note that :class:`~.GradientDescent` also
    has a learning rate. That learning rate can be a float, a list, an array, a function returning
    a generator and will be used to create a generator to be used during the
    optimization process.
    This class wraps ``Generator`` so that we can also access the last yielded value.
    """

    def __init__(
        self,
        learning_rate: float
        | list[float]
        | np.ndarray
        | Callable[[], Generator[float, None, None]],
    ):
        """
        Args:
            learning_rate: Used to create a generator to iterate on.
        """
        if isinstance(learning_rate, (float, int)):
            self._gen = constant(learning_rate)
        elif isinstance(learning_rate, Generator):
            learning_rate, self._gen = tee(learning_rate)
        elif isinstance(learning_rate, (list, np.ndarray)):
            self._gen = (eta for eta in learning_rate)
        else:
            self._gen = learning_rate()

        self._current: float | None = None

    def send(self, value):
        """Send a value into the generator.
        Return next yielded value or raise StopIteration.
        """
        self._current = next(self._gen)
        return self.current

    def throw(self, typ, val=None, tb=None):
        """Raise an exception in the generator.
        Return next yielded value or raise StopIteration.
        """
        if val is None:
            if tb is None:
                raise typ
            val = typ()
        if tb is not None:
            val = val.with_traceback(tb)
        raise val

    @property
    def current(self):
        """Returns the current value of the learning rate."""
        return self._current


def constant(learning_rate: float = 0.01) -> Generator[float, None, None]:
    """Returns a python generator that always yields the same value.

    Args:
        learning_rate: The value to yield.

    Yields:
        The learning rate for the next iteration.
    """

    while True:
        yield learning_rate
