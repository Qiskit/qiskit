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

"""A standard gradient descent optimizer."""

from typing import Union, Callable, Optional, List, Iterator, Generator
from itertools import tee
import numpy as np


class LearningRate(Generator):
    """Represents a Learning Rate.
    Will be an attribute of :class:`~.GradientDescentState`. Note that :class:`~.GradientDescent` also
    has a learning rate. That learning rate can be a float, a list, an array, a function returning
    a generator and will be used to create a generator to be used during the
    optimization process.
    This class serves also as a wrapper on a generator so that we can access the last yielded value.
    """

    def __init__(
        self, learning_rate: Union[float, List[float], np.ndarray, Callable[[], Iterator]]
    ):
        """
        Args:
            learing_rate: Used to create a generator to iterate on.
        """
        if isinstance(learning_rate, (float, int)):
            self._gen = constant(learning_rate)
        elif isinstance(learning_rate, Generator):
            learning_rate, self._gen = tee(learning_rate)
        elif isinstance(learning_rate, (list, np.ndarray)):
            self._gen = (eta for eta in learning_rate)
        else:
            self._gen = learning_rate()

        self._current: Optional[float] = None

    def send(self, ignored_arg):
        self._current = next(self._gen)
        return self.current

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    @property
    def current(self):
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
