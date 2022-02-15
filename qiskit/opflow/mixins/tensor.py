# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The tensor mixin abstract base class."""

from abc import ABC, abstractmethod
from numbers import Integral


class TensorMixin(ABC):
    """The mixin class for tensor operations.

    This class overrides:
        - ``^``, ``__xor__``, `__rxor__` -> :meth:`tensor` between two operators and
        :meth:`tensorpower` with integer.
    The following abstract methods must be implemented by subclasses:
        - :meth:``tensor(self, other)``
        - :meth:``tensorpower(self, other: int)``
    """

    def __xor__(self, other):
        if isinstance(other, Integral):
            return self.tensorpower(other)
        else:
            return self.tensor(other)

    def __rxor__(self, other):
        # a hack to make (I^0)^Z work as intended.
        if other == 1:
            return self
        else:
            return other.tensor(self)

    @abstractmethod
    def tensor(self, other):
        r"""Return tensor product between self and other, overloaded by ``^``."""

    @abstractmethod
    def tensorpower(self, other: int):
        r"""Return tensor product with self multiple times, overloaded by ``^``."""
