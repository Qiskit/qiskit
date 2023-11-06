# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The star algebra mixin abstract base class."""

from abc import ABC, abstractmethod
from numbers import Integral

from qiskit.quantum_info.operators.mixins import MultiplyMixin
from qiskit.utils.deprecation import deprecate_func


class StarAlgebraMixin(MultiplyMixin, ABC):
    """Deprecated: The star algebra mixin class.
    Star algebra is an algebra with an adjoint.

    This class overrides:
        - ``*``, ``__mul__``, `__rmul__`,  -> :meth:`mul`
        - ``/``, ``__truediv__``,  -> :meth:`mul`
        - ``__neg__`` -> :meth:``mul`
        - ``+``, ``__add__``, ``__radd__`` -> :meth:`add`
        - ``-``, ``__sub__``, `__rsub__`,  -> :meth:a`add`
        - ``@``, ``__matmul__`` -> :meth:`compose`
        - ``**``, ``__pow__`` -> :meth:`power`
        - ``~``, ``__invert__`` -> :meth:`adjoint`

    The following abstract methods must be implemented by subclasses:
        - :meth:`mul(self, other)`
        - :meth:`add(self, other)`
        - :meth:`compose(self, other)`
        - :meth:`adjoint(self)`
    """

    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(self) -> None:
        pass

    # Scalar multiplication

    @abstractmethod
    def mul(self, other: complex):
        """Return scalar multiplication of self and other, overloaded by `*`."""

    def __mul__(self, other: complex):
        return self.mul(other)

    def _multiply(self, other: complex):
        return self.mul(other)

    # Addition, substitution

    @abstractmethod
    def add(self, other):
        """Return Operator addition of self and other, overloaded by `+`."""

    def __add__(self, other):
        # Hack to be able to use sum(list_of_ops) nicely because
        # sum adds 0 to the first element of the list.
        if other == 0:
            return self

        return self.add(other)

    def __radd__(self, other):
        # Hack to be able to use sum(list_of_ops) nicely because
        # sum adds 0 to the first element of the list.
        if other == 0:
            return self
        return self.add(other)

    def __sub__(self, other):
        return self.add(-other)

    def __rsub__(self, other):
        return self.neg().add(other)

    # Operator multiplication

    @abstractmethod
    def compose(self, other):
        """Overloads the matrix multiplication operator `@` for self and other.
        `Compose` computes operator composition between self and other (linear algebra-style:
        A@B(x) = A(B(x))).
        """

    def power(self, exponent: int):
        r"""Return Operator composed with self multiple times, overloaded by ``**``."""
        if not isinstance(exponent, Integral):
            raise TypeError(
                f"Unsupported operand type(s) for **: '{type(self).__name__}' and "
                f"'{type(exponent).__name__}'"
            )

        if exponent < 1:
            raise ValueError("The input `exponent` must be a positive integer.")

        res = self
        for _ in range(1, exponent):
            res = res.compose(self)
        return res

    def __matmul__(self, other):
        return self.compose(other)

    def __pow__(self, exponent: int):
        return self.power(exponent)

    # Adjoint

    @abstractmethod
    def adjoint(self):
        """Returns the complex conjugate transpose (dagger) of self.adjoint

        Returns:
            An operator equivalent to self's adjoint.
        """

    def __invert__(self):
        """Overload unary `~` to return Operator adjoint."""
        return self.adjoint()
