# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Mixin for gate operator interface.
"""

from abc import ABC, abstractmethod
from numbers import Integral

from qiskit.exceptions import QiskitError
from qiskit.utils.deprecation import deprecate_function


class GroupMixin(ABC):
    """Abstract Mixin for operator group operations.

    This class defines the following methods

        - :meth:`compose`
        - :meth:`dot`
        - :meth:`tensor`
        - :meth:`expand`
        - :meth:`power`

    And the following operator overloads:

        - ``&``, ``__and__`` -> :meth:`compose`
        - ``^``, ``__xor__`` -> `:meth:`tensor`
        - ``**``, ``__pow__`` -> :meth:`power`

    The following deprecated overloads are also defined:

        - ``*``, ``__mul__`` -> :meth:`dot`
        - ``@``, ``__matmul__`` -> :meth:`compose`

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``compose(self, other, qargs=None, inplace=False)``
        - ``tensor(self, other)``
        - ``expand(self, other)``
    """

    @deprecate_function(
        "Using the `__mul__` operator `A * B` as shorthand for"
        " `A.dot(B)` is deprecated as of version 0.17.0 and will be "
        " removed no earlier than 3 months after the release date."
        " As an alternative, use the compose operator `B & A`"
        " in place of `A * B` as a replacement."
    )
    def __mul__(self, other):
        return self.dot(other)

    @deprecate_function(
        "Using the `__matmul__` operator `A @ B` as shorthand for"
        " `A.compose(B)` is deprecated as of version 0.17.0 and will be "
        " removed no earlier than 3 months after the release date."
        " Use the `A & B` instead."
    )
    def __matmul__(self, other):
        return self.compose(other)

    def __and__(self, other):
        return self.compose(other)

    def __pow__(self, n):
        return self.power(n)

    def __xor__(self, other):
        return self.tensor(other)

    @abstractmethod
    def tensor(self, other):
        r"""Return the tensor product with another CLASS.

        Args:
            other (CLASS): a CLASS object.

        Returns:
            CLASS: the tensor product :math:`a \otimes b`, where :math:`a`
                is the current CLASS, and :math:`b` is the other CLASS.

        .. note::
            The tensor product can be obtained using the ``^`` binary operator.
            Hence ``a.tensor(b)`` is equivalent to ``a ^ b``.

        .. note:
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two operators of the same type ``a.tensor(b) = b.expand(a)``.
        """

    @abstractmethod
    def expand(self, other):
        r"""Return the reverse-order tensor product with another CLASS.

        Args:
            other (CLASS): a CLASS object.

        Returns:
            CLASS: the tensor product :math:`b \otimes a`, where :math:`a`
                is the current CLASS, and :math:`b` is the other CLASS.

        .. note:
            Expand is the opposite operator ordering to :meth:`tensor`.
            For two operators of the same type ``a.expand(b) = b.tensor(a)``.
        """

    @abstractmethod
    def compose(self, other, qargs=None, front=False):
        """Return the operator composition with another CLASS.

        Args:
            other (CLASS): a CLASS object.
            qargs (list or None): Optional, a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            CLASS: The composed CLASS.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
                         incompatible dimensions for specified subsystems.

        .. note::
            Composition (``&``) by default is defined as `left` matrix multiplication for
            matrix operators, while :meth:`dot` is defined as `right` matrix
            multiplication. That is that ``A & B == A.compose(B)`` is equivalent to
            ``B.dot(A)`` when ``A`` and ``B`` are of the same type.

            Setting the ``front=True`` kwarg changes this to `right` matrix
            multiplication and is equivalent to the :meth:`dot` method
            ``A.dot(B) == A.compose(B, front=True)``.
        """

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (CLASS): an operator object.
            qargs (list or None): Optional, a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).

        Returns:
            CLASS: The right matrix multiplied CLASS.
        """
        return self.compose(other, qargs=qargs, front=True)

    def power(self, n):
        """Return the compose of a operator with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            CLASS: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal, or the power is not a positive integer.
        """
        # NOTE: if a subclass can have negative or non-integer powers
        # this method should be overridden in that class.
        if not isinstance(n, Integral) or n < 1:
            raise QiskitError("Can only power with positive integer powers.")
        ret = self
        for _ in range(1, n):
            ret = ret.dot(self)
        return ret
