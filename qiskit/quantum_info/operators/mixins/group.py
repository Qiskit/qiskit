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
# pylint: disable=abstract-method

from abc import ABC, abstractmethod
from numbers import Integral

from qiskit.exceptions import QiskitError


class GroupMixin(ABC):
    """Abstract Mixin for operator group operations.

    This class defines the following methods

        - :meth:`compose`
        - :meth:`dot`
        - :meth:`tensor`
        - :meth:`expand`
        - :meth:`power`

    And the following operator overloads:

        - ``*``, ``__mul__`` -> :meth:`dot`
        - ``@``, ``__matmul__`` -> :meth:`compose`
        - ``**``, ``__pow__`` -> :meth:`power`

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``_compose(self, other, qargs=None, inplace=False)``
        - ``_tensor(cls, a, b)``
    """

    def __mul__(self, other):
        return self.dot(other)

    def __matmul__(self, other):
        return self.compose(other)

    def __pow__(self, n):
        return self.power(n)

    def __xor__(self, other):
        return self.tensor(other)

    def tensor(self, other):
        r"""Return the tensor product with another {cls}.

        Args:
            other ({cls}): a {cls} object.

        Returns:
            {cls}: the tensor product :math:`a \otimes b`, where :math:`a`
                   is the current {cls}, and :math:`b` is the other {cls}.

        .. note::
            The tensor product can be obtained using the ``^`` binary operator.
            Hence ``a.tensor(b)`` is equivalent to ``a ^ b``.

        .. note:
            Tensor uses reversed operator ordering to :meth:`expand`.
            For two operators of the same type ``a.tensor(b) = b.expand(a)``.
        """.format(cls=type(self).__name__)
        return self._tensor(self, other)

    def expand(self, other):
        r"""Return the reverse-order tensor product with another {cls}.

        Args:
            other ({cls}): a {cls} object.

        Returns:
            {cls}: the tensor product :math:`b \otimes a`, where :math:`a`
                   is the current {cls}, and :math:`b` is the other {cls}.

        .. note:
            Expand is the opposite operator ordering to :meth:`tensor`.
            For two operators of the same type ``a.expand(b) = b.tensor(a)``.
        """.format(cls=type(self).__name__)
        return self._tensor(other, self)

    def compose(self, other, qargs=None, front=False):
        """Return the operator composition with another {cls}.

        Args:
            other ({cls}): a {cls} object.
            qargs (list or None): Optional, a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            {cls}: The composed {cls}.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
                         incompatible dimensions for specified subsystems.

        .. note::
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """.format(cls=type(self).__name__)
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        return self._compose(other, qargs=qargs, front=front)

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other ({cls}): an operator object.
            qargs (list or None): Optional, a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).

        Returns:
            {cls}: The operator self * other.

        .. note::
            The dot product can be obtained using the ``*`` binary operator.
            Hence ``a.dot(b)`` is equivalent to ``a * b``. Left operator
            multiplication can be obtained using the :meth:`compose` method.

        """.format(cls=type(self).__name__)
        return self.compose(other, qargs=qargs, front=True)

    def power(self, n):
        """Return the compose of a operator with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            {cls}: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal, or the power is not a positive integer.
        """.format(cls=type(self).__name__)
        # NOTE: if a subclass can have negative or non-integer powers
        # this method should be overridden in that class.
        if not isinstance(n, Integral) or n < 1:
            raise QiskitError("Can only power with positive integer powers.")
        ret = self
        for _ in range(1, n):
            ret = ret.dot(self)
        return ret

    @classmethod
    @abstractmethod
    def _tensor(cls, a, b):
        """Return the tensor product a ⊗ b.

        Args:
            a ({cls}): an operator object.
            b ({cls}): an operator object.

        Returns:
            {cls}: the tensor product a ⊗ b.
        """.format(cls=cls.__name__)

    @abstractmethod
    def _compose(self, other, qargs=None, front=False):
        """Return the dot product a * b

        Args:
            a ({cls}): an operator object.
            b ({cls}): an operator object.
            qargs (list or None): Optional, a list of subsystem positions to
                                  apply other on. If None apply on all
                                  subsystems (default: None).
            Returns:
                {cls}: The operator self @ other.

        Returns:
            {cls}: The operator a.compose(b)
        """.format(cls=type(self).__name__)
