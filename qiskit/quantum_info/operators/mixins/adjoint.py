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


class AdjointMixin(ABC):
    """Abstract Mixin for operator adjoint and transpose operations.

    This class defines the following methods

        - :meth:`transpose`
        - :meth:`conjugate`
        - :meth:`adjoint`

    The following abstract methods must be implemented by subclasses
    using this mixin

        - ``conjugate(self)``
        - ``transpose(self)``
    """

    def adjoint(self):
        """Return the adjoint of the CLASS."""
        return self.conjugate().transpose()

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the CLASS."""

    @abstractmethod
    def transpose(self):
        """Return the transpose of the CLASS."""
