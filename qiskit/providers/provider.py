# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for a provider."""

from abc import ABC, abstractmethod

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.utils import deprecate_func


class Provider:
    """Base common type for all versioned Provider abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """

    version = 0

    @deprecate_func(
        since=1.1,
        additional_msg="The abstract Provider and ProviderV1 classes are deprecated and will be "
        "removed in 2.0. You can just remove it as the parent class and a `get_backend` "
        "method that returns the backends from `self.backend`.",
    )
    def __init__(self):
        pass


class ProviderV1(Provider, ABC):
    """Base class for a Backend Provider."""

    version = 1

    @deprecate_func(
        since=1.1,
        additional_msg="The abstract Provider and ProviderV1 classes are deprecated and will  be "
        "removed in 2.0. You can just remove it as the parent class and a `get_backend` "
        "method that returns the backends from `self.backend`.",
    )
    def get_backend(self, name=None, **kwargs):
        """Return a single backend matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.

        Returns:
            Backend: a backend matching the filtering.

        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
                more than one backend matches the filtering criteria.
        """
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError("More than one backend matches the criteria")
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches the criteria")

        return backends[0]

    @abstractmethod
    def backends(self, name=None, **kwargs):
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.

        Returns:
            list[Backend]: a list of Backends that match the filtering
                criteria.
        """
        pass

    def __eq__(self, other):
        """Equality comparison.

        By default, it is assumed that two `Providers` from the same class are
        equal. Subclassed providers can override this behavior.
        """
        return type(self).__name__ == type(other).__name__
