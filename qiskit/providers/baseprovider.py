# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for a backend provider."""

from abc import ABC, abstractmethod
import warnings

from .exceptions import QiskitBackendNotFoundError


class BaseProvider(ABC):
    """Base class for a Backend Provider."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        warnings.warn(
            "The BaseProvider abstract interface is deprecated as of "
            "the 0.18.0 release and will be removed in a future "
            "release. Instead you should build your backends using "
            "the ProviderV1 abstract class (which is the current "
            "latest version of the provider interface).",
            DeprecationWarning,
            stacklevel=2,
        )
        pass

    def get_backend(self, name=None, **kwargs):
        """Return a single backend matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs: dict used for filtering.

        Returns:
            BaseBackend: a backend matching the filtering.

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
            list[BaseBackend]: a list of Backends that match the filtering
                criteria.
        """
        pass

    def __eq__(self, other):
        """Equality comparison.

        By default, it is assumed that two `Providers` from the same class are
        equal. Subclassed providers can override this behavior.
        """
        return type(self).__name__ == type(other).__name__
