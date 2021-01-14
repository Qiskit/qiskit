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

"""Base class for a provider."""

from abc import ABC, abstractmethod

from qiskit.providers.exceptions import QiskitBackendNotFoundError


class Provider:
    """Base common type for all versioned Provider abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """
    version = 0


class ProviderV1(Provider, ABC):
    """Base class for a Backend Provider."""
    version = 1

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
            raise QiskitBackendNotFoundError('More than one backend matches the criteria')
        if not backends:
            raise QiskitBackendNotFoundError('No backend matches the criteria')

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


class ProviderV2(Provider, ABC):
    """Base class for a Backend Provider."""
    version = 2

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
            raise QiskitBackendNotFoundError('More than one backend matches the criteria')
        if not backends:
            raise QiskitBackendNotFoundError('No backend matches the criteria')

        return backends[0]

    @abstractmethod
    @property
    def backends(self):
        """The backends provided by this provider.

        This can be accessed by name via attributes for example::

            my_provider = Provider()
            my_provider.backend.backend_name

        or called with the signature that matches
        :meth:`~qiskit.providers.ProviderV1.backends`, for example::

            my_provider = Provider()
            my_provider.backend(name='backend_name')

        Returns:
            BackendsList: A :class:`~qiskit.providers.BackendsList` object
        """
        pass

    def __eq__(self, other):
        """Equality comparison.

        By default, it is assumed that two `Providers` from the same class are
        equal. Subclassed providers can override this behavior.
        """
        return type(self).__name__ == type(other).__name__


class BackendList:
    """A service class that allows for autocompletion of backends from a provider.

    Each backend can be accessed as attribute by backend directly for example for two
    backends ``BackendA`` (with a name ``'backend_a'``) and ``BackendB`` (with
    a name ``'backend_b'``)::

        from qiskit.providers import BackendV1

        backends = BackendList([BackendA, BackendB])
        backend_a = backends.backend_a.configuration()

    would get the backend configuration object for backend_a. For backwards
    compatibility a ``BackendList`` object is callable just as
    :meth:`qiskit.providers.ProviderV1.backends`. For example::

        backends = BackendList([BackendA, BackendB])
        backend_list = backends(name='backend_a')

    """

    def __init__(self, backends):
        """Initialize a new ``BackendList`` object.

        Args:
            backends (list): List of :class:`~qiskit.providers.Backend` instances.
        """
        self._backends = backends
        for backend in backends:
            setattr(self, backend.name(), backend)

    def __call__(self, name=None, filters=None, **kwargs):
        """A listing of all backends from this provider.

        Args:
            name (str): The name of a given backend.
            filters (callable): A filter function.
        Returns:
            list: A list of backends, if any.
        """
        # pylint: disable=arguments-differ
        backends = self._backends
        if name:
            backends = [
                backend for backend in backends if backend.name() == name]

        return filter_backends(backends, filters=filters, **kwargs)
