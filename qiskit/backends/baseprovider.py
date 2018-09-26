# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redundant-returns-doc,missing-raises-doc

"""Base class for a backend provider."""

from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Base class for a backend provider.
    """
    def __init__(self, *args, **kwargs):
        pass

    def available_backends(self, *args, **kwargs):
        """
        Returns:
            list[BaseBackend]: a list of backend instances available
            from this provider.

        .. deprecated:: 0.6+
            After 0.6, this function is deprecated. Please use `.backends()`
            instead.
        """
        return self.backends(*args, **kwargs)

    def get_backend(self, name=None, **kwargs):
        """Return a single backend matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs (dict): dict used for filtering.

        Returns:
            BaseProvider: a backend matching the filtering.

        Raises:
            KeyError: if no backend could be found or more than one backend
                matches.
        """
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            raise KeyError('More than one backend matches the criteria')
        elif not backends:
            raise KeyError('No backend matches the criteria')

        return backends[0]

    @abstractmethod
    def backends(self, name=None, **kwargs):
        """Return backend instances.

        Args:
            name (str): name of the backend.
            **kwargs (dict): dict used for filtering.

        Returns:
            list[BaseBackend]:
        """
        pass

    def grouped_backend_names(self):
        """
        Returns dict that defines group names, usually shorter names
        for referring to the backends.

        If a group name is used, the corresponding backend will be chosen
        in order of priority from the value list, depending on availability.

        Returns:
            dict[str: list[str]]: {group_name: list(backend_name)}
        """
        return {}

    def deprecated_backend_names(self):
        """
        Dict that stores the current name for all deprecated backends.
        The conversion to the current name is done with a warning.
        These can be gradually removed in subsequent releases.

        Returns:
            dict[str: str]: {deprecated_name: backend_name}
        """
        return {}

    def aliased_backend_names(self):
        """
        Dict that stores possible aliases for a given backend name.
        Either the name itself or an alias can be used to connect to that
        backend.

        Returns:
            dict[str: list[str]]: {backend_name: list(alias_name)}
        """
        return {}

    def __eq__(self, other):
        """
        Assumes two providers with the same class name clash.
        Derived providers can override this behavior
        (e.g. IBMQSingleProvider instances are equal if and only if
        they have the same authentication attributes as well).
        """
        equality = (type(self).__name__ == type(other).__name__)
        return equality
