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
        """Return the list of available backends.

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
            BaseBackend: a backend matching the filtering.

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
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): name of the backend.
            **kwargs (dict): dict used for filtering.

        Returns:
            list[BaseBackend]: a list of backends matching the filtering
                criteria.
        """
        pass

    def __eq__(self, other):
        """Equality comparison.

        By default, it is assumed that two `Providers` from the same class are
        equal. Subclassed providers can override this behavior.
        """
        return type(self).__name__ == type(other).__name__
