# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redundant-returns-doc,missing-raises-doc

"""Base class for a backend provider."""

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """
    Base class for a backend provider.
    """
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def available_backends(self, *args, **kwargs):
        """
        Returns:
            list[BaseBackend]: a list of backend instances available
            from this provider.
        """
        pass

    @abstractmethod
    def get_backend(self, name):
        """
        Return a backend instance.

        Args:
            name (str): identifier

        Returns:
            BaseBackend: a backend instance.

        Raises:
            KeyError: if `name` is not among the list of backends available
                from this provider.
        """
        pass

    def aliased_backend_names(self):
        """
        Returns dict that defines alias names, usually shorter names
        for referring to the backends.

        If an alias key is used, the corresponding backend will be chosen
        in order of priority from the value list, depending on availability.

        Returns:
            dict[str: list[str]]: {alias_name: list(backend_name)}
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

    def __eq__(self, other):
        """
        Assumes two providers with the same class name clash.
        Derived providers can override this behavior
        (e.g. IBMQProvider instances are equal if and only if
        they have the same authentication attributes as well).
        """
        equality = (type(self).__name__ == type(other).__name__)
        return equality
