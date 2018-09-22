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

        .. deprecated:: 0.6+
            After 0.6, this function is deprecated. Please use `.backends()`
            with a filter instead.
        """
        try:
            return self.backends({'name': name})[0]
        except IndexError:
            raise KeyError('backend "{}" not found.'.format(name))

    @abstractmethod
    def backends(self, filters=None, **kwargs):
        """
        Return backend instances.

        Args:
            filters:
            **kwargs:

        Returns:

        """
        pass

    @abstractmethod
    def _backends_list(self):
        """Return all the available backends in this provider.

        Private function for returning all the available backend instances in
        this provider.

        Returns:
            list[BaseBackend]: list of backend instances.
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

    def resolve_backend_name(self, name):
        """Resolve backend name from a possible short group name, a deprecated name,
        or an alias.

        A group will be resolved in order of member priorities, depending on availability.

        Args:
            name (str): name of backend to resolve

        Returns:
            str: name of resolved backend, which is available from one of the providers

        Raises:
            LookupError: if name cannot be resolved through
            regular available names, nor groups, nor deprecated, nor alias names
        """
        resolved_name = ""
        available = [b.name() for b in self._backends_list()]
        grouped = self.grouped_backend_names()
        deprecated = self.deprecated_backend_names()
        aliased = self.aliased_backend_names()

        if name in available:
            resolved_name = name
        elif name in grouped:
            available_members = [b for b in grouped[name] if b in available]
            if available_members:
                resolved_name = available_members[0]
        elif name in deprecated:
            resolved_name = deprecated[name]
            logger.warning('WARNING: %s is deprecated. Use %s.', name, resolved_name)
        elif name in aliased:
            resolved_name = aliased[name]

        if resolved_name not in available:
            raise LookupError('backend "{}" not found.'.format(name))

        return resolved_name

    def __eq__(self, other):
        """
        Assumes two providers with the same class name clash.
        Derived providers can override this behavior
        (e.g. IBMQSingleProvider instances are equal if and only if
        they have the same authentication attributes as well).
        """
        equality = (type(self).__name__ == type(other).__name__)
        return equality
