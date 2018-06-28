# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Meta-provider that aggregates several providers."""

import logging
from collections import OrderedDict
from itertools import combinations

from qiskit import QISKitError
from qiskit.backends.baseprovider import BaseProvider
from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.local.localprovider import LocalProvider

logger = logging.getLogger(__name__)


class DefaultQISKitProvider(BaseProvider):
    """
    Meta-provider that aggregates several providers.
    """
    def __init__(self):
        super().__init__()

        # Dict of providers.
        self.providers = OrderedDict({'local': LocalProvider()})

    def get_backend(self, name):
        name = self.resolve_backend_name(name)
        for provider in self.providers.values():
            try:
                return provider.get_backend(name)
            except KeyError:
                pass
        raise KeyError(name)

    def available_backends(self, filters=None):
        """Get a list of available backends from all providers (after filtering).

        Args:
            filters (dict or callable): filtering conditions.
                each will either pass through, or be filtered out.
                1) dict: {'criteria': value}
                    the criteria can be over backend's `configuration` or `status`
                    e.g. {'local': False, 'simulator': False, 'available': True}
                2) callable: BaseBackend -> bool
                    e.g. lambda x: x.configuration['n_qubits'] > 5

        Returns:
            list[BaseBackend]: a list of backend instances available
                from all the providers.

        Raises:
            QISKitError: if passing filters that is neither dict nor callable
        """
        # pylint: disable=arguments-differ
        backends = []
        for provider in self.providers.values():
            backends.extend(provider.available_backends())

        if filters is not None:
            if isinstance(filters, dict):
                # exact match filter:
                # e.g. {'n_qubits': 5, 'available': True}
                for key, value in filters.items():
                    backends = [instance for instance in backends
                                if instance.configuration.get(key) == value
                                or instance.status.get(key) == value]
            elif callable(filters):
                # acceptor filter: accept or reject a specific backend
                # e.g. lambda x: x.configuration['n_qubits'] > 5
                accepted_backends = []
                for backend in backends:
                    try:
                        if filters(backend) is True:
                            accepted_backends.append(backend)
                    except Exception:  # pylint: disable=broad-except
                        pass
                backends = accepted_backends
            else:
                raise QISKitError('backend filters must be either dict or callable.')

        return backends

    def aliased_backend_names(self):
        """
        Aggregate alias information from all providers.

        Returns:
            dict[str: list[str]]: aggregated alias dictionary

        Raises:
            ValueError: if a backend is mapped to multiple aliases
        """
        aliases = {}
        for provider in self.providers.values():
            aliases = {**aliases, **provider.aliased_backend_names()}
        for pair in combinations(aliases.values(), r=2):
            if not set.isdisjoint(set(pair[0]), set(pair[1])):
                raise ValueError('duplicate backend alias definition')

        return aliases

    def deprecated_backend_names(self):
        """
        Aggregate deprecated names from all providers.

        Returns:
            dict[str: list[str]]: aggregated alias dictionary
        """
        deprecates = {}
        for provider in self.providers.values():
            deprecates = {**deprecates, **provider.deprecated_backend_names()}

        return deprecates

    def add_provider(self, provider, provider_name):
        """
        Add a new provider to the list of known providers.

        Note:
            If some backend in the new provider has a name in use by an
            already registered provider, the backend will not be available,
            and the name of the backend will still refer to that previously
            registered.

        Args:
            provider (BaseProvider): Provider instance.
            provider_name (str): User-provided name for the provider.

        Returns:
            BaseProvider: the provider instance.

        Raises:
            QISKitError: if a provider with the same name is already in the
                list.
        """
        if provider_name in self.providers.keys():
            raise QISKitError(
                'A provider with name "%s" is already registered.'
                % provider_name)

        # Check for backend name clashes, emitting a warning.
        current_backends = {str(backend) for backend in self.available_backends()}
        added_backends = {str(backend) for backend in provider.available_backends()}
        common_backends = added_backends.intersection(current_backends)
        if common_backends:
            logger.warning(
                'The backend names "%s" (provided by "%s") are already in use. '
                'Consider using unregister() for avoiding name conflicts.',
                list(common_backends), provider_name)

        self.providers[provider_name] = provider

        return provider

    def add_ibmq_provider(self, credentials_dict, provider_name=None):
        """
        Add a new IBMQProvider to the list of known providers.

        Args:
            credentials_dict (dict): dictionary of credentials for a provider.
            provider_name (str): User-provided name for the provider. A name
                will automatically be assigned if possible.
        Raises:
            QISKitError: if a provider with the same name is already in the
                list; or if a provider name could not be assigned.
        Returns:
            IBMQProvider: the new IBMQProvider instance.
        """
        # Automatically assign a name if not specified.
        if not provider_name:
            if 'quantumexperience' in credentials_dict['url']:
                provider_name = 'ibmq'
            elif 'q-console' in credentials_dict['url']:
                provider_name = 'qnet'
            else:
                raise QISKitError(
                    'Cannot parse provider name from credentials.')

        ibmq_provider = IBMQProvider(**credentials_dict)

        return self.add_provider(ibmq_provider, provider_name)

    def remove_provider(self, provider_name):
        """
        Remove a provider from the list of known providers.

        Args:
            provider_name (str): name of the provider to be removed.

        Raises:
            QISKitError: if the provider name is not valid.
        """
        if provider_name == 'local':
            raise QISKitError("Cannot unregister 'local' provider.")
        try:
            self.providers.pop(provider_name)
        except KeyError:
            raise QISKitError("'%s' provider is not registered.")

    def resolve_backend_name(self, name):
        """Resolve backend name from a possible short alias or a deprecated name.

        The alias will be chosen in order of priority, depending on availability.

        Args:
            name (str): name of backend to resolve

        Returns:
            str: name of resolved backend, which is available from one of the providers

        Raises:
            LookupError: if name cannot be resolved through
            regular available names, nor aliases, nor deprecated names
        """
        resolved_name = ""
        available = [b.name for b in self.available_backends()]
        aliased = self.aliased_backend_names()
        deprecated = self.deprecated_backend_names()

        if name in available:
            resolved_name = name
        elif name in aliased:
            available_dealiases = [b for b in aliased[name] if b in available]
            if available_dealiases:
                resolved_name = available_dealiases[0]
        elif name in deprecated:
            resolved_name = deprecated[name]
            logger.warning('WARNING: %s is deprecated. Use %s.', name, resolved_name)

        if resolved_name not in available:
            raise LookupError('backend "{}" not found.'.format(name))

        return resolved_name
