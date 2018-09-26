# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for remote IBMQ backends with admin features."""
from collections import OrderedDict

from qiskit import QISKitError

from qiskit.backends import BaseProvider

from .credentials._configrc import remove_credentials
from .credentials import (Credentials,
                          read_credentials_from_qiskitrc, store_credentials, discover_credentials)
from .ibmqsingleprovider import IBMQSingleProvider

QE_URL = 'https://quantumexperience.ng.bluemix.net/api'


class IBMQProvider(BaseProvider):
    """Provider for remote IBMQ backends with admin features."""
    def __init__(self):
        super().__init__()

        self.accounts = OrderedDict()

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ
        def _match_all(obj, criteria):
            """Return True if all items in criteria matches items in obj."""
            return all(getattr(obj, key_, None) == value_ for
                       key_, value_ in criteria.items())

        # Special handling of the credentials filters.
        credentials_filter = {}
        for key in ['token', 'url', 'hub', 'group', 'project']:
            if key in kwargs:
                credentials_filter[key] = kwargs.pop(key)
        providers = [provider for provider in self.accounts.values() if
                     _match_all(provider.credentials, credentials_filter)]

        # Special handling of the `name` parameter, to support alias resolution.
        if name:
            aliases = self.aliased_backend_names()
            aliases.update(self.deprecated_backend_names())
            name = aliases.get(name, name)

        # Aggregate the list of filtered backends.
        backends = []
        for provider in providers:
            backends = backends + provider.backends(
                name=name, filters=filters, **kwargs)

        return backends

    def deprecated_backend_names(self):
        return {
            'ibmqx_qasm_simulator': 'ibmq_qasm_simulator',
            'ibmqx_hpc_qasm_simulator': 'ibmq_qasm_simulator',
            'real': 'ibmqx1'
            }

    def aliased_backend_names(self):
        return {
            'ibmq_5_yorktown': 'ibmqx2',
            'ibmq_5_tenerife': 'ibmqx4',
            'ibmq_16_rueschlikon': 'ibmqx5',
            'ibmq_20_austin': 'QS1_1'
            }

    def add_account(self, token, url=QE_URL, **kwargs):
        """Authenticate against IBMQ and store the account for future use.

        Args:
            token (str): Quantum Experience or IBM Q API token.
            url (str): URL for Quantum Experience or IBM Q (for IBM Q,
                including the hub, group and project in the URL).
            **kwargs (dict):
                * proxies (dict): Proxy configuration for the API.
                * verify (bool): If False, ignores SSL certificates errors

        Raises:
            QISKitError: if the credentials are already in use.
        """
        credentials = Credentials(token, url, **kwargs)

        # Check if duplicated credentials are already stored. By convention,
        # we assume (hub, group, project) is always unique.
        stored_credentials = read_credentials_from_qiskitrc()

        if credentials.unique_id() in stored_credentials.keys():
            raise QISKitError('Credentials are already stored')

        self._append_provider(credentials)

        # Store the credentials back to disk.
        store_credentials(credentials)

    def remove_account(self, token, url=QE_URL, **kwargs):
        """Remove an account.

        Args:
            token:
            url:
            **kwargs:

        Raises:
            QISKitError: if the credentials could not be removed.
        """
        removed = False
        credentials = Credentials(token, url, **kwargs)

        # Check if the credentials are already stored in session or disk. By
        # convention, we assume (hub, group, project) is always unique.

        stored_credentials = read_credentials_from_qiskitrc()

        # Try to remove from session.
        if credentials.unique_id() in self.accounts.keys():
            del self.accounts[credentials.unique_id()]
            removed = True

        # Try to remove from disk.
        if credentials.unique_id() in stored_credentials.keys():
            remove_credentials(credentials)
            removed = True

        if not removed:
            raise QISKitError('Unable to find credentials')

    def use_account(self, token, url=QE_URL, **kwargs):
        """Authenticate against IBMQ during this session.

        Login into Quantum Experience or IBMQ using the provided credentials,
        adding the account to the current session. The account is not stored
        in disk.

        Args:
            token (str): Quantum Experience or IBM Q API token.
            url (str): URL for Quantum Experience or IBM Q (for IBM Q,
                including the hub, group and project in the URL).
            **kwargs (dict):
                * proxies (dict): Proxy configuration for the API.
                * verify (bool): If False, ignores SSL certificates errors
        """
        credentials = Credentials(token, url, **kwargs)

        self._append_provider(credentials)

    def list_accounts(self):
        """List all accounts."""
        information = []
        for provider in self.accounts.values():
            information.append({
                'token': provider.credentials.token,
                'url': provider.credentials.url,
            })

        return information

    def load_accounts(self):
        """Load all accounts."""
        if self.accounts:
            raise QISKitError('The account list is not empty')

        for credentials in discover_credentials().values():
            self._append_provider(credentials)

        if not self.accounts:
            raise QISKitError('No IBMQ credentials found')

    def _append_provider(self, credentials):
        """Append a provider with the specified credentials to the session.

        Args:
            credentials (Credentials): set of credentials.

        Returns:
            IBMQSingleProvider: new provider.

        Raises:
            QISKitError: if the provider could not be appended.
        """
        # Check if duplicated credentials are already in use. By convention,
        # we assume (hub, group, project) is always unique.
        if credentials.unique_id() in self.accounts.keys():
            raise QISKitError('Credentials are already in use')

        single_provider = IBMQSingleProvider(credentials, self)
        self.accounts[credentials.unique_id()] = single_provider

        return single_provider


def _next_available_name(name, existing_names):
    """Return the next non-clashing name by appending an index."""
    candidate = name
    i = 2
    while candidate in existing_names:
        candidate = '{}_{}'.format(name, i)
    return candidate
