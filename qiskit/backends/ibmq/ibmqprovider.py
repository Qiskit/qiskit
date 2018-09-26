# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for remote IBMQ backends with admin features."""
import itertools
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
        def _match_all(obj, criteria):
            """Return True if all items in criteria matches items in obj."""
            return all(getattr(obj, key_, None) == value_ for
                       key_, value_ in criteria.items())

        # Special handling of the credentials filters.
        credentials_filter = {}
        for key in ['token', 'url', 'hub', 'group', 'project']:
            if key in kwargs:
                credentials_filter[key] = kwargs.pop(key)
        accounts = [account for account in self.accounts.values() if
                    _match_all(account.credentials, credentials_filter)]

        # Special handling of the `name` parameter, to support alias resolution.
        if name:
            aliases = self.aliased_backend_names()
            aliases.update(self.deprecated_backend_names())
            name = aliases.get(name, name)

        # Aggregate the list of filtered backends.
        backends = []
        for provider in accounts:
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

        Keyword Args:
            proxies (dict): Proxy configuration for the API.
            verify (bool): If False, ignores SSL certificates errors
        """
        credentials = Credentials(token, url, **kwargs)

        # Check if the exact same credentials are already stored.
        stored_credentials = read_credentials_from_qiskitrc()
        session_credentials = {account_name: provider.credentials for
                               account_name, provider in self.accounts.items()}

        if credentials in session_credentials.values():
            # Exact same credentials are already in use.
            raise QISKitError('Credentials are already loaded')
        elif credentials in stored_credentials.values():
            # Exact same credentials are already stored under a different name.
            raise QISKitError('Credentials are already stored, but not loaded')

        # Find a suitable account name, avoiding clashing with the .qiskitrc.
        account_name = _next_available_name(credentials.simple_name(),
                                            stored_credentials.keys())
        provider = self._append_provider(credentials, account_name)

        # Store the credentials back to disk.
        stored_credentials[account_name] = credentials
        store_credentials(credentials, account_name=account_name)

        return provider

    def remove_account(self, token, url=QE_URL, **kwargs):
        is_changed = False
        credentials = Credentials(token, url, **kwargs)

        # Check if the exact same credentials are already stored.
        stored_credentials = read_credentials_from_qiskitrc()
        session_credentials = {account_name: provider.credentials for
                               account_name, provider in self.accounts.items()}

        # Try to remove from session.
        if credentials in session_credentials.values():
            account_name = [name for name, provider in self.accounts.items()
                            if provider.credentials == credentials][0]
            del self.accounts[account_name]
            is_changed = True

        # Try to remove from disk.
        if credentials in stored_credentials.values():
            account_name = [name for name, stored in stored_credentials.items()
                            if stored == credentials][0]
            remove_credentials(account_name)
            is_changed = True

        if not is_changed:
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

        Keyword Args:
            proxies (dict): Proxy configuration for the API.
            verify (bool): If False, ignores SSL certificates errors
        """
        credentials = Credentials(token, url, **kwargs)

        session_credentials = {account_name: provider.credentials for
                               account_name, provider in self.accounts.items()}

        if credentials in session_credentials.values():
            # Exact same credentials are already in use.
            raise QISKitError('Credentials are already loaded')

        # Find a suitable account name, avoiding clashing with the session.
        account_name = _next_available_name(credentials.simple_name(),
                                            self.accounts.keys())
        return self._append_provider(credentials, account_name=account_name)

    def list_accounts(self):
        return dict(self.accounts)

    def load_account(self, account_name):
        stored_credentials = discover_credentials()

        if account_name in stored_credentials:
            return self._append_provider(stored_credentials[account_name],
                                         account_name)

        raise QISKitError('No account "{}" found' % account_name)

    def load_accounts(self):
        new_providers = OrderedDict()
        for account_name, credentials in discover_credentials().items():
            new_providers[account_name] = self._append_provider(credentials, account_name)

        if not new_providers:
            raise QISKitError('No IBMQ credentials found')

        return new_providers

    def _append_provider(self, credentials, account_name=None):
        """Append a provider with the specified credentials to the session.

        Args:
            credentials (Credentials): set of credentials.
            account_name (str): account name.

        Returns:
            IBMQSingleProvider: new provider.

        Raises:
            QISKitError: if the provider could not be appended.
        """
        account_name = account_name or credentials.simple_name()
        if account_name in self.accounts:
            raise QISKitError('Account name already exists in this session.')

        single_provider = IBMQSingleProvider(credentials)
        self.accounts[account_name] = single_provider

        return single_provider


def _next_available_name(name, existing_names):
    """Return the next non-clashing name by appending an index."""
    candidate = name
    i = 2
    while candidate in existing_names:
        candidate = '{}_{}'.format(name, i)
    return candidate
