# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for remote IBMQ backends with admin features."""

from collections import OrderedDict

from qiskit.backends import BaseProvider

from .credentials._configrc import remove_credentials
from .credentials import (Credentials,
                          read_credentials_from_qiskitrc, store_credentials, discover_credentials)
from .ibmqaccounterror import IBMQAccountError
from .ibmqsingleprovider import IBMQSingleProvider

QE_URL = 'https://quantumexperience.ng.bluemix.net/api'


class IBMQProvider(BaseProvider):
    """Provider for remote IBMQ backends with admin features.

    This class is the entry point for handling backends from IBMQ, allowing
    using different accounts.
    """
    def __init__(self):
        super().__init__()

        # dict[credentials_unique_id: IBMQSingleProvider]
        # This attribute stores a reference to the different accounts. The
        # keys are tuples (hub, group, project), as the convention is that
        # that tuple uniquely identifies a set of credentials.
        self._accounts = OrderedDict()

    def backends(self, name=None, filters=None, **kwargs):
        # pylint: disable=arguments-differ

        # Special handling of the credentials filters.
        credentials_filter = {}
        for key in ['token', 'url', 'hub', 'group', 'project']:
            if key in kwargs:
                credentials_filter[key] = kwargs.pop(key)
        providers = [provider for provider in self._accounts.values() if
                     self._match_all(provider.credentials, credentials_filter)]

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

    @staticmethod
    def deprecated_backend_names():
        """Returns deprecated backend names."""
        return {
            'ibmqx_qasm_simulator': 'ibmq_qasm_simulator',
            'ibmqx_hpc_qasm_simulator': 'ibmq_qasm_simulator',
            'real': 'ibmqx1'
            }

    @staticmethod
    def aliased_backend_names():
        """Returns aliased backend names."""
        return {
            'ibmq_5_yorktown': 'ibmqx2',
            'ibmq_5_tenerife': 'ibmqx4',
            'ibmq_16_rueschlikon': 'ibmqx5',
            'ibmq_20_austin': 'QS1_1'
            }

    def add_account(self, token, url=QE_URL, **kwargs):
        """Authenticate against IBMQ and store the account for future use.

        Login into Quantum Experience or IBMQ using the provided credentials,
        adding the account to the current session. The account is stored in
        disk for future use.

        Args:
            token (str): Quantum Experience or IBM Q API token.
            url (str): URL for Quantum Experience or IBM Q (for IBM Q,
                including the hub, group and project in the URL).
            **kwargs (dict):
                * proxies (dict): Proxy configuration for the API.
                * verify (bool): If False, ignores SSL certificates errors

        Raises:
            IBMQAccountError: if the credentials are already in use.
        """
        credentials = Credentials(token, url, **kwargs)

        # Check if duplicated credentials are already stored. By convention,
        # we assume (hub, group, project) is always unique.
        stored_credentials = read_credentials_from_qiskitrc()

        if credentials.unique_id() in stored_credentials.keys():
            raise IBMQAccountError('Credentials are already stored')

        self._append_account(credentials)

        # Store the credentials back to disk.
        store_credentials(credentials)

    def remove_account(self, token, url=QE_URL, **kwargs):
        """Remove an account from the session and from disk.

        Args:
            token (str): Quantum Experience or IBM Q API token.
            url (str): URL for Quantum Experience or IBM Q (for IBM Q,
                including the hub, group and project in the URL).
            **kwargs (dict):
                * proxies (dict): Proxy configuration for the API.
                * verify (bool): If False, ignores SSL certificates errors

        Raises:
            IBMQAccountError: if the credentials could not be removed.
        """
        removed = False
        credentials = Credentials(token, url, **kwargs)

        # Check if the credentials are already stored in session or disk. By
        # convention, we assume (hub, group, project) is always unique.
        stored_credentials = read_credentials_from_qiskitrc()

        # Try to remove from session.
        if credentials.unique_id() in self._accounts.keys():
            del self._accounts[credentials.unique_id()]
            removed = True

        # Try to remove from disk.
        if credentials.unique_id() in stored_credentials.keys():
            remove_credentials(credentials)
            removed = True

        if not removed:
            raise IBMQAccountError('Unable to find credentials')

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

        self._append_account(credentials)

    def list_accounts(self):
        """List all accounts currently stored in the session.

        Returns:
            list[dict]: a list with information about the accounts currently
                in the session.
        """
        information = []
        for provider in self._accounts.values():
            information.append({
                'token': provider.credentials.token,
                'url': provider.credentials.url,
            })

        return information

    def load_accounts(self, **kwargs):
        """Load IBMQ accounts found in the system, subject to optional filtering.

        Automatically load the accounts found in the system. This method
        looks for credentials in the following locations, in order, and
        returns as soon as credentials are found:

        1. in the `Qconfig.py` file in the current working directory.
        2. in the environment variables.
        3. in the `qiskitrc` configuration file

        Raises:
            IBMQAccountError: if attempting to load previously loaded accounts,
                    or if no credentials can be found.
        """
        # Special handling of the credentials filters.
        credentials_filter = {}
        for key in ['token', 'url', 'hub', 'group', 'project']:
            if key in kwargs:
                credentials_filter[key] = kwargs.pop(key)

        for credentials in discover_credentials().values():
            if self._match_all(credentials, credentials_filter):
                self._append_account(credentials)

        if not self._accounts:
            raise IBMQAccountError('No IBMQ credentials found.')

    def _append_account(self, credentials):
        """Append an account with the specified credentials to the session.

        Args:
            credentials (Credentials): set of credentials.

        Returns:
            IBMQSingleProvider: new single-account provider.

        Raises:
            IBMQAccountError: if the provider could not be appended.
        """
        # Check if duplicated credentials are already in use. By convention,
        # we assume (hub, group, project) is always unique.
        if credentials.unique_id() in self._accounts.keys():
            raise IBMQAccountError('Credentials are already in use')

        single_provider = IBMQSingleProvider(credentials, self)
        self._accounts[credentials.unique_id()] = single_provider

        return single_provider

    def _match_all(self, obj, criteria):
        """Return True if all items in criteria matches items in obj."""
        return all(getattr(obj, key_, None) == value_ for
                   key_, value_ in criteria.items())
