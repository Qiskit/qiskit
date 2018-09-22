# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Provider for remote IBMQ backends with admin features."""
import itertools

from qiskit.backends.qiskitprovider import QiskitProvider
from .credentials import Credentials
from .ibmqsingleprovider import IBMQSingleProvider

QE_URL = 'https://quantumexperience.ng.bluemix.net/api'


class IBMQProvider(QiskitProvider):
    """Provider for remote IBMQ backends with admin features."""
    def __init__(self):
        super().__init__()

        self.accounts = {}

    def _backends_list(self):
        # TODO: return iterator, also in base
        return list(itertools.chain(
            *[account.backends() for account in self.accounts.values()]))

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

    def add_account(self, token, url=QE_URL):
        raise NotImplementedError

    def remove_account(self, token, url=QE_URL):
        raise NotImplementedError

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

        single_provider = IBMQSingleProvider(credentials)
        self.accounts[credentials.simple_name()] = single_provider

        return single_provider

    def list_accounts(self):
        raise NotImplementedError

    def load_account(self, account_name):
        raise NotImplementedError

    def load_accounts(self):
        raise NotImplementedError
