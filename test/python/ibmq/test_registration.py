# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""
Test the registration and credentials features of the wrapper.
"""
import warnings
import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from unittest import skipIf
from unittest.mock import patch

import qiskit
from qiskit import QiskitError
from qiskit.backends.ibmq.credentials import (
    _configrc, _qconfig, discover_credentials, store_credentials, Credentials,
    read_credentials_from_qiskitrc)
from qiskit.backends.ibmq.credentials._environ import VARIABLES_MAP
from qiskit.backends.ibmq.ibmqprovider import QE_URL
from qiskit.backends.ibmq.ibmqsingleprovider import IBMQSingleProvider
from ..common import QiskitTestCase


IBMQ_TEMPLATE = 'https://localhost/api/Hubs/{}/Groups/{}/Projects/{}'

PROXIES = {'urls': {
    'http': 'http://user:password@127.0.0.1:5678',
    'https': 'https://user:password@127.0.0.1:5678'}
          }


# TODO: NamedTemporaryFiles do not support name in Windows
@skipIf(os.name == 'nt', 'Test not supported in Windows')
class TestIBMQAccounts(QiskitTestCase):
    """Tests for the IBMQ account handling."""
    def test_enable_account(self):
        """Test enabling one account."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN', url='someurl',
                                       proxies=PROXIES)

            # Compare the session accounts with the ones stored in file.
            loaded_accounts = read_credentials_from_qiskitrc()
            _, provider = list(qiskit.IBMQ._accounts.items())[0]

            self.assertEqual(loaded_accounts, {})
            self.assertEqual('QISKITRC_TOKEN', provider.credentials.token)
            self.assertEqual('someurl', provider.credentials.url)
            self.assertEqual(PROXIES, provider.credentials.proxies)

    def test_enable_multiple_accounts(self):
        """Test enabling multiple accounts, combining QX and IBMQ."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN')
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN',
                                       url=IBMQ_TEMPLATE.format('a', 'b', 'c'))
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN',
                                       url=IBMQ_TEMPLATE.format('a', 'b', 'X'))

            # Compare the session accounts with the ones stored in file.
            loaded_accounts = read_credentials_from_qiskitrc()
            self.assertEqual(loaded_accounts, {})
            self.assertEqual(len(qiskit.IBMQ._accounts), 3)

    def test_enable_duplicate_accounts(self):
        """Test enabling the same credentials twice."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN')

            self.assertEqual(len(qiskit.IBMQ._accounts), 1)

    def test_save_account(self):
        """Test saving one account."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.save_account('QISKITRC_TOKEN', url=QE_URL,
                                     proxies=PROXIES)

            # Compare the session accounts with the ones stored in file.
            stored_accounts = read_credentials_from_qiskitrc()
            self.assertEqual(len(stored_accounts.keys()), 1)

    def test_save_multiple_accounts(self):
        """Test saving several accounts, combining QX and IBMQ"""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.save_account('QISKITRC_TOKEN')
            qiskit.IBMQ.save_account('QISKITRC_TOKEN',
                                     url=IBMQ_TEMPLATE.format('a', 'b', 'c'))
            qiskit.IBMQ.save_account('QISKITRC_TOKEN',
                                     url=IBMQ_TEMPLATE.format('a', 'b', 'X'))

            # Compare the session accounts with the ones stored in file.
            stored_accounts = read_credentials_from_qiskitrc()
            self.assertEqual(len(stored_accounts), 3)
            for account_name, provider in qiskit.IBMQ._accounts.items():
                self.assertEqual(provider.credentials,
                                 stored_accounts[account_name])

    def test_save_duplicate_accounts(self):
        """Test saving the same credentials twice."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.save_account('QISKITRC_TOKEN')
            qiskit.IBMQ.save_account('QISKITRC_TOKEN')

            # Compare the session accounts with the ones stored in file.
            stored_accounts = read_credentials_from_qiskitrc()
            self.assertEqual(len(stored_accounts), 1)

    def test_disable_accounts(self):
        """Test disabling an account in a session."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN')
            qiskit.IBMQ.disable_accounts(token='QISKITRC_TOKEN')

            self.assertEqual(len(qiskit.IBMQ._accounts), 0)

    def test_delete_accounts(self):
        """Test deleting an account from disk."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.save_account('QISKITRC_TOKEN')
            self.assertEqual(len(read_credentials_from_qiskitrc()), 1)

            qiskit.IBMQ._accounts.clear()
            qiskit.IBMQ.delete_accounts(token='QISKITRC_TOKEN')
            self.assertEqual(len(read_credentials_from_qiskitrc()), 0)

    def test_disable_all_accounts(self):
        """Test disabling all accounts from session."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN')
            qiskit.IBMQ.enable_account('QISKITRC_TOKEN',
                                       url=IBMQ_TEMPLATE.format('a', 'b', 'c'))
            qiskit.IBMQ.disable_accounts()
            self.assertEqual(len(qiskit.IBMQ._accounts), 0)

    def test_delete_all_accounts(self):
        """Test deleting all accounts from disk."""
        with custom_qiskitrc(), mock_ibmq_provider():
            qiskit.IBMQ.save_account('QISKITRC_TOKEN')
            qiskit.IBMQ.save_account('QISKITRC_TOKEN',
                                     url=IBMQ_TEMPLATE.format('a', 'b', 'c'))
            self.assertEqual(len(read_credentials_from_qiskitrc()), 2)
            qiskit.IBMQ.delete_accounts()
            self.assertEqual(len(qiskit.IBMQ._accounts), 0)
            self.assertEqual(len(read_credentials_from_qiskitrc()), 0)

    def test_pass_bad_proxy(self):
        """Test proxy pass through."""
        failed = False
        try:
            qiskit.IBMQ.enable_account('dummy_token', 'https://dummy_url', proxies=PROXIES)
        except ConnectionError as excep:
            if 'ProxyError' in str(excep):
                failed = True
        self.assertTrue(failed)


# TODO: NamedTemporaryFiles do not support name in Windows
@skipIf(os.name == 'nt', 'Test not supported in Windows')
class TestCredentials(QiskitTestCase):
    """Tests for the credential subsystem."""

    def test_autoregister_no_credentials(self):
        """Test register() with no credentials available."""
        with no_file('Qconfig.py'), custom_qiskitrc(), no_envs():
            with self.assertRaises(QiskitError) as context_manager:
                qiskit.IBMQ.load_accounts()

        self.assertIn('No IBMQ credentials found', str(context_manager.exception))

    def test_store_credentials_overwrite(self):
        """Test overwriting qiskitrc credentials."""
        credentials = Credentials('QISKITRC_TOKEN', url=QE_URL, hub='HUB')
        credentials2 = Credentials('QISKITRC_TOKEN_2', url=QE_URL)

        with custom_qiskitrc():
            store_credentials(credentials)
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Attempt overwriting.
            with warnings.catch_warnings(record=True) as w:
                store_credentials(credentials)
                self.assertIn('already present', str(w[0]))

            with no_file('Qconfig.py'), no_envs(), mock_ibmq_provider():
                # Attempt overwriting.
                store_credentials(credentials2, overwrite=True)
                qiskit.IBMQ.load_accounts()

        # Ensure that the credentials are the overwritten ones - note that the
        # 'hub' parameter was removed.
        self.assertEqual(len(qiskit.IBMQ._accounts), 1)
        self.assertEqual(list(qiskit.IBMQ._accounts.values())[0].credentials.token,
                         'QISKITRC_TOKEN_2')

    def test_environ_over_qiskitrc(self):
        """Test order, without qconfig"""
        credentials = Credentials('QISKITRC_TOKEN', url=QE_URL)

        with custom_qiskitrc():
            # Prepare the credentials: both env and qiskitrc present
            store_credentials(credentials)
            with no_file('Qconfig.py'), custom_envs({'QE_TOKEN': 'ENVIRON_TOKEN',
                                                     'QE_URL': 'ENVIRON_URL'}):
                credentials = discover_credentials()

        self.assertEqual(len(credentials), 1)
        self.assertEqual(list(credentials.values())[0].token, 'ENVIRON_TOKEN')

    def test_qconfig_over_all(self):
        """Test order, with qconfig"""
        credentials = Credentials('QISKITRC_TOKEN', url=QE_URL)

        with custom_qiskitrc():
            # Prepare the credentials: qconfig, env and qiskitrc present
            store_credentials(credentials)
            with custom_qconfig(b"APItoken='QCONFIG_TOKEN'"),\
                    custom_envs({'QE_TOKEN': 'ENVIRON_TOKEN'}):
                credentials = discover_credentials()

        self.assertEqual(len(credentials), 1)
        self.assertEqual(list(credentials.values())[0].token, 'QCONFIG_TOKEN')


# Context managers

@contextmanager
def no_file(filename):
    """Context manager that disallows access to a file."""
    def side_effect(filename_):
        """Return False for the specified file."""
        if filename_ == filename:
            return False
        return isfile_original(filename_)

    # Store the original `os.path.isfile` function, for mocking.
    isfile_original = os.path.isfile
    patcher = patch('os.path.isfile', side_effect=side_effect)
    patcher.start()
    yield
    patcher.stop()


@contextmanager
def no_envs():
    """Context manager that disables qiskit environment variables."""
    # Remove the original variables from `os.environ`.
    # Store the original `os.environ`.
    os_environ_original = os.environ.copy()
    modified_environ = {key: value for key, value in os.environ.items()
                        if key not in VARIABLES_MAP.keys()}
    os.environ = modified_environ
    yield
    # Restore the original `os.environ`.
    os.environ = os_environ_original


@contextmanager
def custom_qiskitrc(contents=b''):
    """Context manager that uses a temporary qiskitrc."""
    # Create a temporary file with the contents.
    tmp_file = NamedTemporaryFile()
    tmp_file.write(contents)
    tmp_file.flush()

    # Temporarily modify the default location of the qiskitrc file.
    default_qiskitrc_file_original = _configrc.DEFAULT_QISKITRC_FILE
    _configrc.DEFAULT_QISKITRC_FILE = tmp_file.name
    yield

    # Delete the temporary file and restore the default location.
    tmp_file.close()
    _configrc.DEFAULT_QISKITRC_FILE = default_qiskitrc_file_original


@contextmanager
def custom_qconfig(contents=b''):
    """Context manager that uses a temporary qconfig.py."""
    # Create a temporary file with the contents.
    tmp_file = NamedTemporaryFile(suffix='.py')
    tmp_file.write(contents)
    tmp_file.flush()

    # Temporarily modify the default location of the qiskitrc file.
    default_qconfig_file_original = _qconfig.DEFAULT_QCONFIG_FILE
    _qconfig.DEFAULT_QCONFIG_FILE = tmp_file.name
    yield

    # Delete the temporary file and restore the default location.
    tmp_file.close()
    _qconfig.DEFAULT_QCONFIG_FILE = default_qconfig_file_original


@contextmanager
def custom_envs(new_environ):
    """Context manager that disables qiskit environment variables."""
    # Remove the original variables from `os.environ`.
    # Store the original `os.environ`.
    os_environ_original = os.environ.copy()
    modified_environ = {**os.environ, **new_environ}
    os.environ = modified_environ
    yield
    # Restore the original `os.environ`.
    os.environ = os_environ_original


@contextmanager
def mock_ibmq_provider():
    """Mock the initialization of IBMQSingleProvider, so it does not query the api."""
    patcher = patch.object(IBMQSingleProvider, '_authenticate', return_value=None)
    patcher2 = patch.object(IBMQSingleProvider, '_discover_remote_backends', return_value={})
    patcher.start()
    patcher2.start()
    yield
    patcher2.stop()
    patcher.stop()
