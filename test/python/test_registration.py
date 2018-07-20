# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Test the registration and credentials features of the wrapper.
"""
import os
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from unittest.mock import patch
from unittest import skipIf

import qiskit
from qiskit import QISKitError
from qiskit.backends.ibmq import IBMQProvider
from qiskit.wrapper.credentials import (_configrc, _qconfig,
                                        discover_credentials, get_account_name)
from qiskit.wrapper.credentials._environ import VARIABLES_MAP
from .common import QiskitTestCase


# TODO: NamedTemporaryFiles do not support name in Windows
@skipIf(os.name == 'nt', 'Test not supported in Windows')
class TestWrapperCredentials(QiskitTestCase):
    """Wrapper autoregistration and credentials test case."""
    def setUp(self):
        super(TestWrapperCredentials, self).setUp()
        self.ibmq_account_name = get_account_name(IBMQProvider)

    def test_autoregister_no_credentials(self):
        """Test register() with no credentials available."""
        with no_file('Qconfig.py'), custom_qiskitrc(), no_envs():
            with self.assertRaises(QISKitError) as cm:
                qiskit.wrapper.register()

        self.assertIn('No IBMQ credentials found', str(cm.exception))

    def test_store_credentials(self):
        """Test storing credentials and using them for autoregister."""
        with no_file('Qconfig.py'), no_envs(), custom_qiskitrc(), mock_ibmq_provider():
            qiskit.wrapper.store_credentials('QISKITRC_TOKEN', proxies={'http': 'foo'})
            provider = qiskit.register()

        self.assertEqual(provider._token, 'QISKITRC_TOKEN')
        self.assertEqual(provider._proxies, {'http': 'foo'})

    def test_store_credentials_overwrite(self):
        """Test overwritind qiskitrc credentials."""
        with custom_qiskitrc():
            qiskit.wrapper.store_credentials('QISKITRC_TOKEN', hub='HUB')
            # Attempt overwriting.
            with self.assertRaises(QISKitError) as cm:
                qiskit.wrapper.store_credentials('QISKITRC_TOKEN')
            self.assertIn('already present', str(cm.exception))

            with no_file('Qconfig.py'), no_envs(), mock_ibmq_provider():
                # Attempt overwriting.
                qiskit.wrapper.store_credentials('QISKITRC_TOKEN_2',
                                                 overwrite=True)
                provider = qiskit.wrapper.register()

        # Ensure that the credentials are the overwritten ones - note that the
        # 'hub' parameter was removed.
        self.assertEqual(provider._token, 'QISKITRC_TOKEN_2')
        self.assertEqual(provider._hub, None)

    def test_environ_over_qiskitrc(self):
        """Test order, without qconfig"""
        with custom_qiskitrc():
            # Prepare the credentials: both env and qiskitrc present
            qiskit.wrapper.store_credentials('QISKITRC_TOKEN')
            with no_file('Qconfig.py'), custom_envs({'QE_TOKEN': 'ENVIRON_TOKEN'}):
                credentials = discover_credentials()

        self.assertIn(self.ibmq_account_name, credentials)
        self.assertEqual(credentials[self.ibmq_account_name]['token'], 'ENVIRON_TOKEN')

    def test_qconfig_over_all(self):
        """Test order, with qconfig"""
        with custom_qiskitrc():
            # Prepare the credentials: qconfig, env and qiskitrc present
            qiskit.wrapper.store_credentials('QISKITRC_TOKEN')
            with custom_qconfig(b"APItoken='QCONFIG_TOKEN'"),\
                    custom_envs({'QE_TOKEN': 'ENVIRON_TOKEN'}):
                credentials = discover_credentials()

        self.assertIn(self.ibmq_account_name, credentials)
        self.assertEqual(credentials[self.ibmq_account_name]['token'], 'QCONFIG_TOKEN')


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
    DEFAULT_QISKITRC_FILE_original = _configrc.DEFAULT_QISKITRC_FILE
    _configrc.DEFAULT_QISKITRC_FILE = tmp_file.name
    yield

    # Delete the temporary file and restore the default location.
    tmp_file.close()
    _configrc.DEFAULT_QISKITRC_FILE = DEFAULT_QISKITRC_FILE_original


@contextmanager
def custom_qconfig(contents=b''):
    """Context manager that uses a temporary qconfig.py."""
    # Create a temporary file with the contents.
    tmp_file = NamedTemporaryFile(suffix='.py')
    tmp_file.write(contents)
    tmp_file.flush()

    # Temporarily modify the default location of the qiskitrc file.
    DEFAULT_QCONFIG_FILE_original = _qconfig.DEFAULT_QCONFIG_FILE
    _qconfig.DEFAULT_QCONFIG_FILE = tmp_file.name
    yield

    # Delete the temporary file and restore the default location.
    tmp_file.close()
    _qconfig.DEFAULT_QCONFIG_FILE = DEFAULT_QCONFIG_FILE_original


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
    """Mock the initialization of IBMQProvider, so it does not query the api."""
    patcher = patch.object(IBMQProvider, '_authenticate', return_value=None)
    patcher2 = patch.object(IBMQProvider, '_discover_remote_backends', return_value={})
    patcher.start()
    patcher2.start()
    yield
    patcher2.stop()
    patcher.stop()
