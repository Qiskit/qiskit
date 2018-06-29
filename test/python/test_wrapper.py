# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Tests for the wrapper functionality."""

import logging
import unittest

import qiskit.wrapper
from qiskit.wrapper import registered_providers
from qiskit.backends.ibmq import IBMQProvider
from qiskit import QISKitError
from .common import QiskitTestCase, requires_qe_access
from .test_backends import remove_backends_from_list


class TestWrapper(QiskitTestCase):
    """Wrapper test case."""
    @requires_qe_access
    def test_wrapper_register_ok(self, QE_TOKEN, QE_URL, hub, group, project):
        """Test wrapper.register()."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)
        backends = qiskit.wrapper.available_backends()
        backends = remove_backends_from_list(backends)
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    @requires_qe_access
    def test_backends_with_filter(self, QE_TOKEN, QE_URL, hub, group, project):
        """Test wrapper.available_backends(filter=...)."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)
        backends = qiskit.wrapper.available_backends({'local': False,
                                                      'simulator': True})
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    def test_local_backends(self):
        """Test wrapper.local_backends(filter=...)."""
        local_backends = qiskit.wrapper.local_backends()
        self.log.info(local_backends)
        self.assertTrue(len(local_backends) > 0)

    @requires_qe_access
    def test_register_twice(self, QE_TOKEN, QE_URL, hub, group, project):
        """Test double registration of the same credentials."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)
        initial_providers = registered_providers()
        with self.assertRaises(QISKitError):
            qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)
        self.assertCountEqual(initial_providers, registered_providers())

    def test_register_bad_credentials(self):
        """Test registering a provider with bad credentials."""
        initial_providers = registered_providers()
        with self.assertRaises(QISKitError):
            qiskit.wrapper.register('FAKE_TOKEN', 'http://unknown')
        self.assertEqual(initial_providers, registered_providers())

    @requires_qe_access
    def test_unregister(self, QE_TOKEN, QE_URL, hub, group, project):
        """Test unregistering."""
        initial_providers = registered_providers()
        ibmqprovider = qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)
        self.assertCountEqual(initial_providers + [ibmqprovider],
                              registered_providers())
        qiskit.wrapper.unregister(ibmqprovider)
        self.assertEqual(initial_providers, registered_providers())

    @requires_qe_access
    def test_unregister_non_existent(self, QE_TOKEN, QE_URL, hub, group, project):
        """Test unregistering a non existent provider."""
        initial_providers = registered_providers()
        ibmqprovider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        with self.assertRaises(QISKitError):
            qiskit.wrapper.unregister(ibmqprovider)
        self.assertEqual(initial_providers, registered_providers())

    @requires_qe_access
    def test_register_backend_name_conflicts(self, QE_TOKEN, QE_URL,
                                             hub, group, project):
        """Test backend name conflicts when registering."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)
        initial_providers = registered_providers()
        initial_backends = qiskit.wrapper.available_backends()
        ibmqx4_backend = qiskit.wrapper.get_backend('ibmqx4')
        with self.assertLogs(level=logging.WARNING) as logs:
            ibmqprovider2 = qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project)

        # Check that one, and only one warning has been issued.
        self.assertEqual(len(logs.records), 1)
        # Check that the provider has been registered.
        self.assertCountEqual(initial_providers + [ibmqprovider2],
                              registered_providers())
        # Check that no new backends have been added.
        self.assertCountEqual(initial_backends,
                              qiskit.wrapper.available_backends())

        # Check the name of the backend still refers to the previous one.
        self.assertEqual(ibmqx4_backend, qiskit.wrapper.get_backend('ibmqx4'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
