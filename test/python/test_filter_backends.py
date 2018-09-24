# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Backends Filtering Test."""

from qiskit import Aer, IBMQ
from qiskit.wrapper import register, available_backends, least_busy
from .common import requires_qe_access, QiskitTestCase


class TestBackendFilters(QiskitTestCase):
    """QISKit Backend Filtering Tests."""

    @requires_qe_access
    def test_filter_config_properties(self, qe_token, qe_url):
        """Test filtering by configuration properties"""
        n_qubits = 20 if self.using_ibmq_credentials else 5

        provider = IBMQ.use_account(qe_token, qe_url)
        filtered_backends = provider.backends(n_qubits=n_qubits, local=False)
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_status_dict(self, qe_token, qe_url):
        """Test filtering by dictionary of mixed status/configuration properties"""
        provider = IBMQ.use_account(qe_token, qe_url)
        filtered_backends = provider.backends(
            operational=True,  # from status
            local=False, simulator=True)  # from configuration

        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_config_callable(self, qe_token, qe_url):
        """Test filtering by lambda function on configuration properties"""
        provider = IBMQ.use_account(qe_token, qe_url)
        filtered_backends = provider.backends(
            filters=lambda x: (not x.configuration()['simulator']
                               and x.configuration()['n_qubits'] > 5))
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_least_busy(self, qe_token, qe_url):
        """Test filtering by least busy function"""
        provider = IBMQ.use_account(qe_token, qe_url)
        filtered_backends = provider.backends()
        names = [backend.name() for backend in filtered_backends]
        filtered_backends = least_busy(names)
        self.assertTrue(filtered_backends)
