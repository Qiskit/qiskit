# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Backends Filtering Test."""

from qiskit import IBMQ
from qiskit.backends.ibmq import least_busy
from .common import requires_qe_access, QiskitTestCase


class TestBackendFilters(QiskitTestCase):
    """Qiskit Backend Filtering Tests."""

    @requires_qe_access
    def test_filter_config_properties(self, qe_token, qe_url):
        """Test filtering by configuration properties"""
        n_qubits = 20 if self.using_ibmq_credentials else 5

        IBMQ.enable_account(qe_token, qe_url)
        filtered_backends = IBMQ.backends(n_qubits=n_qubits, local=False)
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_status_dict(self, qe_token, qe_url):
        """Test filtering by dictionary of mixed status/configuration properties"""
        IBMQ.enable_account(qe_token, qe_url)
        filtered_backends = IBMQ.backends(
            operational=True,  # from status
            local=False, simulator=True)  # from configuration

        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_config_callable(self, qe_token, qe_url):
        """Test filtering by lambda function on configuration properties"""
        IBMQ.enable_account(qe_token, qe_url)
        filtered_backends = IBMQ.backends(
            filters=lambda x: (not x.configuration().simulator
                               and x.configuration().n_qubits > 5))
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_least_busy(self, qe_token, qe_url):
        """Test filtering by least busy function"""
        IBMQ.enable_account(qe_token, qe_url)
        backends = IBMQ.backends()
        filtered_backends = least_busy(backends)
        self.assertTrue(filtered_backends)
