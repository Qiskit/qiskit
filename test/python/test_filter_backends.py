# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Backends Filtering Test."""

from qiskit.wrapper import register, available_backends, least_busy
from .common import requires_qe_access, QiskitTestCase


class TestBackendFilters(QiskitTestCase):
    """QISKit Backend Filtering Tests."""

    @requires_qe_access
    def test_filter_config_dict(self, qe_token, qe_url):
        """Test filtering by dictionary of configuration properties"""
        n_qubits = 20 if self.using_ibmq_credentials else 5

        register(qe_token, qe_url)
        filter_ = {'n_qubits': n_qubits, 'local': False}
        filtered_backends = available_backends(filter_)
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_status_config_dict(self, qe_token, qe_url):
        """Test filtering by dictionary of mixed status/configuration properties"""
        register(qe_token, qe_url)
        filter_ = {'operational': True, 'local': False, 'simulator': True}
        filtered_backends = available_backends(filter_)
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_config_callable(self, qe_token, qe_url):
        """Test filtering by lambda function on configuration properties"""
        register(qe_token, qe_url)
        filtered_backends = available_backends(lambda x: (not x.configuration()['simulator'] and
                                                          x.configuration()['n_qubits'] > 5))
        self.assertTrue(filtered_backends)

    @requires_qe_access
    def test_filter_least_busy(self, qe_token, qe_url):
        """Test filtering by least busy function"""
        register(qe_token, qe_url)
        filtered_backends = least_busy(available_backends())
        self.assertTrue(filtered_backends)
