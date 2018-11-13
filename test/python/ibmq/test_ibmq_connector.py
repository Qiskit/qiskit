# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test IBMQConnector."""

import re
import unittest

from qiskit.backends.ibmq.api import (ApiError, BadBackendError,
                                      IBMQConnector, RegisterSizeError)
from ..common import QiskitTestCase, requires_qe_access


class TestIBMQConnector(QiskitTestCase):
    """Tests for IBMQConnector."""

    def setUp(self):
        self.qasm = SAMPLE_QASM_1
        self.qasms = [{'qasm': SAMPLE_QASM_1},
                      {'qasm': SAMPLE_QASM_2}]

    @requires_qe_access
    def test_api_auth_token(self, qe_token, qe_url):
        """
        Authentication with Quantum Experience Platform
        """
        api = self._get_api(qe_token, qe_url)
        credential = api.check_credentials()
        self.assertTrue(credential)

    def test_api_auth_token_fail(self):
        """
        Authentication with Quantum Experience Platform
        """
        self.assertRaises(ApiError,
                          IBMQConnector, 'fail')

    @requires_qe_access
    def test_api_run_job(self, qe_token, qe_url):
        """
        Check run an job by user authenticated
        """
        api = self._get_api(qe_token, qe_url)
        backend = 'ibmq_qasm_simulator'
        shots = 1
        job = api.run_job(self.qasms, backend, shots)
        check_status = None
        if 'status' in job:
            check_status = job['status']
        self.assertIsNotNone(check_status)

    @requires_qe_access
    def test_api_run_job_fail_backend(self, qe_token, qe_url):
        """
        Check run an job by user authenticated is not run because the backend
        does not exist
        """
        api = self._get_api(qe_token, qe_url)
        backend = 'INVALID_BACKEND'
        shots = 1
        self.assertRaises(BadBackendError, api.run_job, self.qasms,
                          backend, shots)

    @requires_qe_access
    def test_api_get_jobs(self, qe_token, qe_url):
        """
        Check get jobs by user authenticated
        """
        api = self._get_api(qe_token, qe_url)
        jobs = api.get_jobs(2)
        self.assertEqual(len(jobs), 2)

    @requires_qe_access
    def test_api_get_status_jobs(self, qe_token, qe_url):
        """
        Check get status jobs by user authenticated
        """
        api = self._get_api(qe_token, qe_url)
        jobs = api.get_status_jobs(1)
        self.assertEqual(len(jobs), 1)

    @requires_qe_access
    def test_api_backend_status(self, qe_token, qe_url):
        """
        Check the status of a real chip
        """
        backend_name = ('ibmq_20_tokyo'
                        if self.using_ibmq_credentials else 'ibmqx4')
        api = self._get_api(qe_token, qe_url)
        is_available = api.backend_status(backend_name)
        self.assertIsNotNone(is_available['available'])

    @requires_qe_access
    def test_api_backend_calibration(self, qe_token, qe_url):
        """
        Check the calibration of a real chip
        """
        backend_name = ('ibmq_20_tokyo'
                        if self.using_ibmq_credentials else 'ibmqx4')
        api = self._get_api(qe_token, qe_url)
        calibration = api.backend_calibration(backend_name)
        self.assertIsNotNone(calibration)

    @requires_qe_access
    def test_api_backend_parameters(self, qe_token, qe_url):
        """
        Check the parameters of calibration of a real chip
        """
        backend_name = ('ibmq_20_tokyo'
                        if self.using_ibmq_credentials else 'ibmqx4')
        api = self._get_api(qe_token, qe_url)
        parameters = api.backend_parameters(backend_name)
        self.assertIsNotNone(parameters)

    @requires_qe_access
    def test_api_backends_available(self, qe_token, qe_url):
        """
        Check the backends available
        """
        api = self._get_api(qe_token, qe_url)
        backends = api.available_backends()
        self.assertGreaterEqual(len(backends), 1)

    @requires_qe_access
    def test_register_size_limit_exception(self, qe_token, qe_url):
        """
        Check that exceeding register size limit generates exception
        """
        api = self._get_api(qe_token, qe_url)
        backend = 'ibmq_qasm_simulator'
        shots = 1
        qasm = SAMPLE_QASM_3
        self.assertRaises(RegisterSizeError, api.run_job,
                          [{'qasm': qasm}],
                          backend, shots)

    @requires_qe_access
    def test_qx_api_version(self, qe_token, qe_url):
        """
        Check the version of the QX API
        """
        api = self._get_api(qe_token, qe_url)
        version = api.api_version()
        self.assertGreaterEqual(int(version.split(".")[0]), 4)

    @staticmethod
    def _get_api(qe_token, qe_url):
        """Helper for instantating an IBMQConnector."""
        return IBMQConnector(qe_token, config={'url': qe_url})


class TestAuthentication(QiskitTestCase):
    """
    Tests for the authentication features. These tests are in a separate
    TestCase as they need to control the instantiation of
    `IBMQConnector` directly.
    """
    @requires_qe_access
    def test_url_404(self, qe_token, qe_url):
        """Test accessing a 404 URL"""
        url_404 = re.sub(r'/api.*$', '/api/TEST_404', qe_url)
        with self.assertRaises(ApiError):
            _ = IBMQConnector(qe_token,
                              config={'url': url_404})

    def test_invalid_token(self):
        """Test using an invalid token"""
        with self.assertRaises(ApiError):
            _ = IBMQConnector('INVALID_TOKEN')

    @requires_qe_access
    def test_url_unreachable(self, qe_token, qe_url):
        """Test accessing an invalid URL"""
        # pylint: disable=unused-argument
        with self.assertRaises(ApiError):
            _ = IBMQConnector(qe_token, config={'url': 'INVALID_URL'})


SAMPLE_QASM_1 = """IBMQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
u2(-4*pi/3,2*pi) q[0];
u2(-3*pi/2,2*pi) q[0];
u3(-pi,0,-pi) q[0];
u3(-pi,0,-pi/2) q[0];
u2(pi,-pi/2) q[0];
u3(-pi,0,-pi/2) q[0];
measure q -> c;
"""

SAMPLE_QASM_2 = """IBMQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[3];
creg f[2];
x q[0];
measure q[0] -> c[0];
measure q[2] -> f[0];
"""

SAMPLE_QASM_3 = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[45];
creg c[45];
h q[0];
h q[44];
measure q[0] -> c[0];
measure q[44] -> c[44];
"""


if __name__ == '__main__':
    unittest.main(verbosity=2)
