# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Quantum Program QISKit Test."""

import unittest
from IBMQuantumExperience import IBMQuantumExperience

import qiskit.backends
from .common import requires_qe_access, QiskitTestCase


class TestBackends(QiskitTestCase):
    """QISKit Backends (Object) Tests."""

    def test_local_backends_discover(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        local = qiskit.backends.discover_local_backends()
        self.log.info(local)
        self.assertTrue(local)

    def test_local_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        local = qiskit.backends.local_backends()
        self.log.info(local)
        self.assertTrue(local)

    @requires_qe_access
    def test_remote_backends_discover(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends to be discovered.

        If all correct some should exists.
        """
        api = IBMQuantumExperience(QE_TOKEN, {'url': QE_URL})
        remote = qiskit.backends.discover_remote_backends(api)
        self.log.info(remote)
        self.assertTrue(remote)

    @requires_qe_access
    def test_remote_backends_exist(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends.

        If all correct some should exists.
        """
        api = IBMQuantumExperience(QE_TOKEN, {'url': QE_URL})
        qiskit.backends.discover_remote_backends(api)
        remote = qiskit.backends.remote_backends()
        self.log.info(remote)
        self.assertTrue(remote)

    def test_backend_status(self):
        """Test backend_status.

        If all correct should return dictionary with available: True/False.
        """
        my_backend = qiskit.backends.get_backend_instance('local_qasm_simulator')
        out = my_backend.status
        self.assertIn(out['available'], [True])

    def test_get_backend_configuration(self):
        """Test configuration.

        If all correct should return configuration for the
        local_qasm_simulator.
        """
        my_backend = qiskit.backends.get_backend_instance('local_qasm_simulator')
        backend_config = my_backend.configuration
        config_keys = {'name', 'simulator', 'local', 'description',
                       'coupling_map', 'basis_gates'}
        self.assertTrue(config_keys < backend_config.keys())

    @requires_qe_access
    def test_get_backend_configuration_online(self, QE_TOKEN, QE_URL):
        """Test configuration.

        If all correct should return configuration for the
        local_qasm_simulator.
        """
        api = IBMQuantumExperience(QE_TOKEN, {'url': QE_URL})
        backend_list = qiskit.backends.discover_remote_backends(api)
        config_keys = {'name', 'simulator', 'local', 'description',
                       'coupling_map', 'basis_gates'}
        if backend_list:
            backend = backend_list[0]
        my_backend = qiskit.backends.get_backend_instance(backend)
        backend_config = my_backend.configuration
        self.log.info(backend_config)
        self.assertTrue(config_keys < backend_config.keys())

    @requires_qe_access
    def test_get_backend_calibration(self, QE_TOKEN, QE_URL):
        """Test calibration.

        If all correct should return dictionary on length 4.
        """
        api = IBMQuantumExperience(QE_TOKEN, {'url': QE_URL})
        backend_list = qiskit.backends.discover_remote_backends(api)
        if backend_list:
            backend = backend_list[0]
        my_backend = qiskit.backends.get_backend_instance(backend)
        result = my_backend.calibration
        self.log.info(result)
        self.assertEqual(len(result), 4)

    @requires_qe_access
    def test_get_backend_parameters(self, QE_TOKEN, QE_URL):
        """Test parameters.

        If all correct should return dictionary on length 4.
        """
        api = IBMQuantumExperience(QE_TOKEN, {'url': QE_URL})
        backend_list = qiskit.backends.discover_remote_backends(api)
        if backend_list:
            backend = backend_list[0]
        my_backend = qiskit.backends.get_backend_instance(backend)
        result = my_backend.parameters
        self.log.info(result)
        self.assertEqual(len(result), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
