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

"""Backends Test."""

import json
import unittest

import jsonschema

import qiskit.wrapper
from qiskit.backends.ibmq import IBMQProvider
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from .common import requires_qe_access, QiskitTestCase, Path


class TestBackends(QiskitTestCase):
    """QISKit Backends (Object) Tests."""

    def test_local_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        local_provider = DefaultQISKitProvider()
        local = local_provider.available_backends({'local': True})
        self.log.info(local)
        self.assertTrue(len(local) > 0)

    @requires_qe_access
    def test_remote_backends_exist(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remote = ibmq_provider.available_backends({'local': False})
        self.log.info(remote)
        self.assertTrue(len(remote) > 0)

    @requires_qe_access
    def test_remote_backends_exist_real_device(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends that are devices.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remote = ibmq_provider.available_backends({'local': False, 'simulator': False})
        self.log.info(remote)
        self.assertTrue(remote)

    @requires_qe_access
    def test_remote_backends_exist_simulator(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends that are simulators.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remote = ibmq_provider.available_backends({'local': False, 'simulator': True})
        self.log.info(remote)
        self.assertTrue(remote)

    def test_get_backend(self):
        """Test get backends.

        If all correct should return a name the same as input.
        """
        local_provider = DefaultQISKitProvider()
        backend = local_provider.get_backend(name='local_qasm_simulator')
        self.assertEqual(backend.configuration['name'], 'local_qasm_simulator')

    def test_local_backend_status(self):
        """Test backend_status.

        If all correct should pass the vaildation.
        """
        local_provider = DefaultQISKitProvider()
        backend = local_provider.get_backend(name='local_qasm_simulator')
        status = backend.status
        schema_path = self._get_resource_path(
            'backends/backend_status_schema_py.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        jsonschema.validate(status, schema)

    @requires_qe_access
    def test_remote_backend_status(self, QE_TOKEN, QE_URL):
        """Test backend_status.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remotes = ibmq_provider.available_backends({'local': False})
        for backend in remotes:
            status = backend.status
            schema_path = self._get_resource_path(
                'backends/backend_status_schema_py.json', path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(status, schema)

    def test_local_backend_configuration(self):
        """Test backend configuration.

        If all correct should pass the vaildation.
        """
        qiskit_provider = DefaultQISKitProvider()
        local_backends = qiskit_provider.available_backends({'local': True})
        for backend in local_backends:
            configuration = backend.configuration
            schema_path = self._get_resource_path(
                'backends/backend_configuration_schema_old_py.json',
                path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    @requires_qe_access
    def test_remote_backend_configuration(self, QE_TOKEN, QE_URL):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remotes = ibmq_provider.available_backends({'local': False})
        for backend in remotes:
            configuration = backend.configuration
            schema_path = self._get_resource_path(
                'backends/backend_configuration_schema_old_py.json', path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    def test_local_backend_calibration(self):
        """Test backend calibration.

        If all correct should pass the vaildation.
        """
        qiskit_provider = DefaultQISKitProvider()
        local_backends = qiskit_provider.available_backends({'local': True})
        for backend in local_backends:
            calibration = backend.calibration
            # FIXME test against schema and decide what calibration
            # is for a simulator
            self.assertEqual(len(calibration), 0)

    @requires_qe_access
    def test_remote_backend_calibration(self, QE_TOKEN, QE_URL):
        """Test backend calibration.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remotes = ibmq_provider.available_backends({'local': False})
        for backend in remotes:
            calibration = backend.calibration
            # FIXME test against schema and decide what calibration
            # is for a simulator
            if backend.configuration['simulator']:
                self.assertEqual(len(calibration), 0)
            else:
                self.assertEqual(len(calibration), 4)

    def test_local_backend_parameters(self):
        """Test backend parameters.

        If all correct should pass the vaildation.
        """
        qiskit_provider = DefaultQISKitProvider()
        local_backends = qiskit_provider.available_backends({'local': True})
        for backend in local_backends:
            parameters = backend.parameters
            # FIXME test against schema and decide what parameters
            # is for a simulator
            self.assertEqual(len(parameters), 0)

    @requires_qe_access
    def test_remote_backend_parameters(self, QE_TOKEN, QE_URL):
        """Test backend parameters.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL)
        remotes = ibmq_provider.available_backends({'local': False})
        for backend in remotes:
            parameters = backend.parameters
            # FIXME test against schema and decide what parameters
            # is for a simulator
            if backend.configuration['simulator']:
                self.assertEqual(len(parameters), 0)
            else:
                self.assertEqual(len(parameters), 4)

    @requires_qe_access
    def test_wrapper_register_ok(self, QE_TOKEN, QE_URL):
        """Test wrapper.register()."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, provider_name='qiskit')
        backends = qiskit.wrapper.available_backends()
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    @requires_qe_access
    def test_wrapper_available_backends_with_filter(self, QE_TOKEN, QE_URL):
        """Test wrapper.available_backends(filter=...)."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, provider_name='qiskit')
        backends = qiskit.wrapper.available_backends({'local': False, 'simulator': True})
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    def test_wrapper_local_backends(self):
        """Test wrapper.local_backends(filter=...)."""
        local_backends = qiskit.wrapper.local_backends()
        self.log.info(local_backends)
        self.assertTrue(len(local_backends) > 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
