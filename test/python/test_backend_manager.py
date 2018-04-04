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

"""Backend Manager Test."""

import unittest

import os
import json
import jsonschema
import qiskit


from .common import requires_qe_access, QiskitTestCase
_schema_dir = os.path.dirname(qiskit.__file__) + '/schemas/backends'


class TestBackendManager(QiskitTestCase):
    """QISKit Backends (Object) Tests."""

    def test_local_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        qiskit.register(None, package=qiskit)
        local = qiskit.available_backends({'local': True})
        # print(local)
        self.log.info(local)
        self.assertTrue(local)

    @requires_qe_access
    def test_remote_backends_exist(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends.

        If all correct some should exists.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False})
        # print(remote)
        self.log.info(remote)
        self.assertTrue(remote)

    @requires_qe_access
    def test_remote_backends_exist_device(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends that are devices.

        If all correct some should exists.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False, 'simulator': False})
        # print(remote)
        self.log.info(remote)
        self.assertTrue(remote)

    @requires_qe_access
    def test_remote_backends_exist_simulator(self, QE_TOKEN, QE_URL):
        """Test if there are remote backends that are simulators.

        If all correct some should exists.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False, 'simulator': True})
        # print(remote)
        self.log.info(remote)
        self.assertTrue(remote)

    def test_get_backend(self):
        """Test get backends.

        If all correct should return a name the same as input.
        """
        qiskit.register(None, package=qiskit)
        local = qiskit.available_backends({'local': True})
        backend = qiskit.get_backend(local[0])
        self.assertEqual(local[0], backend.configuration['name'])

    def test_local_backend_status(self):
        """Test backend_status.

        If all correct should pass the vaildation.
        """
        qiskit.register(None, package=qiskit)
        local = qiskit.available_backends({'local': True})
        for backend in local:
            my_backend = qiskit.get_backend(backend)
            status = my_backend.status
            schema_path = os.path.join(_schema_dir, 'backend_status_schema_py.json')
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(status, schema)

    @requires_qe_access
    def test_remote_backend_status(self, QE_TOKEN, QE_URL):
        """Test backend_status.

        If all correct should pass the validation.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False})
        for backend in remote:
            my_backend = qiskit.get_backend(backend)
            status = my_backend.status
            # print(status)
            schema_path = os.path.join(_schema_dir, 'backend_status_schema_py.json')
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(status, schema)

    def test_local_backend_configuration(self):
        """Test backend configuration.

        If all correct should pass the vaildation.
        """
        qiskit.register(None, package=qiskit)
        local = qiskit.available_backends({'local': True})
        for backend in local:
            my_backend = qiskit.get_backend(backend)
            configuration = my_backend.configuration
            # print(configuration)
            schema_path = os.path.join(_schema_dir, 'backend_configuration_schema_old_py.json')
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    @requires_qe_access
    def test_remote_backend_configuration(self, QE_TOKEN, QE_URL):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False})
        for backend in remote:
            my_backend = qiskit.get_backend(backend)
            configuration = my_backend.configuration
            # print(configuration)
            schema_path = os.path.join(_schema_dir, 'backend_configuration_schema_old_py.json')
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    def test_local_backend_calibration(self):
        """Test backend calibration.

        If all correct should pass the vaildation.
        """
        qiskit.register(None, package=qiskit)
        local = qiskit.available_backends({'local': True})
        for backend in local:
            my_backend = qiskit.get_backend(backend)
            calibration = my_backend.calibration
            # print(calibration)
            # FIXME test against schema and decide what calibration
            # is for a simulator
            self.assertEqual(len(calibration), 0)

    @requires_qe_access
    def test_remote_backend_calibration(self, QE_TOKEN, QE_URL):
        """Test backend calibration.

        If all correct should pass the validation.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False})
        for backend in remote:
            my_backend = qiskit.get_backend(backend)
            calibration = my_backend.calibration
            # print(calibration)
            # FIXME test against schema and decide what calibration
            # is for a simulator
            # print(len(calibration))
            if my_backend.configuration['simulator']:
                self.assertEqual(len(calibration), 0)
            else:
                self.assertEqual(len(calibration), 4)

    def test_local_backend_parameters(self):
        """Test backend parameters.

        If all correct should pass the vaildation.
        """
        qiskit.register(None, package=qiskit)
        local = qiskit.available_backends({'local': True})
        for backend in local:
            my_backend = qiskit.get_backend(backend)
            parameters = my_backend.parameters
            # print(parameters)
            # FIXME test against schema and decide what parameters
            # is for a simulator
            self.assertEqual(len(parameters), 0)

    @requires_qe_access
    def test_remote_backend_parameters(self, QE_TOKEN, QE_URL):
        """Test backend parameters.

        If all correct should pass the validation.
        """
        qiskit.register(QE_TOKEN, QE_URL, package=qiskit)
        remote = qiskit.available_backends({'local': False})
        for backend in remote:
            my_backend = qiskit.get_backend(backend)
            parameters = my_backend.parameters
            # print(parameters)
            # FIXME test against schema and decide what parameters
            # is for a simulator
            # print(len(parameters))
            if my_backend.configuration['simulator']:
                self.assertEqual(len(parameters), 0)
            else:
                self.assertEqual(len(parameters), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
