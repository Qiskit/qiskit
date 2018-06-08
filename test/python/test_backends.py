# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Backends Test."""

import json
import unittest

import jsonschema

import qiskit.wrapper
from qiskit.backends.ibmq import IBMQProvider
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from .common import requires_qe_access, QiskitTestCase, Path


def remove_backends_from_list(backends):
    """Helper and temporary function for removing specific backends from a list"""
    backends_to_remove = ['ibmq_qasm_simulator']
    return [backend for backend in backends if str(backend) not in backends_to_remove]


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
    def test_remote_backends_exist(self, QE_TOKEN, QE_URL,
                                   hub=None, group=None, project=None):
        """Test if there are remote backends.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remotes = ibmq_provider.available_backends({'local': False})
        remotes = remove_backends_from_list(remotes)
        self.log.info(remotes)
        self.assertTrue(len(remotes) > 0)

    @requires_qe_access
    def test_remote_backends_exist_real_device(self, QE_TOKEN, QE_URL,
                                               hub=None, group=None, project=None):
        """Test if there are remote backends that are devices.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remote = ibmq_provider.available_backends({'local': False, 'simulator': False})
        self.log.info(remote)
        self.assertTrue(remote)

    @requires_qe_access
    def test_remote_backends_exist_simulator(self, QE_TOKEN, QE_URL,
                                             hub=None, group=None, project=None):
        """Test if there are remote backends that are simulators.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remote = ibmq_provider.available_backends({'local': False, 'simulator': True})
        self.log.info(remote)
        self.assertTrue(remote)

    def test_get_backend(self):
        """Test get backends.

        If all correct should return a name the same as input.
        """
        local_provider = DefaultQISKitProvider()
        backend = local_provider.get_backend(name='local_qasm_simulator_py')
        self.assertEqual(backend.configuration['name'], 'local_qasm_simulator_py')

    def test_local_backend_status(self):
        """Test backend_status.

        If all correct should pass the vaildation.
        """
        local_provider = DefaultQISKitProvider()
        backend = local_provider.get_backend(name='local_qasm_simulator')
        status = backend.status
        schema_path = self._get_resource_path(
            'deprecated/backends/backend_status_schema_py.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        jsonschema.validate(status, schema)

    @requires_qe_access
    def test_remote_backend_status(self, QE_TOKEN, QE_URL,
                                   hub=None, group=None, project=None):
        """Test backend_status.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remotes = ibmq_provider.available_backends({'local': False})
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            status = backend.status
            schema_path = self._get_resource_path(
                'deprecated/backends/backend_status_schema_py.json', path=Path.SCHEMAS)
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
                'deprecated/backends/backend_configuration_schema_old_py.json',
                path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    @requires_qe_access
    def test_remote_backend_configuration(self, QE_TOKEN, QE_URL,
                                          hub=None, group=None, project=None):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remotes = ibmq_provider.available_backends({'local': False})
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            configuration = backend.configuration
            schema_path = self._get_resource_path(
                'deprecated/backends/backend_configuration_schema_old_py.json', path=Path.SCHEMAS)
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
    def test_remote_backend_calibration(self, QE_TOKEN, QE_URL,
                                        hub=None, group=None, project=None):
        """Test backend calibration.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remotes = ibmq_provider.available_backends({'local': False})
        remotes = remove_backends_from_list(remotes)
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
    def test_remote_backend_parameters(self, QE_TOKEN, QE_URL,
                                       hub=None, group=None, project=None):
        """Test backend parameters.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(QE_TOKEN, QE_URL, hub, group, project)
        remotes = ibmq_provider.available_backends({'local': False})
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            self.log.info(backend.name)
            parameters = backend.parameters
            # FIXME test against schema and decide what parameters
            # is for a simulator
            if backend.configuration['simulator']:
                self.assertEqual(len(parameters), 0)
            else:
                self.assertTrue(all(key in parameters for key in (
                    'last_update_date',
                    'qubits',
                    'backend')))

    @requires_qe_access
    def test_wrapper_register_ok(self, QE_TOKEN, QE_URL,
                                 hub=None, group=None, project=None):
        """Test wrapper.register()."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project, provider_name='ibmq')
        backends = qiskit.wrapper.available_backends()
        backends = remove_backends_from_list(backends)
        self.log.info(backends)
        self.assertTrue(len(backends) > 0)

    @requires_qe_access
    def test_wrapper_available_backends_with_filter(self, QE_TOKEN, QE_URL,
                                                    hub=None, group=None, project=None):
        """Test wrapper.available_backends(filter=...)."""
        qiskit.wrapper.register(QE_TOKEN, QE_URL, hub, group, project, provider_name='ibmq')
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
