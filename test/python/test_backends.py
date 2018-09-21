# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Backends Test."""

import json

import jsonschema

from qiskit.backends.ibmq import IBMQProvider
from qiskit.backends.local import LocalProvider
from qiskit.wrapper.defaultqiskitprovider import DefaultQISKitProvider
from .common import Path, QiskitTestCase, requires_qe_access


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
        local_provider = LocalProvider()
        local = local_provider.available_backends()
        self.log.info(local)
        self.assertTrue(len(local) > 0)

    @requires_qe_access
    def test_remote_backends_exist(self, qe_token, qe_url):
        """Test if there are remote backends.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        remotes = remove_backends_from_list(remotes)
        self.log.info(remotes)
        self.assertTrue(len(remotes) > 0)

    @requires_qe_access
    def test_remote_backends_exist_real_device(self, qe_token, qe_url):
        """Test if there are remote backends that are devices.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remote = ibmq_provider.available_backends()
        remote = [r for r in remote if not r.configuration()['simulator']]
        self.log.info(remote)
        self.assertTrue(remote)

    @requires_qe_access
    def test_remote_backends_exist_simulator(self, qe_token, qe_url):
        """Test if there are remote backends that are simulators.

        If all correct some should exists.
        """
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remote = ibmq_provider.available_backends()
        remote = [r for r in remote if r.configuration()['simulator']]
        self.log.info(remote)
        self.assertTrue(remote)

    def test_get_backend(self):
        """Test get backends.

        If all correct should return a name the same as input.
        """
        local_provider = DefaultQISKitProvider()
        backend = local_provider.get_backend(name='local_qasm_simulator_py')
        self.assertEqual(backend.configuration()['name'], 'local_qasm_simulator_py')

    def test_local_backend_status(self):
        """Test backend_status.

        If all correct should pass the vaildation.
        """
        # FIXME: reintroduce in 0.6
        self.skipTest('Skipping due to available vs operational')

        local_provider = DefaultQISKitProvider()
        backend = local_provider.get_backend(name='local_qasm_simulator')
        status = backend.status()
        schema_path = self._get_resource_path(
            'deprecated/backends/backend_status_schema_py.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        jsonschema.validate(status, schema)

    @requires_qe_access
    def test_remote_backend_status(self, qe_token, qe_url):
        """Test backend_status.

        If all correct should pass the validation.
        """
        # FIXME: reintroduce in 0.6
        self.skipTest('Skipping due to available vs operational')

        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            self.log.info(backend.status())
            status = backend.status()
            schema_path = self._get_resource_path(
                'deprecated/backends/backend_status_schema_py.json', path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(status, schema)

    def test_local_backend_configuration(self):
        """Test backend configuration.

        If all correct should pass the vaildation.
        """
        local_provider = LocalProvider()
        local_backends = local_provider.available_backends()
        for backend in local_backends:
            configuration = backend.configuration()
            schema_path = self._get_resource_path(
                'deprecated/backends/backend_configuration_schema_old_py.json',
                path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    @requires_qe_access
    def test_remote_backend_configuration(self, qe_token, qe_url):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            configuration = backend.configuration()
            schema_path = self._get_resource_path(
                'deprecated/backends/backend_configuration_schema_old_py.json', path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    def test_local_backend_properties(self):
        """Test backend properties.

        If all correct should pass the validation.
        """
        local_provider = LocalProvider()
        local_backends = local_provider.available_backends()
        for backend in local_backends:
            properties = backend.properties()
            # FIXME test against schema and decide what properties
            # is for a simulator
            self.assertEqual(len(properties), 0)

    @requires_qe_access
    def test_remote_backend_properties(self, qe_token, qe_url):
        """Test backend properties.

        If all correct should pass the validation.
        """
        ibmq_provider = IBMQProvider(qe_token, qe_url)
        remotes = ibmq_provider.available_backends()
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            self.log.info(backend.name())
            properties = backend.properties()
            # FIXME test against schema and decide what properties
            # is for a simulator
            if backend.configuration()['simulator']:
                self.assertEqual(len(properties), 0)
            else:
                self.assertTrue(all(key in properties for key in (
                    'last_update_date',
                    'qubits',
                    'backend')))
