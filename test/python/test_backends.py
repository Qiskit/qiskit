# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Backends Test."""

import json

import jsonschema

from qiskit import IBMQ, Aer
from qiskit.backends.aer import AerProvider
from .common import Path, QiskitTestCase, requires_qe_access


def remove_backends_from_list(backends):
    """Helper and temporary function for removing specific backends from a list"""
    backends_to_remove = ['ibmq_qasm_simulator']
    return [backend for backend in backends if str(backend) not in backends_to_remove]


class TestBackends(QiskitTestCase):
    """QISKit Backends (Object) Tests."""

    def test_aer_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        aer_provider = AerProvider()
        local = aer_provider.backends()
        self.assertTrue(len(local) > 0)

    @requires_qe_access
    def test_remote_backends_exist(self, qe_token, qe_url):
        """Test if there are remote backends.

        If all correct some should exists.
        """
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends()
        self.assertTrue(len(remotes) > 0)

    @requires_qe_access
    def test_remote_backends_exist_real_device(self, qe_token, qe_url):
        """Test if there are remote backends that are devices.

        If all correct some should exists.
        """
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=False)
        self.assertTrue(remotes)

    @requires_qe_access
    def test_remote_backends_exist_simulator(self, qe_token, qe_url):
        """Test if there are remote backends that are simulators.

        If all correct some should exists.
        """
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=True)
        self.assertTrue(remotes)

    def test_get_backend(self):
        """Test get backends.

        If all correct should return a name the same as input.
        """
        backend = Aer.backends(name='qasm_simulator_py')[0]
        self.assertEqual(backend.name(), 'qasm_simulator_py')

    def test_aer_backend_status(self):
        """Test backend_status.

        If all correct should pass the validation.
        """
        backend = Aer.backends(name='qasm_simulator')[0]
        status = backend.status()
        schema_path = self._get_resource_path(
            'backend_status_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)
        jsonschema.validate(status, schema)

    @requires_qe_access
    def test_remote_backend_status(self, qe_token, qe_url):
        """Test backend_status.

        If all correct should pass the validation.
        """
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends()
        remotes = remove_backends_from_list(remotes)
        for backend in remotes:
            status = backend.status()
            self.log.debug(status)
            schema_path = self._get_resource_path(
                'backend_status_schema.json', path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(status, schema)

    def test_aer_backend_configuration(self):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        aer_backends = Aer.backends()
        for backend in aer_backends:
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
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=False)
        for backend in remotes:
            configuration = backend.configuration()
            schema_path = self._get_resource_path(
                'deprecated/backends/backend_configuration_schema_old_py.json', path=Path.SCHEMAS)
            with open(schema_path, 'r') as schema_file:
                schema = json.load(schema_file)
            jsonschema.validate(configuration, schema)

    def test_aer_backend_properties(self):
        """Test backend properties.

        If all correct should pass the validation.
        """
        aer_backends = Aer.backends()
        for backend in aer_backends:
            properties = backend.properties()
            # FIXME test against schema and decide what properties
            # is for a simulator
            self.assertEqual(len(properties), 0)

    @requires_qe_access
    def test_remote_backend_properties(self, qe_token, qe_url):
        """Test backend properties.

        If all correct should pass the validation.
        """
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=False)
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
