# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Backends Test."""

import json
import jsonschema

from qiskit import IBMQ, BasicAer
from qiskit.backends.builtinsimulators import SimulatorsProvider
from .common import Path, QiskitTestCase, requires_qe_access


class TestBackends(QiskitTestCase):
    """Qiskit Backends (Object) Tests."""

    def test_builtin_simulators_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        builtin_simulators = SimulatorsProvider()
        local = builtin_simulators.backends()
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
        backend = BasicAer.backends(name='qasm_simulator')[0]
        self.assertEqual(backend.name(), 'qasm_simulator')

    def test_builtin_simulators_backend_status(self):
        """Test backend_status.

        If all correct should pass the validation.
        """
        schema_path = self._get_resource_path(
            'backend_status_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        for backend in BasicAer.backends():
            status = backend.status()
            jsonschema.validate(status.to_dict(), schema)

    @requires_qe_access
    def test_remote_backend_status(self, qe_token, qe_url):
        """Test backend_status.

        If all correct should pass the validation.
        """
        schema_path = self._get_resource_path(
            'backend_status_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        IBMQ.enable_account(qe_token, qe_url)
        for backend in IBMQ.backends():
            status = backend.status()
            jsonschema.validate(status.to_dict(), schema)

    def test_builtin_simulators_backend_configuration(self):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        schema_path = self._get_resource_path(
            'backend_configuration_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        builtin_simulators = BasicAer.backends()
        for backend in builtin_simulators:
            configuration = backend.configuration()
            jsonschema.validate(configuration.to_dict(), schema)

    @requires_qe_access
    def test_remote_backend_configuration(self, qe_token, qe_url):
        """Test backend configuration.

        If all correct should pass the validation.
        """
        schema_path = self._get_resource_path(
            'backend_configuration_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends()
        for backend in remotes:
            configuration = backend.configuration()
            jsonschema.validate(configuration.to_dict(), schema)

    def test_builtin_simulators_backend_properties(self):
        """Test backend properties.

        If all correct should pass the validation.
        """
        simulators = BasicAer.backends()
        for backend in simulators:
            properties = backend.properties()
            self.assertEqual(properties, None)

    @requires_qe_access
    def test_remote_backend_properties(self, qe_token, qe_url):
        """Test backend properties.

        If all correct should pass the validation.
        """
        schema_path = self._get_resource_path(
            'backend_properties_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=False)
        for backend in remotes:
            properties = backend.properties()
            if backend.configuration().simulator:
                self.assertEqual(properties, None)
            else:
                jsonschema.validate(properties.to_dict(), schema)
