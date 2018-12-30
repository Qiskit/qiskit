# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin


"""Tests for all IBMQ backends."""

import json
import jsonschema

from qiskit import IBMQ
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import compile
from qiskit.qobj import QobjHeader
from qiskit.test import Path, QiskitTestCase, requires_qe_access, slow_test


class TestIBMQBackends(QiskitTestCase):
    """Tests for all the IBMQ backends."""
    def setUp(self):
        super().setUp()

        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self.qc1 = QuantumCircuit(qr, cr, name='circuit0')
        self.qc1.h(qr[0])
        self.qc1.measure(qr, cr)

    @requires_qe_access
    def test_remote_backends_exist(self, qe_token, qe_url):
        """Test if there are remote backends."""
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends()
        self.assertTrue(len(remotes) > 0)

    @requires_qe_access
    def test_remote_backends_exist_real_device(self, qe_token, qe_url):
        """Test if there are remote backends that are devices."""
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=False)
        self.assertTrue(remotes)

    @requires_qe_access
    def test_remote_backends_exist_simulator(self, qe_token, qe_url):
        """Test if there are remote backends that are simulators."""
        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends(simulator=True)
        self.assertTrue(remotes)

    @requires_qe_access
    def test_remote_backend_status(self, qe_token, qe_url):
        """Test backend_status."""
        schema_path = self._get_resource_path(
            'backend_status_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        IBMQ.enable_account(qe_token, qe_url)
        for backend in IBMQ.backends():
            status = backend.status()
            jsonschema.validate(status.to_dict(), schema)

    @requires_qe_access
    def test_remote_backend_configuration(self, qe_token, qe_url):
        """Test backend configuration."""
        schema_path = self._get_resource_path(
            'backend_configuration_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        IBMQ.enable_account(qe_token, qe_url)
        remotes = IBMQ.backends()
        for backend in remotes:
            configuration = backend.configuration()
            jsonschema.validate(configuration.to_dict(), schema)

    @requires_qe_access
    def test_remote_backend_properties(self, qe_token, qe_url):
        """Test backend properties."""
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

    @requires_qe_access
    def test_qobj_headers_in_result_sims(self, qe_token, qe_url):
        """Test that the qobj headers are passed onto the results for sims."""
        IBMQ.enable_account(qe_token, qe_url)
        backends = IBMQ.backends(simulator=True)

        custom_qobj_header = {'x': 1, 'y': [1, 2, 3], 'z': {'a': 4}}

        for backend in backends:
            with self.subTest(backend=backend):
                qobj = compile(self.qc1, backend)

                # Update the Qobj header.
                qobj.header = QobjHeader.from_dict(custom_qobj_header)
                # Update the Qobj.experiment header.
                qobj.experiments[0].header.some_field = 'extra info'

                result = backend.run(qobj).result()
                self.assertEqual(result.header.to_dict(), custom_qobj_header)
                self.assertEqual(result.results[0].header.some_field,
                                 'extra info')

    @slow_test
    @requires_qe_access
    def test_qobj_headers_in_result_devices(self, qe_token, qe_url):
        """Test that the qobj headers are passed onto the results for devices."""
        IBMQ.enable_account(qe_token, qe_url)
        backends = IBMQ.backends(simulator=False)

        custom_qobj_header = {'x': 1, 'y': [1, 2, 3], 'z': {'a': 4}}

        for backend in backends:
            with self.subTest(backend=backend):
                qobj = compile(self.qc1, backend)

                # Update the Qobj header.
                qobj.header = QobjHeader.from_dict(custom_qobj_header)
                # Update the Qobj.experiment header.
                qobj.experiments[0].header.some_field = 'extra info'

                result = backend.run(qobj).result()
                self.assertEqual(result.header.to_dict(), custom_qobj_header)
                self.assertEqual(result.results[0].header.some_field,
                                 'extra info')
