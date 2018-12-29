# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""Backends Test."""

import json
import jsonschema

from qiskit import BasicAer
from qiskit.providers.builtinsimulators import SimulatorsProvider
from ..common import Path, QiskitTestCase


class TestBackends(QiskitTestCase):
    """Qiskit BasicAer Backends (Object) Tests."""

    def test_builtin_simulators_backends_exist(self):
        """Test if there are local backends.

        If all correct some should exists.
        """
        builtin_simulators = SimulatorsProvider()
        local = builtin_simulators.backends()
        self.assertTrue(len(local) > 0)

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

    def test_builtin_simulators_backend_properties(self):
        """Test backend properties.

        If all correct should pass the validation.
        """
        simulators = BasicAer.backends()
        for backend in simulators:
            properties = backend.properties()
            self.assertEqual(properties, None)
