# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.


"""BasicAer Backends Test."""

import json
import jsonschema

from qiskit import BasicAer
from qiskit.providers.builtinsimulators import SimulatorsProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.test import Path, QiskitTestCase


class TestBasicAerBackends(QiskitTestCase):
    """Qiskit BasicAer Backends (Object) Tests."""

    def test_builtin_simulators_backends_exist(self):
        """Test if there are local backends."""
        builtin_simulators = SimulatorsProvider()
        local = builtin_simulators.backends()
        self.assertTrue(len(local) > 0)

    def test_get_backend(self):
        """Test get backends."""
        backend = BasicAer.backends(name='qasm_simulator')[0]
        self.assertEqual(backend.name(), 'qasm_simulator')

    def test_builtin_simulators_backend_status(self):
        """Test backend_status."""
        schema_path = self._get_resource_path(
            'backend_status_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        for backend in BasicAer.backends():
            status = backend.status()
            jsonschema.validate(status.to_dict(), schema)

    def test_builtin_simulators_backend_configuration(self):
        """Test backend configuration."""
        schema_path = self._get_resource_path(
            'backend_configuration_schema.json', path=Path.SCHEMAS)
        with open(schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        builtin_simulators = BasicAer.backends()
        for backend in builtin_simulators:
            configuration = backend.configuration()
            jsonschema.validate(configuration.to_dict(), schema)

    def test_builtin_simulators_backend_properties(self):
        """Test backend properties."""
        simulators = BasicAer.backends()
        for backend in simulators:
            properties = backend.properties()
            self.assertEqual(properties, None)

    def test_deprecated(self):
        """Test that deprecated names map the same backends as the new names.
        """
        def _get_first_available_backend(provider, backend_names):
            """Gets the first available backend."""
            if isinstance(backend_names, str):
                backend_names = [backend_names]
            for backend_name in backend_names:
                try:
                    return provider.get_backend(backend_name).name()
                except QiskitBackendNotFoundError:
                    pass
            return None

        deprecated_names = BasicAer._deprecated_backend_names()
        for oldname, newname in deprecated_names.items():
            with self.subTest(oldname=oldname, newname=newname):
                try:
                    resolved_newname = _get_first_available_backend(BasicAer, newname)
                    real_backend = BasicAer.get_backend(resolved_newname)
                except QiskitBackendNotFoundError:
                    # The real name of the backend might not exist
                    pass
                else:
                    self.assertEqual(BasicAer.backends(oldname)[0], real_backend)

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(QiskitBackendNotFoundError, BasicAer.get_backend, 'bad_name')

    def test_aliases_return_empty_list(self):
        """Test backends() return an empty list if name is unknown."""
        self.assertEqual(BasicAer.backends("bad_name"), [])
