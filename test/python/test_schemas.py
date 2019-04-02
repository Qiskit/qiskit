# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Schemas test."""

import json
import logging
import os

from qiskit.validation.jsonschema.schema_validation import (
    validate_json_against_schema, _get_validator)
from qiskit.providers.models import (BackendConfiguration, BackendProperties,
                                     BackendStatus, JobStatus, PulseDefaults)
from qiskit.result import Result
from qiskit.test import QiskitTestCase, Path


logger = logging.getLogger(__name__)


class TestSchemaExamples(QiskitTestCase):
    """
    Tests schema validation
    """
    _json_examples_per_schema = {
        "backend_configuration": [
            "backend_configuration_openpulse_example.json",
            "backend_configuration_openqasm_example.json",
            "backend_configuration_openqasm_simulator_example.json"],
        "backend_properties": [
            "backend_properties_example.json"],
        "backend_status": [
            "backend_status_example.json"],
        "default_pulse_configuration": [
            "default_pulse_configuration_example.json"],
        "job_status": [
            "job_status_example.json"],
        "qobj": [
            "qobj_openpulse_example.json",
            "qobj_openqasm_example.json"],
        "result": [
            "result_openqasm_example.json",
            "result_openpulse_level_0_example.json",
            "result_openpulse_level_1_example.json",
            "result_statevector_simulator_example.json",
            "result_unitary_simulator_example.json"]
    }

    def setUp(self):
        self.examples_base_path = self._get_resource_path('examples',
                                                          Path.SCHEMAS)

    def test_examples_are_valid(self):
        """Validate example json files against respective schemas"""
        schemas = TestSchemaExamples._json_examples_per_schema
        for schema_name, examples in schemas.items():
            with self.subTest(schema_test=schema_name):
                for example_schema in examples:
                    with self.subTest(example=example_schema):
                        with open(os.path.join(self.examples_base_path,
                                               example_schema),
                                  'r') as example_file:
                            example = json.load(example_file)
                            msg = 'JSON failed validation of {}.'\
                                  'Set Qiskit log level to DEBUG'\
                                  'for further information.'\
                                  ''.format(schema_name)

                            validate_json_against_schema(example,
                                                         schema_name, msg)

                        # Check for validating examples using the qiskit models.
                        obj_map = {'result': Result,
                                   'backend_configuration': BackendConfiguration,
                                   'backend_properties': BackendProperties,
                                   'backend_status': BackendStatus,
                                   'job_status': JobStatus,
                                   'default_pulse_configuration': PulseDefaults}
                        cls = obj_map.get(schema_name, None)
                        if cls and 'openpulse' not in example_schema:
                            _ = cls.from_dict(example)

    def test_schemas_are_valid(self):
        """Validate that schemas are valid jsonschema"""
        schemas = TestSchemaExamples._json_examples_per_schema
        for schema_name in schemas:
            with self.subTest(schema_test=schema_name):
                _get_validator(schema_name, check_schema=True)
