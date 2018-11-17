# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

"""Schemas test."""
import json
import os
from qiskit._schema_validation import (validate_json_against_schema,
                                       _get_validator)
from qiskit import __path__ as qiskit_path
from qiskit.backends.models import (BackendConfiguration, BackendProperties,
                                    BackendStatus, JobStatus)
from qiskit.validation.result import Result
from .common import QiskitTestCase


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
        self.examples_base_path = os.path.join(qiskit_path[0], 'schemas',
                                               'examples')

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
                        # TODO: temporary quick check for validating examples
                        # using the qiskit.validation-based Result.
                        obj_map = {'result': Result,
                                   'backend_configuration': BackendConfiguration,
                                   'backend_properties': BackendProperties,
                                   'backend_status': BackendStatus,
                                   'job_status': JobStatus}
                        cls = obj_map.get(schema_name, None)
                        if cls and 'openpulse' not in example_schema:
                            _ = cls.from_dict(example)

    def test_schemas_are_valid(self):
        """Validate that schemas are valid jsonschema"""
        schemas = TestSchemaExamples._json_examples_per_schema
        for schema_name in schemas:
            with self.subTest(schema_test=schema_name):
                _get_validator(schema_name, check_schema=True)
