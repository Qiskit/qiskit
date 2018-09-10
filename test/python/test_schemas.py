# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=redefined-builtin

""" Schemas test"""
import unittest
import json
import os
from qiskit import (compile, SchemaValidationError,
                    validate_json_against_schema, _create_validator)
from qiskit import __path__ as qiskit_path
import qiskit._schema_validation
from .common import QiskitTestCase, Path


class TestSchemaExamples(QiskitTestCase):
    """
    Tests schema valiation
    """

    def setUp(self):
        self.schema_tests = []
        self.schema_tests.append({"schema_name": "backend_configuration",
                             "examples": [
                                 "backend_configuration_openpulse_example.json",
                                 "backend_configuration_openqasm_example.json",
                                 "backend_configuration_openqasm_simulator_example.json"
                                 ]})
        self.schema_tests.append({"schema_name": "backend_properties",
                             "examples": [
                                 "backend_properties_example.json"
                                 ]})
        self.schema_tests.append({"schema_name": "backend_status",
                             "examples": [
                                 "backend_status_example.json"
                                 ]})
        self.schema_tests.append({"schema_name": "default_pulse_configuration",
                             "examples": [
                                 "default_pulse_configuration_example.json"
                                 ]})
        self.schema_tests.append({"schema_name": "job_status",
                             "examples": [
                                 "job_status_example.json"
                                 ]})
        self.schema_tests.append({"schema_name": "qobj",
                             "examples": [
                                 "qobj_openpulse_example.json",
                                 "qobj_openqasm_example.json"
                                 ]})
        self.schema_tests.append({"schema_name": "result",
                             "examples": [
                                 "result_openqasm_example.json",
                                 "result_openpulse_level_0_example.json",
                                 "result_openpulse_level_1_example.json",
                                 "result_statevector_simulator_example.json",
                                 "result_unitary_simulator_example.json"
                                 ]})
        self.examples_base_path = os.path.join(qiskit_path[0], 'schemas',
                                               'examples')

    def test_examples_are_valid(self):
        """ Validate example json files against respective schemas"""
        for schema_test in self.schema_tests:
            schema_name = schema_test['schema_name']
            with self.subTest(schema_test=schema_name):
                for example_schema in schema_test['examples']:
                    with self.subTest(example=example_schema):
                        with open(os.path.join(self.examples_base_path,
                                  example_schema), 'r') as example_file:
                            example = json.load(example_file)
                            msg = '''JSON failed validation of {}.
                                    Set Qiskit log level to DEBUG
                                    for further information.'''.format(schema_name)

                            validate_json_against_schema(example, schema_name, msg)

    def test_schemas_are_valid(self):
        """ Validate that schemas are valid jsonschema"""
        for schema_test in self.schema_tests:
            schema_name = schema_test['schema_name']
            with self.subTest(schema_test=schema_name):
                _create_validator(schema_name, check_schema=True)
