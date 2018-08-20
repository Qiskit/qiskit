# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Created on Sat Apr 21 01:58:06 2018

Test the schemas against the examples as a batch file.
Run as `python test_schemas.py'
"""

import os
import json
import jsonschema as jsch

schema_tests = []
verbose_err = False
any_error = False
cur_file_path = os.path.dirname(os.path.abspath(__file__))
# go two directories up
cur_file_path = os.path.dirname(cur_file_path)
cur_file_path = os.path.dirname(cur_file_path)
cur_file_path = os.path.join(cur_file_path, 'qiskit', 'schemas')
print(cur_file_path)

"""List the schemas and their examples."""
schema_tests.append({"schema": "backend_configuration_schema.json",
                     "examples": [
                         "backend_configuration_openpulse_example.json",
                         "backend_configuration_openqasm_example.json"
                         ]})
schema_tests.append({"schema": "backend_properties_schema.json",
                     "examples": [
                         "backend_properties_example.json"
                         ]})
schema_tests.append({"schema": "backend_status_schema.json",
                     "examples": [
                         "backend_status_example.json"
                         ]})
schema_tests.append({"schema": "default_pulse_configuration_schema.json",
                     "examples": [
                         "default_pulse_configuration_example.json"
                         ]})
schema_tests.append({"schema": "job_status_schema.json",
                     "examples": [
                         "job_status_example.json"
                         ]})
schema_tests.append({"schema": "qobj_schema.json",
                     "examples": [
                         "qobj_openpulse_example.json",
                         "qobj_openqasm_example.json"
                         ]})
schema_tests.append({"schema": "result_schema.json",
                     "examples": [
                         "result_openqasm_example.json",
                         "result_openpulse_level_0_example.json",
                         "result_openpulse_level_1_example.json",
                         "result_statevector_simulator_example.json",
                         "result_unitary_simulator_example.json"
                         ]})

"""Run through each schema."""
for schema_test in schema_tests:
    print('Schema: %s' % (schema_test['schema']))
    for example_schema in schema_test['examples']:
        schema = json.load(open(os.path.join(cur_file_path,
                                             schema_test['schema']), 'r'))
        example = json.load(open(os.path.join(cur_file_path, 'examples',
                                              example_schema), 'r'))
        try:
            jsch.validate(example, schema)
        except jsch.ValidationError as err:
            print("Error on example %s:" % example_schema)
            any_error = True
            if verbose_err:
                print(err)
            continue
        print('Passed: %s' % (example_schema))
if not any_error:
    print('ALL SCHEMAS PASSED')
