# -*- coding: utf-8 -*-
# Copyright 2018 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Created on Sat Apr 21 01:58:06 2018

@author: dcmckay

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
schema_tests.append({"schema": "backend_config_schema.json",
                     "examples": [
                         "backend_config_example_openpulse.json",
                         "backend_config_example_openqasm.json"
                         ]})
schema_tests.append({"schema": "backend_props_schema.json",
                     "examples": [
                         "backend_props_example.json"
                         ]})
schema_tests.append({"schema": "backend_status_schema.json",
                     "examples": [
                         "backend_status_example.json"
                         ]})
schema_tests.append({"schema": "default_pulse_config_schema.json",
                     "examples": [
                         "default_pulse_config_example.json"
                         ]})
schema_tests.append({"schema": "job_status_schema.json",
                     "examples": [
                         "job_status_example.json"
                         ]})
schema_tests.append({"schema": "qobj_schema.json",
                     "examples": [
                         "qobj_example_openpulse.json",
                         "qobj_example_openqasm.json"
                         ]})
schema_tests.append({"schema": "result_schema.json",
                     "examples": [
                         "result_example_openqasm.json",
                         "result_example_openpulse_lev0.json",
                         "result_example_openpulse_lev1.json",
                         "result_example_snapshots.json",
                         "result_example_statevector_simulator.json",
                         "result_example_unitary_simulator.json"
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
