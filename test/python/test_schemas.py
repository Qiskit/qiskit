# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Schemas test."""

import json
import os

from qiskit.validation.jsonschema.schema_validation import (
    validate_json_against_schema,
    _get_validator,
)
from qiskit.providers.models import (
    QasmBackendConfiguration,
    PulseBackendConfiguration,
    BackendProperties,
    BackendStatus,
    JobStatus,
    PulseDefaults,
)
import qiskit
from qiskit.result import Result
from qiskit.test import QiskitTestCase

SCHEMAS_PATH = os.path.join(os.path.dirname(os.path.abspath(qiskit.__file__)), "schemas")


class TestSchemaExamples(QiskitTestCase):
    """
    Tests schema validation
    """

    _json_examples_per_schema = {
        "qasm_backend_configuration": (
            "backend_configuration",
            [
                "backend_configuration_openqasm_example.json",
                "backend_configuration_openqasm_simulator_example.json",
            ],
        ),
        "pulse_backend_configuration": (
            "backend_configuration",
            ["backend_configuration_openpulse_example.json"],
        ),
        "backend_properties": ["backend_properties_example.json"],
        "backend_status": ["backend_status_example.json"],
        "default_pulse_configuration": ["default_pulse_configuration_example.json"],
        "job_status": ["job_status_example.json"],
        "qobj": [
            "qobj_openpulse_example.json",
            "qobj_openqasm_example.json",
            "qasm_w_pulse_gates.json",
        ],
        "result": [
            "result_openqasm_example.json",
            "result_openpulse_level_0_example.json",
            "result_openpulse_level_1_example.json",
            "result_statevector_simulator_example.json",
            "result_unitary_simulator_example.json",
        ],
    }

    def setUp(self):
        super().setUp()
        self.examples_base_path = os.path.join(SCHEMAS_PATH, "examples")

    def test_examples_are_valid(self):
        """Validate example json files against respective schemas"""
        schemas = TestSchemaExamples._json_examples_per_schema
        for test_name, examples in schemas.items():
            if isinstance(examples, tuple):
                schema_name = examples[0]
                examples = examples[1]
            else:
                schema_name = test_name

            with self.subTest(schema_test=test_name):
                for example_schema in examples:
                    with self.subTest(example=example_schema):
                        with open(
                            os.path.join(self.examples_base_path, example_schema)
                        ) as example_file:
                            example = json.load(example_file)
                            msg = (
                                "JSON failed validation of {}."
                                "Set Qiskit log level to DEBUG"
                                "for further information."
                                "".format(test_name)
                            )

                            validate_json_against_schema(example, schema_name, msg)

                        # Check for validating examples using the qiskit models.
                        obj_map = {
                            "result": Result,
                            "qasm_backend_configuration": QasmBackendConfiguration,
                            "pulse_backend_configuration": PulseBackendConfiguration,
                            "backend_properties": BackendProperties,
                            "backend_status": BackendStatus,
                            "job_status": JobStatus,
                            "default_pulse_configuration": PulseDefaults,
                        }
                        cls = obj_map.get(schema_name, None)
                        if cls and "openpulse" not in example_schema:
                            _ = cls.from_dict(example)

    def test_schemas_are_valid(self):
        """Validate that schemas are valid jsonschema"""
        schemas = TestSchemaExamples._json_examples_per_schema
        for test_name, examples in schemas.items():
            if isinstance(examples, tuple):
                schema_name = examples[0]
                examples = examples[1]
            else:
                schema_name = test_name
            with self.subTest(schema_test=schema_name):
                _get_validator(schema_name, check_schema=True)
