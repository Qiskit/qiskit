# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utils for testing the standard gates."""
import os
import glob
import json
import importlib
import importlib.util
from pathlib import Path
from qiskit.tools import EquivalenceChecker
from qiskit.test.base import QiskitTestCase


class TestAlternativeChecker(QiskitTestCase):
    """Checks alternative implementation for existing circuits"""
    CURRENT_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
    CIRCUIT_LIBRARY_PATH = os.path.join(CURRENT_FILE_PATH.parents[2],
                                        "qiskit", "circuit", "library")

    def load_class_from_file(self, class_name, file_name):
        """Loads a specific class (e.g. circuit class)"""
        spec = importlib.util.spec_from_file_location(class_name, file_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)

    def load_circuit(self, dirname, filename, classname):
        """Loads a specific circuit by using its defining class"""
        full_filename = os.path.join(dirname, filename)
        circuit_class = self.load_class_from_file(classname, full_filename)
        return circuit_class()

    def test_check_for_alternatives(self):
        """Checks all suggested alternatives
        are equivalent to the original circuits"""
        for alt_dir in glob.glob(
                self.CIRCUIT_LIBRARY_PATH + '/**/alternatives',
                recursive=True):
            original_dir = str(Path(alt_dir).parents[0])
            try:
                data_filename = os.path.join(alt_dir, "data.json")
                with open(data_filename, "r") as data_file:
                    data = json.load(data_file)

                for circuit_data in data:
                    class_name = circuit_data['class_name']
                    original_filename = circuit_data['original_file']
                    alt_filename = circuit_data['alternative_file']
                    original_circuit = self.load_circuit(original_dir,
                                                         original_filename,
                                                         class_name)
                    alt_circuit = self.load_circuit(alt_dir,
                                                    alt_filename,
                                                    class_name)

                    checker = EquivalenceChecker()
                    result = checker.run(original_circuit, alt_circuit)
                    self.assertTrue(result['result'])
            except FileNotFoundError():
                pass
