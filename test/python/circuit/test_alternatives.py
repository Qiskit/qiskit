# -*- coding: utf-8 -*-

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
from qiskit.tools import EquivalenceChecker

import os
import glob
import json
import importlib
import importlib.util
from pathlib import Path
CURRENT_FILE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
CIRCUIT_LIBRARY_PATH = os.path.join(CURRENT_FILE_PATH.parents[2],
                                    "qiskit", "circuit", "library")


def load_class_from_file(class_name, file_name):
    spec = importlib.util.spec_from_file_location(class_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

def load_circuit(dirname, filename, classname):
    full_filename = os.path.join(dirname, filename)
    circuit_class = load_class_from_file(classname, full_filename)
    return circuit_class()

def check_for_alternatives():
    for alt_dir in glob.glob(CIRCUIT_LIBRARY_PATH + '/**/alternatives',
                             recursive=True):
        original_dir = str(Path(alt_dir).parents[0])
        print("Found alternative circuits for {}".format(original_dir))

        with open(os.path.join(alt_dir, "data.json"), "r") as data_file:
            data = json.load(data_file)

        for circuit_data in data:
            class_name = circuit_data['class_name']
            original_filename = circuit_data['original_file']
            alt_filename = circuit_data['alternative_file']
            original_circuit = load_circuit(original_dir, original_filename, class_name)
            alt_circuit = load_circuit(alt_dir, alt_filename, class_name)

            checker = EquivalenceChecker()
            result = checker.run(original_circuit, alt_circuit)
            print("Compared {} between {} and {}:".format(class_name, original_filename, alt_filename))
            if result['result']:
                print("Circuits equivalent, can replace")
            else:
                print("Circuits not equivalent, cannot replace")

check_for_alternatives()