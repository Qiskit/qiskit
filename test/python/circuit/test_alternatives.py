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

def check_for_alternatives():
    for alt_dir in glob.glob(CIRCUIT_LIBRARY_PATH + '/**/alternatives',
                             recursive=True):
        original_dir = str(Path(alt_dir).parents[0])
        print("Found alternative circuits for {}".format(original_dir))

        with open(os.path.join(alt_dir, "data.json"), "r") as data_file:
            data = json.load(data_file)
        for circuit_data in data:
            class_name = circuit_data['class_name']
            original_filename = os.path.join(original_dir,circuit_data['original_file'])
            alternative_filename = os.path.join(alt_dir, circuit_data['alternative_file'])
            original_class = load_class_from_file(class_name, original_filename)
            alternative_class = load_class_from_file(class_name, alternative_filename)
            original_circuit = original_class()
            alternative_circuit = alternative_class()
            print(original_circuit)
            print(alternative_circuit)



check_for_alternatives()