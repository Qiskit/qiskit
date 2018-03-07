# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,anomalous-backslash-in-string

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
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
this python file shows how to use the symbolic backend: local_sympy_qasm_simulator
"""
from __future__ import print_function
import os
from qiskit import QuantumProgram

Q_program = QuantumProgram()
currentFolder = os.path.dirname(os.path.realpath(__file__))
qasm_file = currentFolder + "/../../../examples/qasm/naive.qasm"
myqasm = Q_program.load_qasm_file(qasm_file, "my_example")
print("analyzing: " + qasm_file)
circuits = ['my_example']
backend = 'local_sympy_qasm_simulator'
result = Q_program.execute(circuits, backend=backend, shots=10, timeout=300)
print("final quantum amplitude vector: ")
print(result.get_data('my_example')['quantum_state'])
print("\n")
