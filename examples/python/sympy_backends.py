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
Example use of the symbolic simulator backends, which keep precise forms of
amplitudes.

Note: if you have only cloned the QISKit repository but not
used `pip install`, the examples only work from the root directory.
"""

import os
from qiskit import QuantumProgram


def use_sympy_backends():
    qprogram = QuantumProgram()
    current_dir = os.path.dirname(os.path.realpath(__file__))
    qasm_file = current_dir + "/../qasm/simple.qasm"
    qasm_circuit = qprogram.load_qasm_file(qasm_file)
    print("analyzing: " + qasm_file)
    print(qprogram.get_qasm(qasm_circuit))
    
    # sympy statevector simulator
    backend = 'local_statevector_simulator_sympy'
    result = qprogram.execute([qasm_circuit], backend=backend, shots=1, timeout=300)
    print("final quantum amplitude vector: ")
    print(result.get_data(qasm_circuit)['statevector'])

    # sympy unitary simulator
    backend = 'local_unitary_simulator_sympy'
    result = qprogram.execute([qasm_circuit], backend=backend, shots=1, timeout=300)
    print("\nunitary matrix of the circuit: ")
    print(result.get_data(qasm_circuit)['unitary'])

if __name__ == "__main__":
    use_sympy_backends()
