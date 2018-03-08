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
[What is this?]
The local_sympy_qasm_simulator backend offers a local simulator, which applies
sympy to perform the symbolic evaluation of the circuit.

[Advantage]
1 the tool obviates the manual calculation with a pen and paper, enabling
 quick adjustment of your prototype code.
2 the tool leverages sympy's symbolic computational power to keep the most
precise form of the amplitude vector, e.g., e^{I*pi/4}. Besides, the tool
leverages sympy's simplification engine to simplify the expressions as much as possible.
3 the tool supports u gates, including u1, u2, u3, cu1, cu2, cu3.

[Analysis of the results and the Limit]
1 It can simplify the expressions, including the complex ones such as sqrt(2)*I*exp(-I*pi/4)/4.
2 It may miss some simplification opportunities.
For instance, the amplitude "0.245196320100808*sqrt(2)*exp(-I*pi/4) - 0.048772580504032*sqrt(2)*I*exp(-I*pi/4)"
can be further simplified.
3 It may produce the results hard to interpret.
4 Memory error may occur if there are many qubits in the system.
This is due to the limit of classical computers and show the advantage of the quantum hardware.

"""
from __future__ import print_function
import os
from qiskit import QuantumProgram


def use_sympy_qasmsimulator():
    qprogram = QuantumProgram()
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    qasm_file = currentFolder + "/../qasm/simple.qasm"
    myqasm = qprogram.load_qasm_file(qasm_file, "my_example")
    print("analyzing: " + qasm_file)
    circuits = ['my_example']
    backend = 'local_sympy_qasm_simulator'
    result = qprogram.execute(circuits, backend=backend, shots=10, timeout=300)
    print("final quantum amplitude vector: ")
    print(result.get_data('my_example')['quantum_state'])
    print("\n")

def use_sympy_unitarysimulator():
    qprogram = QuantumProgram()
    current_folder = os.path.dirname(os.path.realpath(__file__))
    qasm_file = current_folder + "/../qasm/simple.qasm"
    my_qasm = qprogram.load_qasm_file(qasm_file, "my_example")
    print("analyzing: " + qasm_file)
    circuits = ['my_example']
    backend = 'local_sympy_unitary_simulator'
    result = qprogram.execute(circuits, backend=backend, timeout=10)
    print("unitary matrix of the circuit: ")
    print(result.get_data('my_example')['unitary'])

if __name__ == "__main__":
    use_sympy_qasmsimulator()
    use_sympy_unitarysimulator()
