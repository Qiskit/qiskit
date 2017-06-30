# -*- coding: utf-8 -*-

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

"""Quick program to test local simulator backends.

From the command line give it a filename and a list of simulators to test.
The output is whatever is returned from the simulator. I would like to add one
test that runs the example.qasm

Author: Jay Gambetta

    python test_simulators.py qasm/example.qasm qasm_simulator

"""
import sys
import numpy as np
from qiskit import QuantumProgram
from qiskit.simulators._unitarysimulator import UnitarySimulator
from qiskit.simulators._qasmsimulator import QasmSimulator
import qiskit.qasm as qasm
import qiskit.unroll as unroll

seed = 88
filename = sys.argv[1]
qp = QuantumProgram()
qp.load_qasm("example", qasm_file=filename)
temp = ""

basis_gates = []  # unroll to base gates
unroller = unroll.Unroller(qasm.Qasm(data=qp.get_qasm("example")).parse(),
                           unroll.JsonBackend(basis_gates))
unroller.execute()
circuit = unroller.backend.circuit

if "unitary_simulator" in sys.argv:
    a = UnitarySimulator(circuit).run()
    dim = len(a['data']['unitary'])
    print('\nUnitary simulator on State |psi> = U|0> :')
    quantum_state = np.zeros(dim, dtype=complex)
    quantum_state[0] = 1
    print(np.dot(a['data']['unitary'], quantum_state))
    temp = temp + "unitrary simulator: " + a['status'] + "\n"

if "qasm_simulator_single_shot" in sys.argv:
    print('\nUsing the qasm simulator in single shot mode: ')
    b = QasmSimulator(circuit, 1, seed).run()
    print(b['data']['quantum_state'])
    print(b['data']['classical_state'])
    temp = temp + "qasm simulator single shot: " + b['status'] + "\n"

if "qasm_simulator" in sys.argv:
    print('\nUsing the qasm simulator:')
    shots = 1024
    c = QasmSimulator(circuit, shots, seed).run()
    print(c['data']['counts'])
    temp = temp + "qasm simulator: " + c['status'] + "\n"

if not temp:
    print("No simulators entered. Please add one of\n\
           unitary_simulator\n\
           qasm_simulator\n\
           qasm_simulator_single_shot")
else:
    print(temp)
