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

"""Quick program to test logal simulator backends.

Author: Jay Gambetta

    python test_simulators.py ../qasm/example.qasm qasm_simulator

"""
import sys
import numpy as np
from qiskit import QuantumProgram
from qiskit.simulators._unitarysimulator import UnitarySimulator
from qiskit.simulators._qasmsimulator import QasmSimulator


seed = 88
filename = sys.argv[1]
qp = QuantumProgram()
qp.load_qasm("example", qasm_file=filename)
temp = ""

if "unitary_simulator" in sys.argv:
    a = UnitarySimulator(qp.get_qasm("example")).run()
    dim = len(a['data']['unitary'])
    print('\nUnitary simulator on State |psi> = U|0> :')
    quantum_state = np.zeros(dim, dtype=complex)
    quantum_state[0] = 1
    print(np.dot(a['data']['unitary'], quantum_state))
    temp = temp + "unitrary simulator: " + a['status'] + "\n"

if "qasm_simulator_single_shot" in sys.argv:
    print('\nUsing the qasm simulator in single shot mode: ')
    b = QasmSimulator(qp.get_qasm("example"), 1, seed).run()
    print(b['data']['quantum_state'])
    print(b['data']['classical_state'])
    temp = temp + "qasm simulator single shot: " + b['status'] + "\n"

if "qasm_simulator" in sys.argv:
    print('\nUsing the qasm simulator:')
    shots = 1024
    c = QasmSimulator(qp.get_qasm("example"), shots, seed).run()
    print(c['data']['counts'])
    temp = temp + "qasm simulator: " + c['status'] + "\n"

if not temp:
    print("No simulators entered. Please add one of\n\
           unitary_simulator\n\
           qasm_simulator\n\
           qasm_simulator_single_shot")
else:
    print(temp)
