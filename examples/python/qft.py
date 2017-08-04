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

"""
Quantum Fourier Transform examples.
"""

import sys
import os
import math

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram

import Qconfig

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = "ibmqx2"
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

###############################################################
# Make a quantum program for the GHZ state.
###############################################################
QPS_SPECS = {
    "circuits": [{
        "name": "qft3",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]},
        {
        "name": "qft4",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]},
        {
        "name": "qft5",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]}
    ]
}


def input_state(circ, q, n):
    """n-qubit input state for QFT that produces output 1."""
    for j in range(n):
        circ.h(q[j])
        circ.u1(math.pi/float(2**(j)), q[j]).inverse()


def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi/float(2**(j-k)), q[j], q[k])
        circ.h(q[j])


qp = QuantumProgram(specs=QPS_SPECS)
q = qp.get_quantum_register("q")
c = qp.get_classical_register("c")

qft3 = qp.get_circuit("qft3")
qft4 = qp.get_circuit("qft4")
qft5 = qp.get_circuit("qft5")

input_state(qft3, q, 3)
qft3.barrier()
qft(qft3, q, 3)
qft3.barrier()
for j in range(3):
    qft3.measure(q[j], c[j])

input_state(qft4, q, 4)
qft4.barrier()
qft(qft4, q, 4)
qft4.barrier()
for j in range(4):
    qft4.measure(q[j], c[j])

input_state(qft5, q, 5)
qft5.barrier()
qft(qft5, q, 5)
qft5.barrier()
for j in range(5):
    qft5.measure(q[j], c[j])

print(qft3.qasm())
print(qft4.qasm())
print(qft5.qasm())


###############################################################
# Set up the API and execute the program.
###############################################################
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])

result = qp.execute(["qft3", "qft4", "qft5"], backend='ibmqx_qasm_simulator',
                    coupling_map=coupling_map, shots=1024)
print(result)
print(result.get_ran_qasm("qft3"))
print(result.get_ran_qasm("qft4"))
print(result.get_ran_qasm("qft5"))
print(result.get_counts("qft3"))
print(result.get_counts("qft4"))
print(result.get_counts("qft5"))


result = qp.execute(["qft3"], backend=backend,
                    coupling_map=coupling_map, shots=1024, timeout=120)
print(result)
print(result.get_ran_qasm("qft3"))
print(result.get_counts("qft3"))
