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
Illustrate compiling several circuits to different backends.
"""

import sys
import os

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
# Make a quantum program for the GHZ and Bell states.
###############################################################
QPS_SPECS = {
    "circuits": [{
        "name": "ghz",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]},{
        "name": "bell",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5
        }]}
    ]
}

qp = QuantumProgram(specs=QPS_SPECS)
ghz = qp.get_circuit("ghz")
bell = qp.get_circuit("bell")
q = qp.get_quantum_register("q")
c = qp.get_classical_register("c")

# Create a GHZ state
ghz.h(q[0])
for i in range(4):
    ghz.cx(q[i], q[i+1])
# Insert a barrier before measurement
ghz.barrier()
# Measure all of the qubits in the standard basis
for i in range(5):
    ghz.measure(q[i], c[i])

# Create a Bell state
bell.h(q[0])
bell.cx(q[0], q[1])
bell.barrier()
bell.measure(q[0], c[0])
bell.measure(q[1], c[1])

print(ghz.qasm())
print(bell.qasm())

###############################################################
# Set up the API and execute the program.
###############################################################
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])

bellobj = qp.compile(["bell"], backend='local_qasm_simulator', shots=1024)
ghzobj = qp.compile(["ghz"], backend='ibmqx_qasm_simulator', shots=1024,
                    coupling_map=coupling_map)

bellresult = qp.run(bellobj)
ghzresult = qp.run(ghzobj)

print(bellresult.get_counts("bell"))
print(ghzresult.get_counts("ghz"))
