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
Quantum teleportation example based on an OpenQASM example.
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
backend = "ibmqx_qasm_simulator"
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

###############################################################
# Make a quantum program for quantum teleportation.
###############################################################
QPS_SPECS = {
    "circuits": [{
        "name": "teleport",
        "quantum_registers": [{
            "name": "q",
            "size": 3
        }],
        "classical_registers": [
            {"name": "c0",
             "size": 1},
            {"name": "c1",
             "size": 1},
            {"name": "c2",
             "size": 1},
        ]}]
}

qp = QuantumProgram(specs=QPS_SPECS)
qc = qp.get_circuit("teleport")
q = qp.get_quantum_register("q")
c0 = qp.get_classical_register("c0")
c1 = qp.get_classical_register("c1")
c2 = qp.get_classical_register("c2")

# Prepare an initial state
qc.u3(0.3, 0.2, 0.1, q[0])

# Prepare a Bell pair
qc.h(q[1])
qc.cx(q[1], q[2])

# Barrier following state preparation
qc.barrier(q)

# Measure in the Bell basis
qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], c0[0])
qc.measure(q[1], c1[0])

# Apply a correction
qc.z(q[2]).c_if(c0, 1)
qc.x(q[2]).c_if(c1, 1)
qc.measure(q[2], c2[0])

###############################################################
# Set up the API and execute the program.
###############################################################
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])

# Experiment does not support feedback, so we use the simulator

# First version: not mapped
result = qp.execute(["teleport"], backend=backend,
                    coupling_map=None, shots=1024)
print(result)
print(result.get_counts("teleport"))

# Second version: mapped to qx2 coupling graph
result = qp.execute(["teleport"], backend=backend,
                    coupling_map=coupling_map, shots=1024)
print(result)
print(result.get_ran_qasm("teleport"))
print(result.get_counts("teleport"))

# Both versions should give the same distribution
