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
Simple test of the mapper on an example that swaps a "1" state.

Author: Andrew Cross
        Jesus Perez <jesusper@us.ibm.com>
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
# Set the device name and coupling map.
###############################################################
device = "simulator"
coupling_map = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                11: [12], 12: [13], 13: [14], 14: [15]}

###############################################################
# Make a quantum program using some swap gates.
###############################################################
def swap(qc, q0, q1):
    """Swap gate."""
    qc.cx(q0, q1)
    qc.cx(q1, q0)
    qc.cx(q0, q1)


n = 3  # make this at least 3

QPS_SPECS = {
    "name": "Program",
    "circuits": [{
        "name": "swapping",
        "quantum_registers": [
            {"name": "q",
             "size": n},
            {"name": "r",
             "size": n}
        ],
        "classical_registers": [
            {"name": "ans",
             "size": 2*n},
        ]}]
}

qp = QuantumProgram(specs=QPS_SPECS)
qc = qp.get_circuit("swapping")
q = qp.get_quantum_registers("q")
r = qp.get_quantum_registers("r")
ans = qp.get_classical_registers("ans")

# Set the first bit of q
qc.x(q[0])

# Swap the set bit
swap(qc, q[0], q[n-1])
swap(qc, q[n-1], r[n-1])
swap(qc, r[n-1], q[1])
swap(qc, q[1], r[1])

# Insert a barrier before measurement
qc.barrier()
# Measure all of the qubits in the standard basis
for j in range(n):
    qc.measure(q[j], ans[j])
    qc.measure(r[j], ans[j+n])

###############################################################
# Set up the API and execute the program.
###############################################################
result = qp.set_api(Qconfig.APItoken, Qconfig.config["url"])
if not result:
    print("Error setting API")
    sys.exit(1)

# First version: not compiled
result = qp.execute(["swapping"], device=device, coupling_map=None, shots=1024)
print(qp.get_compiled_qasm("swapping"))
print(qp.get_counts("swapping"))

# Second version: compiled to coupling graph
result = qp.execute(["swapping"], device=device, coupling_map=coupling_map, shots=1024)
print(qp.get_compiled_qasm("swapping"))
print(qp.get_counts("swapping"))

# Both versions should give the same distribution
