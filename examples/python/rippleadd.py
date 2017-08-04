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
Ripple adder example based on Cuccaro et al, quant-ph/0410184.
"""

import sys
import os

# We don't know from where the user is running the example,
# so we need a relative position from this file path.
# TODO: Relative imports for intra-package imports are highly discouraged.
# http://stackoverflow.com/a/7506006
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from qiskit import QuantumProgram, QuantumCircuit

import Qconfig

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = "ibmqx_qasm_simulator"
coupling_map = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                11: [12], 12: [13], 13: [14], 14: [15]}

###############################################################
# Make a quantum program for the n-bit ripple adder.
###############################################################
n = 2

QPS_SPECS = {
    "circuits": [{
        "name": "rippleadd",
        "quantum_registers": [
            {"name": "a",
             "size": n},
            {"name": "b",
             "size": n},
            {"name": "cin",
             "size": 1},
            {"name": "cout",
             "size": 1}
        ],
        "classical_registers": [
            {"name": "ans",
             "size": n + 1},
        ]}]
}

qp = QuantumProgram(specs=QPS_SPECS)
qc = qp.get_circuit("rippleadd")
a = qp.get_quantum_register("a")
b = qp.get_quantum_register("b")
cin = qp.get_quantum_register("cin")
cout = qp.get_quantum_register("cout")
ans = qp.get_classical_register("ans")


def majority(p, a, b, c):
    """Majority gate."""
    p.cx(c, b)
    p.cx(c, a)
    p.ccx(a, b, c)


def unmajority(p, a, b, c):
    """Unmajority gate."""
    p.ccx(a, b, c)
    p.cx(c, a)
    p.cx(a, b)


# Build a temporary subcircuit that adds a to b,
# storing the result in b
adder_subcircuit = QuantumCircuit(cin, a, b, cout)
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n - 1):
    majority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
adder_subcircuit.cx(a[n - 1], cout[0])
for j in reversed(range(n - 1)):
    unmajority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])

# Set the inputs to the adder
qc.x(a[0])  # Set input a = 0...0001
qc.x(b)   # Set input b = 1...1111
# Apply the adder
qc += adder_subcircuit
# Measure the output register in the computational basis
for j in range(n):
    qc.measure(b[j], ans[j])
qc.measure(cout[0], ans[n])

###############################################################
# Set up the API and execute the program.
###############################################################
qp.set_api(Qconfig.APItoken, Qconfig.config["url"])

# First version: not mapped
result = qp.execute(["rippleadd"], backend=backend,
                    coupling_map=None, shots=1024)
print(result)
print(result.get_counts("rippleadd"))

# Second version: mapped to 2x8 array coupling graph
obj = qp.compile(["rippleadd"], backend=backend,
                 coupling_map=coupling_map, shots=1024)
result = qp.run(obj)

print(result)
print(result.get_ran_qasm("rippleadd"))
print(result.get_counts("rippleadd"))

# Both versions should give the same distribution
