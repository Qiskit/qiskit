"""
Ripple adder example based on OPENQASM example.

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


def swap(qc, q0, q1):
    """Swap gate."""
    qc.cx(q0, q1)
    qc.cx(q1, q0)
    qc.cx(q0, q1)

n = 3

QPS_SPECS = {
    "name": "Program",
    "circuits": [{
        "name": "test16",
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

QP_program = QuantumProgram(specs=QPS_SPECS)
qc = QP_program.circuit("test16")
q = QP_program.quantum_registers("q")
r = QP_program.quantum_registers("r")
ans = QP_program.classical_registers("ans")

qc.x(q[0])  # Set input q = 1...0000

swap(qc, q[0], q[n-1])
swap(qc, q[n-1], r[n-1])

for j in range(n):
    qc.measure(q[j], ans[j])
    qc.measure(r[j], ans[j+n])

# 2x8 array
coupling_map = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                11: [12], 12: [13], 13: [14], 14: [15]}

result = QP_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
if not result:
    print("Error setting API")
    sys.exit(1)

# First version: not compiled
result = QP_program.execute(device='simulator', coupling_map=None, shots=1024)
# print(result["compiled_circuits"][0]["qasm"])
print(result["compiled_circuits"][0]["result"]["data"]["counts"])

# Second version: compiled to 2x8 array coupling graph
result = QP_program.execute(device='simulator', coupling_map=coupling_map, shots=1024)
# print(result["compiled_circuits"][0]["qasm"])
print(result["compiled_circuits"][0]["result"]["data"]["counts"])

# Both versions should give the same distribution
