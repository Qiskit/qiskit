"""
GHZ state example illustrating mapping onto the qx5qv2 device.

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

QPS_SPECS = {
    "name": "ghz",
    "circuits": [{
        "name": "ghz",
        "quantum_registers": [{
            "name": "q",
            "size": 5
        }],
        "classical_registers": [
            {"name": "c",
             "size": 5}
        ]}]
}

QP_program = QuantumProgram(specs=QPS_SPECS)
qc = QP_program.circuit("ghz")
q = QP_program.quantum_registers("q")
c = QP_program.classical_registers("c")

qc.h(q[0])
for i in range(4):
    qc.cx(q[i], q[i+1])
qc.barrier()
for i in range(5):
    qc.measure(q[i], c[i])

# qx5qv2 coupling
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

result = QP_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
if not result:
    print("Error setting API")
    sys.exit(1)

# First version: not compiled
print("no compilation, simulator")
result = QP_program.execute(device='simulator', coupling_map=None, shots=1024)
# print(result["compiled_circuits"][0]["qasm"])
print(result["compiled_circuits"][0]["result"]["data"]["counts"])

# Second version: compiled to qc5qv2 coupling graph
print("compilation to qc5qv2, simulator")
result = QP_program.execute(device='simulator', coupling_map=coupling_map, shots=1024)
# print(result["compiled_circuits"][0]["qasm"])
print(result["compiled_circuits"][0]["result"]["data"]["counts"])

# Third version: compiled to qc5qv2 coupling graph and run on qx5q
print("compilation to qc5qv2, device")
result = QP_program.execute(device='qx5q', coupling_map=coupling_map, shots=1024)
if result["status"] == "Error":
    print(result)
else:
    print(result["compiled_circuits"][0]["qasm"])
    print(result["compiled_circuits"][0]["result"]["data"]["counts"])
