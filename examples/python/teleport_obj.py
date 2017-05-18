"""
Quantum teleportation example based on OPENQASM example.

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
    "name": "Program",
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

QP_program = QuantumProgram(specs=QPS_SPECS)
qc = QP_program.circuit("teleport")
q = QP_program.quantum_registers("q")
c0 = QP_program.classical_registers("c0")
c1 = QP_program.classical_registers("c1")
c2 = QP_program.classical_registers("c2")

qc.u3(0.3, 0.2, 0.1, q[0])
qc.h(q[1])
qc.cx(q[1], q[2])
qc.barrier(q)

qc.cx(q[0], q[1])
qc.h(q[0])
qc.measure(q[0], c0[0])
qc.measure(q[1], c1[0])

qc.z(q[2]).c_if(c0, 1)
qc.x(q[2]).c_if(c1, 1)
qc.measure(q[2], c2[0])

# qx5qv2 coupling
coupling_map = {0: [1, 2],
                1: [2],
                2: [],
                3: [2, 4],
                4: [2]}

# Experiment does not support feedback, so we use the simulator

result = QP_program.set_api(Qconfig.APItoken, Qconfig.config["url"])
if not result:
    print("Error setting API")
    sys.exit(1)

# First version: not compiled
result = QP_program.execute(device='simulator', coupling_map=None, shots=1024)
# print(result["compiled_circuits"][0]["qasm"])
print(result["compiled_circuits"][0]["result"]["data"]["counts"])

# Second version: compiled to qx5qv2 coupling graph
result = QP_program.execute(device='simulator', coupling_map=coupling_map, shots=1024)
# print(result["compiled_circuits"][0]["qasm"])
print(result["compiled_circuits"][0]["result"]["data"]["counts"])

# Both versions should give the same distribution
