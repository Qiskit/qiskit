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

n = 3

QPS_SPECS = {
    "name": "Program",
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

QP_program = QuantumProgram(specs=QPS_SPECS)
qc = QP_program.circuit("rippleadd")
a = QP_program.quantum_registers("a")
b = QP_program.quantum_registers("b")
cin = QP_program.quantum_registers("cin")
cout = QP_program.quantum_registers("cout")
ans = QP_program.classical_registers("ans")


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


# Build subcircuit to add a to b, storing result in b
adder_subcircuit = QP_program.create_circuit(
    "adder_subcircuit", [cin, a, b, cout])
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n - 1):
    majority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
adder_subcircuit.cx(a[n - 1], cout[0])
for j in reversed(range(n - 1)):
    unmajority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])

# Build the adder example
qc.x(a[0])  # Set input a = 0...0001
qc.x(b)   # Set input b = 1...1111
qc += adder_subcircuit
for j in range(n):
    qc.measure(b[j], ans[j])  # Measure the output register
qc.measure(cout[0], ans[n])

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
