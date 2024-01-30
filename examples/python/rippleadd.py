# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Ripple adder example based on Cuccaro et al., quant-ph/0410184.

"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import transpile
from qiskit.providers.basic_provider import BasicSimulator

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = BasicSimulator()
coupling_map = [
    [0, 1],
    [0, 8],
    [1, 2],
    [1, 9],
    [2, 3],
    [2, 10],
    [3, 4],
    [3, 11],
    [4, 5],
    [4, 12],
    [5, 6],
    [5, 13],
    [6, 7],
    [6, 14],
    [7, 15],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [13, 14],
    [14, 15],
]

###############################################################
# Make a quantum program for the n-bit ripple adder.
###############################################################
n = 2

a = QuantumRegister(n, "a")
b = QuantumRegister(n, "b")
cin = QuantumRegister(1, "cin")
cout = QuantumRegister(1, "cout")
ans = ClassicalRegister(n + 1, "ans")
qc = QuantumCircuit(a, b, cin, cout, ans, name="rippleadd")


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
qc.x(b)  # Set input b = 1...1111
# Apply the adder
qc &= adder_subcircuit
# Measure the output register in the computational basis
for j in range(n):
    qc.measure(b[j], ans[j])
qc.measure(cout[0], ans[n])

###############################################################
# execute the program.
###############################################################

# First version: not mapped
job = backend.run(transpile(qc, backend=backend, coupling_map=None), shots=1024)
result = job.result()
print(result.get_counts(qc))

# Second version: mapped to 2x8 array coupling graph
job = backend.run(transpile(qc, basis_gates=["u", "cx"], coupling_map=coupling_map), shots=1024)
result = job.result()
print(result.get_counts(qc))

# Both versions should give the same distribution
