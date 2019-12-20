# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
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
This example has assertions added as well.

"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute

###############################################################
# Set the backend name and coupling map.
###############################################################
backend = BasicAer.get_backend("qasm_simulator")
coupling_map = [[0,1], [0, 8], [1, 2], [1, 9], [2, 3], [2, 10], [3, 4], [3, 11],
                [4, 5], [4, 12], [5, 6], [5, 13], [6, 7], [6, 14], [7, 15], [8, 9],
                [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15]]

###############################################################
# Make a quantum program for the n-bit ripple adder.
###############################################################
n = 2

a = QuantumRegister(n, "a")
b = QuantumRegister(n, "b")
cin = QuantumRegister(1, "cin")
cout = QuantumRegister(1, "cout")
ans = ClassicalRegister(n+1, "ans")
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

# Empty list of all breakpoints (for assertions) to be added
breakpoints = []

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

# A breakpoint is a copy of the quantum circuit, including all its
# instructions up to that point, with an assertion at the end.
# An assertion is a measurement that performs a statistical test.
# This breakpoint is to assert that a = 1.
breakpoints.append(qc.get_breakpoint_classical([a[i] for i in range(n)], \
    [ans[i] for i in range(n)], 0.05, 1))

qc.x(b)   # Set input b = 1...1111

# Here's another breakpoint to assert that b = 1...1111
breakpoints.append(qc.get_breakpoint_classical([b[i] for i in range(n)], \
    [ans[i] for i in range(n)], 0.05, 2**n - 1))

# Apply the adder
qc += adder_subcircuit

# This breakpoint confirms successful addition: a + b = 0.
# You can comment out the line below if you don't want that
# breakpoint anymore.
breakpoints.append(qc.get_breakpoint_classical([b[i] for i in range(n)], \
    [ans[i] for i in range(n)], 0.05, 0))

# Measure the output register in the computational basis
# Note that you can replace the measurement syntax below
#for j in range(n):
#    qc.measure(b[j], ans[j])
# with the measurement syntax below
qc.measure([b[i] for i in range(n)], [ans[i] for i in range(n)])
qc.measure(cout[0], ans[n])

###############################################################
# execute the program.
###############################################################

print("\n\nFirst version: not mapped")
job = execute(breakpoints +[qc], backend=backend, coupling_map=None, shots=1024)
result = job.result()
# Show the assertion
for breakpoint in breakpoints:
    print("Results of our " + result.get_assertion_type(breakpoint) + " Assertion:")
    tup = result.get_assertion_stats(breakpoint)
    print('chisq = %f\npval = %f\npassed = %s\n' % tup)
    assert ( result.get_assertion_passed(breakpoint) )
#print(result.get_counts(qc))

print("\n\nSecond version: mapped to 2x8 array coupling graph")
job = execute(breakpoints + [qc], backend=backend, coupling_map=coupling_map, shots=1024)
result = job.result()
# Show the assertion
for breakpoint in breakpoints:
    print("Results of our " + result.get_assertion_type(breakpoint) + " Assertion:")
    tup = result.get_assertion_stats(breakpoint)
    print('chisq = %f\npval = %f\npassed = %s\n' % tup)
    assert ( result.get_assertion_passed(breakpoint) )
#print(result.get_counts(qc))

# Both versions should give the same distribution
