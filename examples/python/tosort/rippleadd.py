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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import qiskit
# Work in progress
import qiskit.unroll as unroll
import qiskit.mapper as mapper
from qiskit.qasm import Qasm


def print_2x8(layout):
    """Print a 2x8 layout."""
    rev_layout = {b: a for a, b in layout.items()}
    print("")
    for i in range(8):
        qubit = ("q", i)
        if qubit in rev_layout:
            print("%s[%d] " % (rev_layout[qubit][0], rev_layout[qubit][1]),
                  end="")
        else:
            print("XXXX ", end="")
    print("")
    for i in range(8, 16):
        qubit = ("q", i)
        if qubit in rev_layout:
            print("%s[%d] " % (rev_layout[qubit][0], rev_layout[qubit][1]),
                  end="")
        else:
            print("XXXX ", end="")
    print("")


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


n = 4
a = qiskit.QuantumRegister("a", n)
b = qiskit.QuantumRegister("b", n)
cin = qiskit.QuantumRegister("cin", 1)
cout = qiskit.QuantumRegister("cout", 1)
ans = qiskit.ClassicalRegister("ans", n + 1)

# Build subcircuit to add a to b, storing result in b
adder_subcircuit = qiskit.QuantumCircuit(cin, a, b, cout)
majority(adder_subcircuit, cin[0], b[0], a[0])
for j in range(n - 1):
    majority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
adder_subcircuit.cx(a[n - 1], cout[0])
for j in reversed(range(n - 1)):
    unmajority(adder_subcircuit, a[j], b[j + 1], a[j + 1])
unmajority(adder_subcircuit, cin[0], b[0], a[0])

# Build the adder example
qc = qiskit.QuantumCircuit(cin, a, b, cout, ans)
qc.x(a[0])  # Set input a = 0...0001
qc.x(b)   # Set input b = 1...1111
qc += adder_subcircuit
for j in range(n):
    qc.measure(b[j], ans[j])  # Measure the output register
qc.measure(cout[0], ans[n])

######################################################################
# Map the QuantumCircuit to the 2x8 device
######################################################################

print("QuantumCircuit OPENQASM")
print("-----------------------")
print(qc.qasm())

# Unroll this now just for the purpose of gate counting
u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["cx", "x", "ccx", "id"]))
u.execute()
C = u.backend.circuit

print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())
print("operations:")
for key, val in C.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# First pass: expand subroutines to a basis of 1 and 2 qubit gates.
######################################################################

u = unroll.Unroller(Qasm(data=qc.qasm()).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx", "id"]))
u.execute()
C = u.backend.circuit

print("")
print("Unrolled OPENQASM to QX basis")
print("-------------------------------------------")
# print(C.qasm(qeflag=True))

# print("")
print("size    = %d" % C.size())
print("depth   = %d" % C.depth())
print("width   = %d" % C.width())
print("bits    = %d" % C.num_cbits())
print("factors = %d" % C.num_tensor_factors())
print("operations:")
for key, val in C.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Second pass: choose a layout on the coupling graph and add SWAPs.
######################################################################

# This is a 2 by 8 array of 16 qubits
couplingdict = {0: [1, 8], 1: [2, 9], 2: [3, 10], 3: [4, 11], 4: [5, 12],
                5: [6, 13], 6: [7, 14], 7: [15], 8: [9], 9: [10], 10: [11],
                11: [12], 12: [13], 13: [14], 14: [15]}

# This is all-all.
# couplingdict = {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                1: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                2: [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#                3: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
#                4: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
#                5: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
#                6: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11],
#                7: [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
#                8: [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11],
#                9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11],
#                10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
#                11: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

coupling = mapper.Coupling(couplingdict)
C_mapped, layout = mapper.swap_mapper(C, coupling)

print("")
print("Initial qubit positions on 2x8 layout")
print("-------------------------------------------")
print_2x8(layout)

print("")
print("Inserted SWAP gates for 2x8 layout")
print("-------------------------------------------")
# print(C_mapped.qasm(qeflag=True))

# print("")
print("size    = %d" % C_mapped.size())
print("depth   = %d" % C_mapped.depth())
print("width   = %d" % C_mapped.width())
print("bits    = %d" % C_mapped.num_cbits())
print("factors = %d" % C_mapped.num_tensor_factors())
print("operations:")
for key, val in C_mapped.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Third pass: expand SWAP subroutines and adjust cx gate orientations.
######################################################################
u = unroll.Unroller(Qasm(data=C_mapped.qasm(qeflag=True)).parse(),
                    unroll.CircuitBackend(["u1", "u2", "u3", "cx", "id"]))
u.execute()
C_mapped_unrolled = u.backend.circuit
C_directions = mapper.direction_mapper(C_mapped_unrolled, coupling)

print("")
print("Changed CNOT directions")
print("-------------------------------------------")
# print(C_directions.qasm(qeflag=True))

# print("")
print("size    = %d" % C_directions.size())
print("depth   = %d" % C_directions.depth())
print("width   = %d" % C_directions.width())
print("bits    = %d" % C_directions.num_cbits())
print("factors = %d" % C_directions.num_tensor_factors())
print("operations:")
for key, val in C_directions.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Fourth pass: collect runs of cx gates and cancel them.
######################################################################
mapper.cx_cancellation(C_directions)

print("")
print("Cancelled redundant CNOT gates")
print("-------------------------------------------")
# print(C_directions.qasm(qeflag=True))

# print("")
print("size    = %d" % C_directions.size())
print("depth   = %d" % C_directions.depth())
print("width   = %d" % C_directions.width())
print("bits    = %d" % C_directions.num_cbits())
print("factors = %d" % C_directions.num_tensor_factors())
print("operations:")
for key, val in C_directions.count_ops().items():
    print("  %s: %d" % (key, val))

######################################################################
# Fifth pass: expand single qubit gates to u1, u2, u3 and simplify.
######################################################################
C_directions_unrolled = mapper.optimize_1q_gates(C_directions)

print("")
print("Cancelled redundant single qubit gates")
print("-------------------------------------------")
print(C_directions_unrolled.qasm(qeflag=True))

print("")
print("size    = %d" % C_directions_unrolled.size())
print("depth   = %d" % C_directions_unrolled.depth())
print("width   = %d" % C_directions_unrolled.width())
print("bits    = %d" % C_directions_unrolled.num_cbits())
print("factors = %d" % C_directions_unrolled.num_tensor_factors())
print("operations:")
for key, val in C_directions_unrolled.count_ops().items():
    print("  %s: %d" % (key, val))
